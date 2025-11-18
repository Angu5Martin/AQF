import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import pytorch_lightning as pl

def qlike_loss(y_hat, y):
    return torch.mean(torch.log(y_hat + 1e-8) + y / (y_hat + 1e-8))

def mae_metric(y_hat, y):
    return torch.mean(torch.abs(y - y_hat))


def rmse_metric(y_hat, y):
    return torch.mean(torch.sqrt(torch.mean((y - y_hat) ** 2, dim=0)))


def smape_metric(y_hat, y):
    return torch.mean(2 * torch.abs(y - y_hat) / (y + y_hat))


def me_metric(y_hat, y):
    return torch.mean(torch.max(torch.abs(y - y_hat), dim=0).values)


def medae_metric(y_hat, y):
    return torch.mean(torch.median(torch.abs(y - y_hat), dim=0).values)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # manual padding for causality
        )

    def forward(self, x):
        # x: (B, C, T)
        pad_len = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_len, 0))  # pad only on the left (causal)
        return self.conv(x)


class TimeAttention(nn.Module):
    """Additive (Bahdanau-style) attention over the time dimension for one layer.

    Input: tensor of shape (B, C, T)
    - applies a small MLP on the channel dimension at each time step and
      produces attention scores over T, then weighted-sums to (B, C).
    Output: (context (B, C), attn_weights (B, T))
    """

    def __init__(self, channels, hidden_size):
        super().__init__()
        self.W = nn.Linear(channels, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        # x: (B, C, T) -> work on (B, T, C)
        # b, c, t = x.shape
        H = x.permute(0, 2, 1)  # (B, T, C)
        # score each time step
        scores = self.v(torch.tanh(self.W(H)))  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        attn = torch.softmax(scores, dim=1)  # (B, T)
        attn_exp = attn.unsqueeze(-1)  # (B, T, 1)
        context = (H * attn_exp).sum(dim=1)  # (B, C)
        return context, attn
    
class TimeGate(nn.Module):
    def __init__(self, channels, hidden_size):
        super().__init__()
        self.W = nn.Linear(channels, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, C, T)
        H = x.permute(0, 2, 1)    # (B,T,C)
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)   # (B,T)
        attn = torch.sigmoid(scores)                         # (B,T)
        out = x * attn.unsqueeze(1)                          # (B,C,T)
        return out



class ResidualBlock(nn.Module):
    """Residual block without gated filter/gate. Uses a single dilated causal conv
    that expands to `dilation_channels`, then produces a residual (1x1 -> residual_channels)
    and a skip (1x1 -> skip_channels). Additionally, we apply a TimeAttention
    module per block on the skip features across time to get a per-layer context.
    """

    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, hidden_attn=32, dropout=0.0):
        super().__init__()
        # single dilated conv (no gating)
        self.dilated_conv = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)
        self.activation = nn.ReLU()

        # projections
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)

        # attention over time for this layer, applied to skip features
        self.time_attn = TimeGate(residual_channels, hidden_attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, residual_channels, T)
        z = self.time_attn(x)
        z = self.dilated_conv(z)  # (B, dilation_channels, T)
        z = self.activation(z)
        # z = self.dropout(z)

        # residual path (per-time)
        residual = self.residual_conv(z)  # (B, residual_channels, T)

        # skip features (per-time)
        skip_time = self.skip_conv(z)  # (B, skip_channels, T)
        # Apply ReLU nonlinearity to skip features (sigma_ReLU)
        x = x + residual

        return x, skip_time
    

class DilatedCausalCNN(pl.LightningModule):
    """Dilated causal CNN with per-layer temporal attention implementing
    the form: sigma^2_{T+1} = alpha0 + sum_l sigma_ReLU(F^{(l)}),
    where each F^{(l)} is first time-attended (within layer) and then
    summed across layers. The per-layer 1x1 skip projections provide
    learnable scaling before summation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        residual_channels=64,
        dilation_channels=128,
        skip_channels=128,
        end_channels=64,
        kernel_size=3,
        num_blocks=2,
        num_layers=6,
        hidden_attn=32,
        dropout=0.0,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                self.blocks.append(
                    ResidualBlock(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )

        # final projection from aggregated layer contexts to output
        # end-processing modules for Option A
        self.post1 = nn.Sequential(nn.Conv1d(skip_channels, end_channels, kernel_size=1), nn.ReLU())
        self.post2 = nn.Sequential(nn.Conv1d(end_channels, out_channels, kernel_size=1), nn.ReLU())
        self.final_time_attn = TimeAttention(out_channels, hidden_attn)

        self.alpha0 = nn.Parameter(torch.zeros(out_channels))
        self.lr = lr
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        layer_skips = []  # collect (B, skip_channels, T)
        for block in self.blocks:
                    x, skip_time_relu = block(x)
                    layer_skips.append(skip_time_relu)


        # Option A: sum skip_time across layers (B, C, T)
        # skip_sum = torch.stack(layer_skips, dim=0).sum(dim=0)
        skip_sum = torch.stack(layer_skips, dim=0).sum(dim=0)  # (B, skip_channels, T)
        skip_sum = F.relu(skip_sum)
        post = self.post1(skip_sum)
        post = self.post2(post)

        # final time attention
        context, _ = self.final_time_attn(post)
        out = context + self.alpha0
        # assert torch.all(out >= 0), "Final context has negative values!"
        return out
    
    def compute_metrics(self, y_hat, y):
        metrics = {
        'qlike': qlike_loss(y_hat, y),
        'mae': mae_metric(y_hat, y),
        'rmse': rmse_metric(y_hat, y),
        'smape': smape_metric(y_hat, y),
        'me': me_metric(y_hat, y),
        'medae': medae_metric(y_hat, y)
        }
        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = qlike_loss(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        metrics = self.compute_metrics(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        for k, v in metrics.items():
            self.log(f"train_{k}", v, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = qlike_loss(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        metrics = self.compute_metrics(y_hat, y)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = qlike_loss(y_hat, y)
        # loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        metrics = self.compute_metrics(y_hat, y)
        for k, v in metrics.items():
            self.log(f"test_{k}", v, on_epoch=True, on_step=False)
        # Collect outputs in an instance attribute for use in the on_test_epoch_end hook.
        if not hasattr(self, "_collected_test_outputs"):
            self._collected_test_outputs = []
        # detach and move to cpu later when aggregating
        self._collected_test_outputs.append({"y_hat": y_hat.detach(), "y": y.detach()})
    
    # def test_epoch_end(self, outputs):
    #     y_hat = torch.cat([o["y_hat"] for o in outputs], dim=0)
    #     y     = torch.cat([o["y"]     for o in outputs], dim=0)

    #     # store for external usage
    #     self.test_preds = y_hat.cpu()
    #     self.test_targets = y.cpu()

    def on_test_epoch_end(self) -> None:
        """PL v2 hook replacement for the removed `test_epoch_end`.
        Aggregate outputs previously saved in `self._collected_test_outputs`.
        """
        outputs = getattr(self, "_collected_test_outputs", None)
        if not outputs:
            return
        y_hat = torch.cat([o["y_hat"] for o in outputs], dim=0)
        y     = torch.cat([o["y"]     for o in outputs], dim=0)
        # store for external usage (move to cpu)
        self.test_preds = y_hat.cpu()
        self.test_targets = y.cpu()
        # clean up
        del self._collected_test_outputs


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

class StockVolDataset(Dataset):
    def __init__(self, df: pd.DataFrame, num_days: int = 1, mode="univariate", return_timestamps=False, return_tickers=False):
        """
        Args:
            df: pandas DataFrame with DateTimeIndex (intraday frequency), columns=tickers, values=prices
            num_days: length of observation window in days
        """
        assert isinstance(df.index, pd.DatetimeIndex)
        assert mode in ["multivariate", "univariate"], "mode must be 'multivariate' or 'univariate'"
        self.num_days = num_days
        self.mode = mode

        # compute log-returns (keep as DataFrame for convenience)
        self.logret_df = np.log(df / df.shift(1))

        # group by unique calendar days
        self.days = sorted(set(self.logret_df.index.normalize()))
        self.samples = []  # list of (X_numpy, y_numpy)
        # store per-sample metadata helpful for normalization
        self.timestamps = [] if return_timestamps else None
        self.tickers = [] if return_tickers else None

        for i in range(num_days, len(self.days)):
            # input window: last num_days of log-returns (inclusive of start_day, exclusive of end_day)
            start_day = self.days[i - num_days]
            end_day = self.days[i]
            # select half-open interval [start_day, end_day) to avoid leakage
            X_df = self.logret_df.loc[start_day:end_day].iloc[1:]


            # next-day realized variance: intraday returns for day `end_day`
            # intraday_next = self.logret_df[(self.logret_df.index >= end_day) & (self.logret_df.index < end_day + pd.Timedelta(days=1))]
            intraday_next = self.logret_df.loc[end_day:end_day + pd.Timedelta(days=0.5)]
            # compute daily realized variance (sum of squared returns) — keep as numpy
            y = 1e4 * (intraday_next ** 2).sum(axis=0).values  # shape (num_assets,)

            X = X_df.values  # shape (T_window, num_assets)

            if self.mode == "multivariate":
                self.samples.append((X, y))
                if return_timestamps:
                    self.timestamps.append(start_day)
            elif self.mode == "univariate":
                # for univariate mode: create one sample per asset (column)
                n_assets = X.shape[1]
                for c in range(n_assets):
                    self.samples.append((X[:, c:c+1], y[c:c+1]))
                    if return_timestamps:
                        self.timestamps.append(start_day)
                    if return_tickers:
                        self.tickers.append(df.columns[c])

        # placeholders for normalization (set from DataModule)
        self._X_mean = None  # numpy array shape (num_assets,)
        self._X_std = None

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Provide per-asset mean/std (numpy arrays) computed on training data only."""
        self._X_mean = np.asarray(mean, dtype=np.float32)
        self._X_std = np.asarray(std, dtype=np.float32)
        # avoid zero division
        self._X_std[self._X_std == 0] = 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        # X: (T_window, num_assets) or (T_window, 1) in univariate
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # apply normalization if available
        if self._X_mean is not None and self._X_std is not None:
            mean = torch.tensor(self._X_mean, dtype=torch.float32)
            std = torch.tensor(self._X_std, dtype=torch.float32)
            X = (X - mean.unsqueeze(0)) / std.unsqueeze(0)
            

        return X, y


class StockVolDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, val_start_date, test_start_date, num_days: int = 1, mode = "univariate", ticker_split = False, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.df = df
        self.val_start_date = val_start_date
        self.test_start_date = test_start_date
        self.num_days = num_days
        self.mode = mode
        self.ticker_split = ticker_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # build full dataset with metadata enabled so we can compute train statistics
        self.full_dataset = StockVolDataset(self.df, num_days=self.num_days, mode=self.mode, return_timestamps=True, return_tickers=(self.mode == "univariate"))

        timestamps = pd.to_datetime(self.full_dataset.timestamps)

        train_mask = timestamps < self.val_start_date
        val_mask = (timestamps >= self.val_start_date) & (timestamps < self.test_start_date)
        test_mask = timestamps >= self.test_start_date

        # optional ticker split (univariate mode)
        train_tickers = None
        if self.mode == "univariate" and self.ticker_split:
            tickers = np.array(self.full_dataset.tickers)
            unique_tickers = np.unique(tickers)
            rng = np.random.default_rng(256)
            rng.shuffle(unique_tickers)
            mid = len(unique_tickers) // 2
            train_tickers = unique_tickers[:mid]
            test_tickers = unique_tickers[mid:]

            train_ticker_mask = np.isin(tickers, train_tickers)
            test_ticker_mask = np.isin(tickers, test_tickers)

            train_mask = train_mask & train_ticker_mask
            val_mask = val_mask & train_ticker_mask
            test_mask = test_mask & test_ticker_mask

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        # --- compute normalization stats from training samples only ---
        num_assets = self.df.shape[1]


        # concatenate X matrices from training samples along time axis
        X_list = [self.full_dataset.samples[i][0] for i in train_idx]
        if len(X_list) == 0:
            raise ValueError("No training samples found when computing normalization stats")
        X_cat = np.concatenate(X_list, axis=0)  # (N_total_time, num_assets)
        mean = X_cat.mean(axis=0)
        std = X_cat.std(axis=0)

        # set normalization on the underlying full dataset (Subsets will inherit it)
        self.full_dataset.set_normalization(mean, std)

        # create subsets
        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)
        self.test_dataset = Subset(self.full_dataset, test_idx)
        
        self.test_tickers = np.array(self.full_dataset.tickers)[test_idx] if self.mode == "univariate" else None

        print(f"Dataset split: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class LossHistory(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses.append(loss.item())
