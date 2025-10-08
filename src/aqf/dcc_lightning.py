"""
Dilated causal convolutional network with explicit skip connections (WaveNet-style)
implemented using PyTorch Lightning.

Features:
- Causal N-D dilated convolution blocks (time dimension first: (B, T, ...))
- Residual + skip connections (separate skip_channels argument)
- End channels for post-skip processing
- Support for multiple blocks × layers
- Early stopping callback in Trainer
- PyTorch LightningModule wrapper with optimizer/scheduler
- Synthetic sine wave dataset and Lightning DataModule for quick testing
- Example `if __name__ == '__main__'` training run

Requirements:
- torch
- pytorch-lightning

Install: pip install torch pytorch-lightning
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import pytorch_lightning as pl

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


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.filter_conv = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, dilation_channels, kernel_size, dilation)

        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, residual_channels, T)
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        z = filter_out * gate_out
        z = self.dropout(z)

        residual = self.residual_conv(z)
        skip = self.skip_conv(z)

        # residual connection (same shape)
        x = x + residual

        return x, skip

class LayerwiseAttention(nn.Module):
    def __init__(self, skip_channels, hidden_size):
        super().__init__()
        # Bahdanau-style additive attention
        self.W = nn.Linear(skip_channels, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)


    def forward(self, layer_skips):
    # layer_skips: list of tensors (B, C, T)
        pooled = [s.mean(dim=2) for s in layer_skips] # (B, C)
        H = torch.stack(pooled, dim=1) # (B, L, C)


        scores = self.v(torch.tanh(self.W(H))) # (B, L, 1)
        attn_weights = torch.softmax(scores, dim=1) # (B, L, 1)


        context = (H * attn_weights).sum(dim=1) # (B, C)
        return context, attn_weights.squeeze(-1)

class DilatedCausalCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual_channels=64,
        dilation_channels=64,
        skip_channels=128,
        end_channels=64,  # hidden size for attention scoring
        kernel_size=3,
        num_blocks=2,
        num_layers=6,
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

        # Layer-wise attention over skip connections
        self.attention = LayerwiseAttention(skip_channels, end_channels)

        # Final projection
        self.fc_out = nn.Linear(skip_channels, out_channels)

        self.lr = lr

    @staticmethod
    def qlike_loss(y_true, y_pred, eps=1e-12):
        y_hat = torch.clamp(y_pred, min=eps)
        ratio = y_true / y_hat
        loss = ratio - torch.log(ratio) - 1
        return loss.mean()
    
    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        skip_outputs = []
        for block in self.blocks:
            x, skip = block(x)
            skip_outputs.append(F.relu(skip))

        context, attn_weights = self.attention(skip_outputs)
        out = self.fc_out(context)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = self.qlike_loss(y, y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = self.qlike_loss(y, y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # loss = self.qlike_loss(y, y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

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

        # compute log-returns
        self.logret = np.log(df / df.shift(1))

        # group by unique calendar days
        self.days = sorted(set(self.logret.index.normalize()))
        self.samples = []
        self.timestamps = [] if return_timestamps else None
        self.tickers = [] if return_tickers else None

        for i in range(num_days, len(self.days)):
            # input window: last num_days of log-returns
            start_day = self.days[i - num_days]
            end_day = self.days[i]
            X = self.logret.loc[start_day:end_day].iloc[1:]  # drop first NaN row

            # realized vol = sqrt(sum intraday logret^2)) for next day
            intraday_next = self.logret.loc[end_day:end_day + pd.Timedelta(days=0.5)]
            y = np.sqrt((intraday_next ** 2).sum(axis=0))

            if self.mode == "multivariate":
                self.samples.append((X.values, y.values))
                if return_timestamps: 
                    self.timestamps.append(start_day)
            elif self.mode == "univariate":
                for c in range(X.shape[1]):
                    self.samples.append((X.values[:, c:c+1], y.values[c:c+1]))
                    if return_timestamps: 
                        self.timestamps.append(start_day)
                    if return_tickers:
                        self.tickers.append(c)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        # X: (T_window, num_stocks) → (T_window, C)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)  # (C,)
        return X, y


class StockVolDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, val_start_date, test_start_date, num_days: int = 1, mode = "multivariate", ticker_split = False, batch_size: int = 32):
        super().__init__()
        self.df = df
        self.val_start_date = val_start_date
        self.test_start_date = test_start_date
        self.num_days = num_days
        self.mode = mode
        self.ticker_split = ticker_split
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.full_dataset = StockVolDataset(self.df, num_days=self.num_days, mode=self.mode, return_timestamps=True, return_tickers=(self.mode == "univariate"))
        timestamps = pd.to_datetime(self.full_dataset.timestamps)

        train_mask = timestamps < self.val_start_date
        val_mask = (timestamps >= self.val_start_date) & (timestamps < self.test_start_date)
        test_mask = timestamps >= self.test_start_date

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

        self.train_dataset = Subset(self.full_dataset, train_idx)
        self.val_dataset = Subset(self.full_dataset, val_idx)
        self.test_dataset = Subset(self.full_dataset, test_idx)

        print(f"Dataset split: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


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


# def cli_main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--seq_len", type=int, default=256)
#     parser.add_argument("--num_samples", type=int, default=2000)
#     parser.add_argument("--residual_channels", type=int, default=64)
#     parser.add_argument("--skip_channels", type=int, default=128)
#     parser.add_argument("--end_channels", type=int, default=64)
#     parser.add_argument("--num_blocks", type=int, default=2)
#     parser.add_argument("--num_layers", type=int, default=6)
#     parser.add_argument("--kernel_size", type=int, default=3)
#     parser.add_argument("--dropout", type=float, default=0.0)
#     parser.add_argument("--max_epochs", type=int, default=5)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--patience", type=int, default=10)
#     args = parser.parse_args()

#     datamodule = SineDataModule(batch_size=args.batch_size, seq_len=args.seq_len, num_samples=args.num_samples)
#     model = DilatedCausalCNN(
#         in_channels=1,
#         out_channels=1,
#         residual_channels=args.residual_channels,
#         skip_channels=args.skip_channels,
#         end_channels=args.end_channels,
#         kernel_size=args.kernel_size,
#         num_blocks=args.num_blocks,
#         num_layers=args.num_layers,
#         dropout=args.dropout,
#         lr=args.lr,
#     )

#     early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

#     history = LossHistory()
#     trainer = pl.Trainer(
#         max_epochs=args.max_epochs,
#         enable_checkpointing=False,
#         callbacks=[early_stop, history],
#         log_every_n_steps=1,
#     )

#     trainer.fit(model, datamodule=datamodule)

#     # Plot
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8, 5))
#     plt.plot(history.train_losses, label="Training Loss")
#     plt.plot(history.val_losses, label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training & Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.show()



# if __name__ == "__main__":
#     cli_main()
