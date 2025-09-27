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

from typing import List, Optional, Tuple

import math
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class CausalConvND(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.left_pad, 0), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, ...)
        b, t, c, *spatial = x.shape
        rest = int(torch.tensor(spatial).prod().item()) if spatial else 1
        x = x.reshape(b * rest, t, c).transpose(1, 2)  # (B*rest, C, T)
        x = self.pad(x)
        x = self.conv(x)  # (B*rest, C_out, T)
        x = x.transpose(1, 2).reshape(b, t, -1, *spatial)
        return x


class DilatedResidualBlock(nn.Module):
    def __init__(self, residual_channels: int, skip_channels: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.conv = CausalConvND(residual_channels, residual_channels, kernel_size=kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C, ...)
        b, t, c, *spatial = x.shape
        rest = int(torch.tensor(spatial).prod().item()) if spatial else 1

        out = self.conv(x)  # (B, T, C, ...)
        out = self.relu(out)
        out = self.dropout(out)

        # project residual
        res = out.reshape(b * rest, t, c).transpose(1, 2)
        res = self.res_conv(res)
        res = res.transpose(1, 2).reshape(b, t, c, *spatial)

        # project skip
        skip = out.reshape(b * rest, t, c).transpose(1, 2)
        skip = self.skip_conv(skip)  # (B*rest, skip_channels, T)
        skip = skip.transpose(1, 2).reshape(b, t, -1, *spatial)

        return x + res, skip


class DilatedCausalCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        residual_channels: int = 64,
        skip_channels: int = 128,
        end_channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 2,
        num_layers: int = 6,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.front_conv = CausalConvND(in_channels, residual_channels, kernel_size=1)

        blocks: List[nn.Module] = []
        for b in range(num_blocks):
            for i in range(num_layers):
                dilation = 2 ** i
                blocks.append(
                    DilatedResidualBlock(residual_channels, skip_channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
                )
        self.blocks = nn.ModuleList(blocks)

        self.out_proj1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.out_proj2 = nn.Conv1d(end_channels, out_channels, kernel_size=1)

        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, *spatial = x.shape
        h = self.front_conv(x)  # (B, T, residual_channels, ...)

        skips = []
        for block in self.blocks:
            h, skip = block(h)
            skips.append(skip)

        # sum over skip connections
        skip_sum = sum(skips)

        rest = int(torch.tensor(spatial).prod().item()) if spatial else 1
        skip_sum = skip_sum.reshape(b * rest, t, -1).transpose(1, 2)  # (B*rest, skip_channels, T)

        out = F.relu(self.out_proj1(skip_sum))
        out = self.out_proj2(out)

        out = out.transpose(1, 2).reshape(b, t, -1, *spatial)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class SineDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 256, freq_range: Tuple[float, float] = (0.1, 1.0), noise_std: float = 0.1):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.freq_range = freq_range
        self.noise_std = noise_std
        self.data = []
        for _ in range(num_samples):
            f = float(torch.empty(1).uniform_(freq_range[0], freq_range[1]).item())
            phase = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
            t = torch.arange(seq_len, dtype=torch.float32)
            signal = torch.sin(2 * math.pi * f * (t / seq_len) + phase)
            signal += 0.2 * torch.sin(2 * math.pi * (f * 3) * (t / seq_len) + phase * 0.5)
            signal = signal.unsqueeze(-1)  # (1, T, 1)
            noise = torch.randn_like(signal) * noise_std
            sample = signal + noise
            self.data.append(sample)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.data[idx]
        y = x.clone()
        return x, y


class SineDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, seq_len: int = 256, num_samples: int = 2000, val_split: float = 0.1, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        ds = SineDataset(num_samples=self.num_samples, seq_len=self.seq_len)
        val_len = int(self.num_samples * self.val_split)
        train_len = self.num_samples - val_len
        self.train_ds, self.val_ds = random_split(ds, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


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


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--residual_channels", type=int, default=64)
    parser.add_argument("--skip_channels", type=int, default=128)
    parser.add_argument("--end_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    datamodule = SineDataModule(batch_size=args.batch_size, seq_len=args.seq_len, num_samples=args.num_samples)
    model = DilatedCausalCNN(
        in_channels=1,
        out_channels=1,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        kernel_size=args.kernel_size,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

    history = LossHistory()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        enable_checkpointing=False,
        callbacks=[early_stop, history],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=datamodule)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(history.train_losses, label="Training Loss")
    plt.plot(history.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()



# if __name__ == "__main__":
#     cli_main()
