# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import warnings
import numpy as np
import torch
import os


from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


import importlib
import src.aqf.data_access
import src.aqf.tickers
import src.aqf.dcc_lightning
importlib.reload(src.aqf.data_access)
importlib.reload(src.aqf.tickers)
importlib.reload(src.aqf.dcc_lightning)
# from src.aqf.data_access import FirstRateDataClient
# from src.aqf.tickers_full import NASDAQ_TICKERS
from src.aqf.dcc_lightning import StockVolDataModule, DilatedCausalCNN, LossHistory

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Configure plotly
import plotly.io as pio
pio.renderers.default = 'notebook'

def scatter_pred_vs_real(y_true, y_pred, data_type, dates, tickers, save_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    N, A = y_true.shape
    if tickers is None:
        tickers = [f"Asset {i}" for i in range(A)]

    plt.figure(figsize=(7, 6))
    if data_type == 'multivariate':
        colors = plt.cm.tab20(np.linspace(0, 1, A))
        for i in range(A):
            plt.scatter(y_true[:, i], y_pred[:, i], color=colors[i], alpha=0.7)
        
    else:
        tickers_unique = list(set(tickers))
        colors = plt.cm.tab20(np.linspace(0, 1, len(tickers_unique)))
        color_dict = dict(zip(tickers_unique, colors))
        color_pt_list = [color_dict[t] for t in tickers]

        plt.scatter(y_true[:, 0], y_pred[:, 0], color=color_pt_list, alpha=0.7)

    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k--", lw=2)

    plt.xlabel("Realized Variance")
    plt.ylabel("Predicted Variance")
    plt.legend(ncol=2, fontsize=7)
    plt.title("Predicted vs Realized Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(f'{save_path}/pred_vs_real_scatter.png')
    plt.show()
    os.makedirs(f'{save_path}/var_paths', exist_ok=True)

    if data_type == 'multivariate':
        for i in range(A):
            plt.plot(dates, y_true[:, i], label=f'Realized Variance - {tickers[i]}', color='blue')
            plt.plot(dates, y_pred[:, i], label=f'Predicted Variance - {tickers[i]}', color='red')
            plt.legend(ncol=2, fontsize=7)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}/var_paths/{tickers[i]}.png')
            plt.cla()
        
    else:
        tickers_unique = list(set(tickers))
        for ticker in tickers_unique:
            idxs = [i for i, t in enumerate(tickers) if t == ticker]
            plt.plot(dates, y_true[idxs, 0], label=f'Realized Variance - {ticker}', color='blue')
            plt.plot(dates, y_pred[idxs, 0], label=f'Predicted Variance - {ticker}', color='red')
            plt.legend(ncol=2, fontsize=7)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}/var_paths/{ticker}.png')
            plt.cla()

    

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU device:", torch.cuda.get_device_name(0))
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    df = pd.read_pickle('full_data_1min_corrected.pkl')
    num_days = 10
    X_freq = 30
    y_freq = 5
    mode = "univariate"
    ticker_split = True
    batch_size = 512
    # val_start_date = '2020-10-01'
    # test_start_date = '2021-01-01'
    val_start_date = '2021-01-01'
    test_start_date = '2021-02-01'
    residual_channels=32
    skip_channels=64
    dilation_channels=128
    end_channels=32
    kernel_size=3
    num_blocks=2
    num_layers=6
    dropout=0.0
    clip_val=1.0
    lr= 1e-3
    patience=50
    max_epochs=1000
    run_name = f'split_FF_{X_freq}-{y_freq}_min_{num_days}day_{mode}'
    
    print(df.index)
    datamodule = StockVolDataModule(df=df, X_freq=X_freq, y_freq=y_freq, val_start_date=val_start_date, test_start_date=test_start_date,
                                 num_days=num_days, mode=mode, ticker_split=ticker_split, batch_size=batch_size)
    
    num_stocks = df.shape[1]
    model = DilatedCausalCNN(
    in_channels=(num_stocks if mode=="multivariate" else 1),   # features = number of stocks
    out_channels=(num_stocks if mode=="multivariate" else 1),
    residual_channels=residual_channels,
    dilation_channels=dilation_channels,
    skip_channels=skip_channels,
    end_channels=end_channels,
    kernel_size=kernel_size,
    num_blocks=num_blocks,
    num_layers=num_layers,
    dropout=dropout,
    lr=lr,)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")


    checkpoint_cb = ModelCheckpoint(
        dirpath= f"checkpoints/{run_name}/",
        filename="best_model",
        save_last=True,          # saves "last.ckpt"
        save_top_k=1,            # saves the best model based on monitored metric
        monitor="val_loss",
        mode="min",
    )

    history = LossHistory()
    torch.cuda.empty_cache()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop, history, checkpoint_cb],
        accelerator="gpu",
        gradient_clip_val=clip_val,
    )
    

    print(trainer.accelerator)

    trainer.fit(model, datamodule=datamodule)

    plt.figure(figsize=(8, 5))
    plt.plot(history.train_losses, label="Training Loss")
    plt.plot(history.val_losses[1:], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'checkpoints/{run_name}/loss_curve.png')
    plt.show()

    model = DilatedCausalCNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(model, datamodule=datamodule)
    y_pred = model.test_preds
    y_true = model.test_targets
    tickers = df.columns.tolist() if mode=="multivariate" else datamodule.test_tickers.tolist()
    dates = datamodule.test_dates
    scatter_pred_vs_real(y_true, y_pred, data_type=mode, dates=dates, tickers=tickers, 
                         save_path=f'checkpoints/{run_name}')
    






    

