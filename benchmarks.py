
"""
The code adopts volatility baselines presented in the respected paper. 
The baselines include:
    Martingale
    HAR model
    HEAVY model
    GARCH(1,1) 
    EGARCH(1,1) 
    TARCH(1,1) 
    APARCH(1,1) 
    LSTM
With the respected metrics, including MAE, RMSE, SMAPE, ME, MedAE, QLIKE
"""

#importing packages
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta

from arch import arch_model
from src.aqf.data_access import FirstRateDataClient

#For LSTM
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False



# Test tickers with the respected modifications 
TICKERS = [
    "AAL",
    "AAPL",
    "ADBE",
    "AMAT",
    "AMGN",
    "ASML",
    "BMRN",
    "CDNS",
    "CHKP",
    "CMCSA",
    "CSX",
    "EA",
    "EXPE",
    "FAST",
    "FI",      
    "HAS",
    "HSIC",
    "ILMN",
    "INTC",
    "JD",
    "KHC",
    "KLAC",
    "LULU",
    "MAR",
    "MCHP",
    "MDLZ",
    "META",    
    "MNST",
    "MU",
    "NTAP",
    "ORLY",
    "PCAR",
    "PYPL",
    "REGN",
    "SIRI",
    "SNPS",
    "ULTA",
    "VRSK",
    "VRSN",
    "WBA",
    "WDC",
    "WYNN",
    "XEL",
]

# Training tickers for LSTM
TRAIN_TICKERS = [
    "ADI", "ADP", "ADSK", "ALGN", "AMD", "AMZN", "AVGO", "BIDU", "BIIB", "BKNG",
    "CHTR", "COST", "CSCO", "CTAS", "CTSH", "DLTR", "EBAY", "EXC", "FOX", "FOXA",
    "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTU", "LRCX", "MELI", "MSFT", "NFLX",
    "NTES", "NVDA", "NXPI", "PAYX", "PEP", "QCOM", "ROST", "SBUX", "SWKS", "TMUS",
    "TSLA", "TTWO", "TXN", "UAL", "VRTX", "WDAY", "WTW"
]

# Overall time horizon
DATA_START = pd.Timestamp("2019-09-30")
DATA_END   = pd.Timestamp("2021-09-30")

# Train set
TRAIN_START = pd.Timestamp("2019-09-30")
TRAIN_END   = pd.Timestamp("2020-09-30")

# Validation set
VAL_START   = pd.Timestamp("2020-10-01")
VAL_END     = pd.Timestamp("2020-12-31")

# Test set
TEST_START  = pd.Timestamp("2021-01-01")
TEST_END    = pd.Timestamp("2021-09-30")

# Sampling for 5-min intervals
INTRADAY_FREQ_MINUTES = 5

# Scaling for basis points
SCALE_RV = 1e4

# As the data loading takes an increasing memory capacity, the solution to overcome the issue has been to only load 10 days intraday data at a time and to compute the respected realized variance of that, hence making sure that the computer memory doesn't overload.
CHUNK_DAYS = 10

# Safety measure for forecasting
MIN_HISTORY_DAYS = 60

# Epsilon error term
EPS = 1e-12

# LSTM hyperparameters 
LSTM_SEQ_LEN = 20
LSTM_HIDDEN = 16
LSTM_LAYERS = 1
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3


# Respected metrics
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred) + EPS
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def me_metric(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.max(np.abs(y_true - y_pred)))


def medae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.median(np.abs(y_true - y_pred)))


def qlike(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    f = np.clip(y_pred, EPS, None)
    y = np.clip(y_true, EPS, None)
    return float(np.mean(np.log(f) + y / f))


METRIC_FUNCS = {
    "MAE": mae,
    "RMSE": rmse,
    "SMAPE": smape,
    "ME": me_metric,
    "MedAE": medae,
    "QLIKE": qlike,
}



#For the respected 10 days of data loaded, the function computes daily realized variance using the sampled log-returns and daily close prices
def process_intraday_chunk(df_chunk: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_chunk is None or len(df_chunk) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = df_chunk[["timestamp", "ticker", "close"]].copy()
    df = df[df["ticker"].isin(TICKERS + TRAIN_TICKERS)]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # sampling for 5-min intervals
    df = df.set_index("timestamp")
    mask = (df.index.minute % INTRADAY_FREQ_MINUTES == 0)
    df_5 = df[mask].copy()
    if df_5.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Daily close from 5-min data
    df_5["date"] = df_5.index.normalize()
    daily_close_chunk = (
        df_5.groupby(["date", "ticker"])["close"]
            .last()
            .unstack("ticker")
            .sort_index()
    )

    # RV from squared 5-min log returns
    df_5["log_price"] = np.log(df_5["close"])
    df_5["log_ret"] = df_5.groupby("ticker")["log_price"].diff()
    df_5 = df_5.dropna(subset=["log_ret"])

    rv_chunk = (
        df_5.groupby(["date", "ticker"])["log_ret"]
            .apply(lambda x: SCALE_RV * np.sum(x.values ** 2))
            .unstack("ticker")
            .sort_index()
    )

    return daily_close_chunk, rv_chunk


# Processing and aggregating chunks of days for the whole time horizon, gettignthe relevant daily close returns and realized variances for the respected tickers
def build_daily_series_in_chunks(
    client: FirstRateDataClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    daily_close_chunks: list[pd.DataFrame] = []
    rv_chunks: list[pd.DataFrame] = []

    current_start = start

    while current_start <= end:
        current_end = min(current_start + timedelta(days=CHUNK_DAYS - 1), end)
    
        df_chunk = client.load_date_range(
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
        )

        if df_chunk is None or len(df_chunk) > 0:
            dc_chunk, rv_chunk = process_intraday_chunk(df_chunk)
            if not dc_chunk.empty:
                daily_close_chunks.append(dc_chunk)
            if not rv_chunk.empty:
                rv_chunks.append(rv_chunk)

        current_start = current_end + timedelta(days=1)

    if not daily_close_chunks or not rv_chunks:
        raise RuntimeError("No daily data constructed from intraday chunks.")

    # Concatenate over time
    daily_close = pd.concat(daily_close_chunks).sort_index()
    rv_daily = pd.concat(rv_chunks).sort_index()

    # Removing repeated dates, if the loaded chunks overlap
    daily_close = daily_close.groupby(daily_close.index).last()
    rv_daily = rv_daily.groupby(rv_daily.index).last()

    # Alignment to tickers
    all_tickers = sorted(set(TICKERS + TRAIN_TICKERS))
    common_cols = sorted(set(all_tickers) & set(daily_close.columns) & set(rv_daily.columns))
    daily_close = daily_close[common_cols]
    rv_daily = rv_daily[common_cols]

    # Daily closing log returns
    daily_ret = np.log(daily_close / daily_close.shift(1))

    # Dropping timezone
    for df in (daily_close, daily_ret, rv_daily):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    return daily_close, daily_ret, rv_daily



# Helper function for splitting the training, validation and testing periods
def make_train_val_test_masks(dates: pd.DatetimeIndex):
    dates = pd.to_datetime(dates)
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask   = (dates >= VAL_START)   & (dates <= VAL_END)
    test_mask  = (dates >= TEST_START)  & (dates <= TEST_END)
    return train_mask, val_mask, test_mask



# Martingale model, predicting tomorrow's value with today's one
def martingale_forecast(rv_series: pd.Series) -> pd.Series:
    rv = rv_series.sort_index()
    return rv.shift(1)


def har_rv_forecast(rv_series: pd.Series) -> pd.Series:

    # Sorting
    rv = rv_series.sort_index().dropna()
    if len(rv) < 30:
        raise RuntimeError("HAR-RV: not enough data in RV series.")

    dates_all = rv.index

    #  Safety
    log_rv = np.log(rv + EPS)

    # Daily, weekly, monthly components for the respected trading days
    log_rv_d = log_rv.shift(1)                    
    log_rv_w = log_rv.shift(1).rolling(5).mean()  
    log_rv_m = log_rv.shift(1).rolling(22).mean() 

    df = pd.DataFrame({
        "log_rv":   log_rv,
        "log_rv_d": log_rv_d,
        "log_rv_w": log_rv_w,
        "log_rv_m": log_rv_m,
    }).dropna()

    if df.empty:
        raise RuntimeError("HAR-RV: no rows after lag/rolling/dropna.")

    d_idx = df.index

    train_mask, val_mask, test_mask = make_train_val_test_masks(d_idx)
    est_mask = train_mask | val_mask

    if est_mask.sum() < 30:
        raise RuntimeError("HAR-RV: not enough data in train+val.")

    # Matrix for ordinary least squares
    X_est = df.loc[est_mask, ["log_rv_d", "log_rv_w", "log_rv_m"]].values
    y_est = df.loc[est_mask, "log_rv"].values

    
    X_design = np.column_stack([np.ones(len(X_est)), X_est])

    beta, *_ = np.linalg.lstsq(X_design, y_est, rcond=None)

    # Forecast 
    X_all = df[["log_rv_d", "log_rv_w", "log_rv_m"]].values
    X_all_design = np.column_stack([np.ones(len(X_all)), X_all])
    log_rv_hat_all = X_all_design @ beta

    # Scale
    rv_hat_full = pd.Series(np.exp(log_rv_hat_all), index=d_idx)

    # Restrict to test dates with enough history
    test_dates = d_idx[test_mask]
    test_dates = [d for d in test_dates if np.sum(d_idx < d) >= MIN_HISTORY_DAYS]

    if not test_dates:
        raise RuntimeError("HAR-RV: no valid test dates after MIN_HISTORY_DAYS check.")

    rv_hat_test = rv_hat_full.loc[test_dates]

    # Clipping to avoid outliers
    rv_q1, rv_q99 = np.quantile(rv.values, [0.01, 0.99])
    rv_hat_test = rv_hat_test.clip(lower=rv_q1, upper=rv_q99)

    return rv_hat_test


def heavy_rv_forecast(rv_series: pd.Series, ret_series: pd.Series) -> pd.Series:
 
    # Sort
    rv = rv_series.sort_index().dropna()
    ret = ret_series.sort_index().dropna()

    # Align dates
    common = rv.index.intersection(ret.index)
    if len(common) < MIN_HISTORY_DAYS + 10:
        raise RuntimeError("HEAVY: not enough overlapping days.")

    rv_c = rv.loc[common]
    r2_c = (ret.loc[common] ** 2)

    log_rv = np.log(rv_c + EPS)

    # Lagged features
    df = pd.DataFrame({
        "log_rv": log_rv,
        "log_rv_lag": log_rv.shift(1),
        "r2_lag": r2_c.shift(1),
    }).dropna()

    dates = df.index

    # Standardizing
    r2_mean = df["r2_lag"].mean()
    r2_std = df["r2_lag"].std()
    if r2_std < EPS:
        r2_std = 1.0
    df["r2_lag_std"] = (df["r2_lag"] - r2_mean) / r2_std

    # Training, validation and test helper
    train_mask, val_mask, test_mask = make_train_val_test_masks(dates)
    est_mask = train_mask | val_mask

    if est_mask.sum() < 30:
        raise RuntimeError("HEAVY: not enough data in train+val.")

    # Ordinary least squares
    X_est = df.loc[est_mask, ["log_rv_lag", "r2_lag_std"]].values
    y_est = df.loc[est_mask, "log_rv"].values

  
    X_design = np.column_stack([np.ones(len(X_est)), X_est])
    beta, *_ = np.linalg.lstsq(X_design, y_est, rcond=None)

    # Forecast 
    X_all = df[["log_rv_lag", "r2_lag_std"]].values
    X_all_design = np.column_stack([np.ones(len(X_all)), X_all])
    log_rv_hat_all = X_all_design @ beta

    # Scale
    rv_hat_full = pd.Series(np.exp(log_rv_hat_all), index=dates)

    # Restrict to test dates with enough history
    test_dates = dates[test_mask]
    test_dates = [d for d in test_dates if np.sum(dates < d) >= MIN_HISTORY_DAYS]

    if not test_dates:
        raise RuntimeError("HEAVY: no valid test dates after MIN_HISTORY_DAYS check.")

    # Clipping to avoid outliers
    rv_hat_test = rv_hat_full.loc[test_dates]
    rv_q1, rv_q99 = np.quantile(rv_c.values, [0.01, 0.99])
    rv_hat_test = rv_hat_test.clip(lower=rv_q1, upper=rv_q99)

    return rv_hat_test

# Important: as mentioned, we fix p=q=1
def forecast_arch_family_ret_to_rv(
    ret_series: pd.Series,
    rv_series: pd.Series,
    model_name: str,
    vol: str,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    dist: str = "normal",
    scale_ret: float = 100.0,   
    scale_rv: float = SCALE_RV,
) -> pd.Series:
  
    ret_s = ret_series.sort_index().dropna()
    rv_s  = rv_series.sort_index().dropna()

    common_dates = ret_s.index.intersection(rv_s.index)
    if len(common_dates) < MIN_HISTORY_DAYS + 10:
        raise RuntimeError(f"{model_name}: not enough overlapping days.")

    ret_c = ret_s.loc[common_dates]
    rv_c  = rv_s.loc[common_dates]

    dates = ret_c.index
    train_mask, val_mask, test_mask = make_train_val_test_masks(dates)
    if not test_mask.any():
        raise RuntimeError(f"{model_name}: no test period in overlap.")

    # Fit on  dates 
    r_fit = ret_c.copy()
    r_fit_scaled = r_fit * scale_ret

    am = arch_model(
        r_fit_scaled.values,
        mean="Zero",
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist=dist,
    )
    res = am.fit(disp="off")

    sigma_scaled = res.conditional_volatility
    var_scaled = sigma_scaled ** 2
    var_unscaled = var_scaled / (scale_ret ** 2)
    rv_hat_full = pd.Series(var_unscaled * scale_rv, index=dates)

    test_dates = dates[test_mask]
    test_dates = [d for d in test_dates if np.sum(dates < d) >= MIN_HISTORY_DAYS]

    if not test_dates:
        raise RuntimeError(f"{model_name}: no valid test dates after MIN_HISTORY_DAYS check.")

    return rv_hat_full.loc[test_dates]


#LSTM Baseline

class RVSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class RVLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


# Training LSTM
def train_global_lstm_on_rv(rv_daily: pd.DataFrame) -> pd.DataFrame | None:

    if not HAS_TORCH:
        print("PyTorch not available, skipping LSTM baseline.")
        return None


    rv_log = np.log(rv_daily + EPS)

    # Training on respected tickers
    primary_train_universe = [t for t in TRAIN_TICKERS if t in rv_log.columns]
    if len(primary_train_universe) > 0:
        train_universe = primary_train_universe
    else:
        train_universe = [t for t in TICKERS if t in rv_log.columns]
        
    # Building training sequences 
    seqs = []
    targets = []

    for ticker in train_universe:
        s = rv_log[ticker].dropna()
        if len(s) < LSTM_SEQ_LEN + 10:
            continue
        dates = s.index
        train_mask, val_mask, test_mask = make_train_val_test_masks(dates)
        usable_mask = train_mask | val_mask

        # sequences ending at day i of length L, i.e. rollingwindow of length L
        for i in range(LSTM_SEQ_LEN, len(dates)):
            if not usable_mask[i]:
                continue
            # preventing lookahead bias
            if not usable_mask[i-LSTM_SEQ_LEN:i].all():
                continue
            window_dates = dates[i-LSTM_SEQ_LEN:i]
            seq = s.loc[window_dates].values  
            target = s.iloc[i]
            seqs.append(seq)
            targets.append(target)

    if len(seqs) == 0:
        print("No LSTM training sequences constructed, skipping LSTM.")
        return None

    seqs = np.array(seqs, dtype=np.float32)  #  for reference, shape: (N, L)
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)  #for reference, shape: (N, 1)

    # Standardize 
    mu = seqs.mean()
    sigma = seqs.std() if seqs.std() > 0 else 1.0

    seqs_norm = (seqs - mu) / sigma
    targets_norm = (targets - mu) / sigma

    seqs_norm = seqs_norm[..., None]  #For reference, shape: (N, L, 1)

    dataset = RVSequenceDataset(seqs_norm, targets_norm)
    loader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RVLSTM(
        input_size=1,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(LSTM_EPOCHS):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(dataset)

    # Testing
    model.eval()
    preds = {}
    with torch.no_grad():
        for ticker in TICKERS:
            if ticker not in rv_log.columns:
                continue
            s = rv_log[ticker].dropna()
            if len(s) < LSTM_SEQ_LEN + 10:
                continue
            dates = s.index
            train_mask, val_mask, test_mask = make_train_val_test_masks(dates)

            # Use all days up to test for conditioning
            for i in range(LSTM_SEQ_LEN, len(dates)):
                if not test_mask[i]:
                    continue
                # Full window before the test date 
                window_dates = dates[i-LSTM_SEQ_LEN:i]
                seq = s.loc[window_dates].values.astype(np.float32)
                seq_norm = ((seq - mu) / sigma).reshape(1, LSTM_SEQ_LEN, 1)
                X = torch.from_numpy(seq_norm).to(device)
                y_hat_norm = model(X).cpu().numpy().ravel()[0]
                # add the mu and the sigma terms
                log_rv_hat = y_hat_norm * sigma + mu
                rv_hat = float(np.exp(log_rv_hat))
                date_i = dates[i]
                preds.setdefault(ticker, {})[date_i] = rv_hat

    # df of forecasts 
    rv_hat_lstm = pd.DataFrame(index=rv_daily.index, columns=rv_daily.columns, dtype=float)
    for ticker, dct in preds.items():
        for dt, val in dct.items():
            if dt in rv_hat_lstm.index:
                rv_hat_lstm.at[dt, ticker] = val

    return rv_hat_lstm



# evaluation of all baselines for every ticker
def evaluate_ticker(
    ticker: str,
    daily_rv: pd.DataFrame,
    daily_ret: pd.DataFrame,
    lstm_pred: pd.DataFrame | None = None,
) -> pd.DataFrame:
   
    if ticker not in daily_rv.columns or ticker not in daily_ret.columns:
        print(f"  {ticker}: not in daily_rv or daily_ret, skipping.")
        return pd.DataFrame()

    rv_s = daily_rv[ticker].dropna()
    ret_s = daily_ret[ticker].dropna()

    # Aligning dates
    common_dates = rv_s.index.intersection(ret_s.index)
    if len(common_dates) < MIN_HISTORY_DAYS + 10:
        print(f"  {ticker}: not enough overlapping days ({len(common_dates)}), skipping.")
        return pd.DataFrame()

    rv_s = rv_s.loc[common_dates]
    ret_s = ret_s.loc[common_dates]

    dates = rv_s.index
    train_mask, val_mask, test_mask = make_train_val_test_masks(dates)
    if not test_mask.any():
        print(f"  {ticker}: no test period, skipping.")
        return pd.DataFrame()

    y_test = rv_s[test_mask].copy()
    results = []

    # Martingale baseline
    try:
        f_mart = martingale_forecast(rv_s)
        f_test_m = f_mart[test_mask].dropna()
        y_test_m = y_test.loc[f_test_m.index]
        if len(f_test_m) > 0:
            metrics = {name: func(y_test_m.values, f_test_m.values)
                       for name, func in METRIC_FUNCS.items()}
            metrics["Model"] = "Martingale"
            metrics["Ticker"] = ticker
            results.append(metrics)
    except Exception as e:
        print(f"  Martingale failed for {ticker}: {e}")

    # HAR baseline
    try:
        rv_hat_har = har_rv_forecast(rv_s)
        idx_har = rv_hat_har.index.intersection(y_test.index)
        if len(idx_har) > 0:
            y_test_h = y_test.loc[idx_har]
            f_test_h = rv_hat_har.loc[idx_har]
            metrics = {name: func(y_test_h.values, f_test_h.values)
                       for name, func in METRIC_FUNCS.items()}
            metrics["Model"] = "HAR-RV"
            metrics["Ticker"] = ticker
            results.append(metrics)
    except Exception as e:
        print(f"  HAR failed for {ticker}: {e}")

    # HEAVY baseline
    try:
        rv_hat_heavy = heavy_rv_forecast(rv_s, ret_s)
        idx_h = rv_hat_heavy.index.intersection(y_test.index)
        if len(idx_h) > 0:
            y_test_he = y_test.loc[idx_h]
            f_test_he = rv_hat_heavy.loc[idx_h]
            metrics = {name: func(y_test_he.values, f_test_he.values)
                       for name, func in METRIC_FUNCS.items()}
            metrics["Model"] = "HEAVY(1,1)"
            metrics["Ticker"] = ticker
            results.append(metrics)
    except Exception as e:
        print(f"  HEAVY(1,1) failed for {ticker}: {e}")

    # GARCH-family baseline, with p=q=1
    garch_models = [
        ("GARCH(1,1)", "GARCH",   1, 1, 0),
        ("EGARCH(1,1)", "EGARCH", 1, 1, 0),
        ("GJR-GARCH(1,1)", "GARCH", 1, 1, 1),
        ("APARCH(1,1)", "APARCH", 1, 1, 0),
    ]

    for name, vol, p, q, o in garch_models:
        try:
            rv_hat = forecast_arch_family_ret_to_rv(
                ret_series=ret_s,
                rv_series=rv_s,
                model_name=name,
                vol=vol,
                p=p,
                q=q,
                o=o,
            )
            idx_g = rv_hat.index.intersection(y_test.index)
            if len(idx_g) == 0:
                continue
            y_test_g = y_test.loc[idx_g]
            f_test_g = rv_hat.loc[idx_g]
            metrics = {mname: func(y_test_g.values, f_test_g.values)
                       for mname, func in METRIC_FUNCS.items()}
            metrics["Model"] = name
            metrics["Ticker"] = ticker
            results.append(metrics)
        except Exception as e:
            print(f"  {name} failed for {ticker}: {e}")

    # LSTM baseline 
    if lstm_pred is not None and ticker in lstm_pred.columns:
        try:
            rv_hat_lstm = lstm_pred[ticker].dropna()
            idx_l = rv_hat_lstm.index.intersection(y_test.index)
            if len(idx_l) > 0:
                y_test_l = y_test.loc[idx_l]
                f_test_l = rv_hat_lstm.loc[idx_l]
                metrics = {mname: func(y_test_l.values, f_test_l.values)
                           for mname, func in METRIC_FUNCS.items()}
                metrics["Model"] = "LSTM-RV"
                metrics["Ticker"] = ticker
                results.append(metrics)
        except Exception as e:
            print(f"  LSTM-RV failed for {ticker}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)



def main():
    client = FirstRateDataClient()

    # Build daily_close, daily_ret, rv_daily 
    daily_close, daily_ret, rv_daily = build_daily_series_in_chunks(
        client,
        DATA_START,
        DATA_END,
    )

    #  LSTM baseline 
    if HAS_TORCH:
        lstm_pred = train_global_lstm_on_rv(rv_daily)
    else:
        lstm_pred = None

    # Evaluate baselines for each ticker ticker
    all_results = []

    for ticker in sorted(TICKERS):
        res_t = evaluate_ticker(ticker, rv_daily, daily_ret, lstm_pred)
        if not res_t.empty:
            all_results.append(res_t)

    if not all_results:
        print("No results produced.")
        return

    results_df = pd.concat(all_results, ignore_index=True)

    # Aggregate across tickers
    metric_cols = list(METRIC_FUNCS.keys())
    agg = (
        results_df
        .groupby("Model", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("QLIKE")
    )

    print("Aggregate across tickers (mean metrics, test period):")
    print("====================")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
