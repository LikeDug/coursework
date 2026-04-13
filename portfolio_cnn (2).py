"""
=====================================================================
  Portfolio Allocation in Stock Markets using CNN (PyTorch)
=====================================================================
  Author  : Senior ML Engineer
  Data    : Real market data via Yahoo Finance (yfinance)
  Model   : 2-D Convolutional Neural Network (CNN)
  Assets  : AAPL, MSFT, GOOGL, AMZN  (configurable)
=====================================================================

WHY CNN FOR PORTFOLIO DATA?
-----------------------------
Each daily market snapshot is structured as a (4 features x 4 assets)
matrix — identical in shape to a tiny grayscale image.

  Rows    -> [Price, Volatility, Return, Risk]   (financial features)
  Columns -> [AAPL, MSFT, GOOGL, AMZN]           (assets)

Conv kernels slide across this "image" learning cross-feature patterns:
  e.g.  kernel detects "high return + low risk"       -> up-weight asset
  e.g.  kernel detects "high volatility + falling price" -> down-weight

AdaptiveAvgPool makes the head size-independent, so the same
architecture works regardless of how many trading days are loaded.
=====================================================================
"""

# ── Standard library ──────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

# ── Third-party: data ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import yfinance as yf                          # pip install yfinance

# ── Third-party: deep learning ────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── Third-party: visualisation ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                          # headless / file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Third-party: metrics ──────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Global seeds for reproducibility ─────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════
#  SECTION 1 — REAL DATA INGESTION  (Yahoo Finance)
# ════════════════════════════════════════════════════════════════════

def generate_portfolio_data_real(
    tickers: list   = None,
    start_date: str = "2020-01-01",
    end_date: str   = None,
    period: str     = "1d",
) -> tuple:
    """
    Download OHLCV data from Yahoo Finance and engineer the four
    financial features used as CNN input channels.

    Feature engineering
    -------------------
    For every trading day t (after a 20-day warm-up window):

      Price      — raw Adj Close, then min-max scaled across all
                   dates and assets together  -> [0, 1]
      Volatility — 20-day rolling annualised std-dev of log-returns
                   = rolling_std(log_ret, 20) x sqrt(252)
      Return     — 20-day rolling annualised mean of log-returns
                   = rolling_mean(log_ret, 20) x 252
      Risk       — copy of Volatility (Sharpe denominator proxy)

    All four features are normalised jointly (min-max over the full
    time series x asset space) so the CNN receives values in [0, 1].

    Target allocation
    -----------------
    Each day's label = softmax( Return / (Risk + eps) ) over assets.
    This is the "Sharpe-score allocation" — a differentiable proxy
    for a classical mean-variance optimiser.  Small Gaussian noise
    (sigma = 0.005) prevents the CNN from memorising exact scores.

    Parameters
    ----------
    tickers    : list of Yahoo Finance ticker symbols
    start_date : history start  (YYYY-MM-DD)
    end_date   : history end    (YYYY-MM-DD); defaults to today
    period     : bar interval   ("1d", "1wk", "1mo")

    Returns
    -------
    X            : np.ndarray, float32, shape (N, 4, n_assets)
    y            : np.ndarray, float32, shape (N, n_assets)
    feature_names: list[str]   ["Price", "Volatility", "Return", "Risk"]
    asset_names  : list[str]   same order as tickers
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"\n  Downloading data for {tickers}  "
          f"({start_date} -> {end_date}, interval={period}) ...")

    # ── 1. Download adjusted close prices ────────────────────────
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=period,
        progress=False,
        auto_adjust=True,          # 'Close' already adjusted
    )

    # yfinance >= 0.2 returns MultiIndex columns; normalise to Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if len(tickers) == 1 else raw

    # Keep only the requested tickers (safety guard)
    prices = prices[tickers].dropna(how="all")

    # Forward-fill tiny gaps (bank holidays etc.)
    prices = prices.ffill().dropna()

    print(f"  Raw price rows after cleaning: {len(prices)}")

    # ── 2. Log-returns (more stationary than pct_change) ─────────
    log_ret = np.log(prices / prices.shift(1)).dropna()

    # ── 3. Rolling 20-day features (annualised) ───────────────────
    WINDOW    = 20
    roll_std  = log_ret.rolling(WINDOW).std()  * np.sqrt(252)   # volatility
    roll_mean = log_ret.rolling(WINDOW).mean() * 252             # expected return

    # Align all series on same valid index (NaN in first WINDOW-1 rows)
    features_df = pd.concat(
        [
            prices.add_suffix("_Price"),
            roll_std.add_suffix("_Volatility"),
            roll_mean.add_suffix("_Return"),
            roll_std.add_suffix("_Risk"),          # Risk == Volatility here
        ],
        axis=1,
    ).dropna()

    n_samples = len(features_df)
    n_assets  = len(tickers)

    print(f"  Usable samples after rolling window: {n_samples}")

    # ── 4. Build 3-D array  (N, 4_features, n_assets) ────────────
    X = np.zeros((n_samples, 4, n_assets), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        X[:, 0, i] = features_df[f"{ticker}_Price"].values
        X[:, 1, i] = features_df[f"{ticker}_Volatility"].values
        X[:, 2, i] = features_df[f"{ticker}_Return"].values
        X[:, 3, i] = features_df[f"{ticker}_Risk"].values

    # ── 5. Min-max normalisation per feature (jointly over assets) ─
    # Normalise each feature slice jointly so relative asset ordering
    # is preserved — a crucial property for the CNN.
    for f in range(4):
        f_min = X[:, f, :].min()
        f_max = X[:, f, :].max()
        if f_max > f_min:
            X[:, f, :] = (X[:, f, :] - f_min) / (f_max - f_min)

    # ── 6. Build target allocations  (Sharpe-score softmax) ───────
    returns_norm = X[:, 2, :]          # shape (N, n_assets) in [0, 1]
    risk_norm    = X[:, 3, :]

    # Sharpe-like score: higher return / lower risk -> more weight
    scores     = returns_norm / (risk_norm + 1e-6)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    y          = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Tiny Gaussian noise breaks exact label-feature correspondence,
    # forcing the CNN to generalise rather than memorise.
    noise = np.random.normal(0, 0.005, y.shape).astype(np.float32)
    y     = np.clip(y + noise, 0, 1)
    y    /= y.sum(axis=1, keepdims=True)          # re-normalise to simplex

    feature_names = ["Price", "Volatility", "Return", "Risk"]
    asset_names   = tickers

    print(f"  Dataset ready  --  X: {X.shape}   y: {y.shape}\n")

    return X, y.astype(np.float32), feature_names, asset_names


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 — CNN MODEL
# ════════════════════════════════════════════════════════════════════

class PortfolioCNN(nn.Module):
    """
    2-D CNN for portfolio weight prediction.

    Input  : (B, 1, n_features=4, n_assets)   <- single-channel "image"
    Output : (B, n_assets)                     <- allocation simplex

    Architecture
    ────────────
    Conv Block 1 : Conv2d(1  -> 32, 2x2) -> GELU -> BatchNorm2d
    Conv Block 2 : Conv2d(32 -> 64, 2x2) -> GELU -> BatchNorm2d
    AdaptiveAvgPool2d(1, 1)              -> flatten to 64-d vector
    FC  : Linear(64 -> 128) -> GELU -> Dropout(0.4)
          Linear(128 -> n_assets) -> Softmax

    Design notes
    ────────────
    * GELU (vs ReLU) : smoother gradient flow, empirically better on
      financial regression tasks with near-zero targets.
    * BatchNorm after conv : critical with real data whose distributions
      shift across market regimes; stabilises training.
    * AdaptiveAvgPool : the head is input-size-agnostic.
    * Softmax output : hard constraint — allocations are always a valid
      probability distribution (sum = 1, each >= 0).
    """

    def __init__(self, n_assets: int = 4):
        super().__init__()

        # (B, 1, 4, n_assets) -> (B, 32, 3, n_assets-1)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(32),
        )

        # (B, 32, 3, n_assets-1) -> (B, 64, 2, n_assets-2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.GELU(),
            nn.BatchNorm2d(64),
        )

        # (B, 64, H, W) -> (B, 64, 1, 1) — global spatial average
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, n_assets),
        )

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for faster, stable convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_pool(x)           # (B, 64, 1, 1)
        x = x.view(x.size(0), -1)         # (B, 64)
        x = self.fc(x)                    # (B, n_assets)
        return self.softmax(x)


# ════════════════════════════════════════════════════════════════════
#  SECTION 3 — TRAINER
# ════════════════════════════════════════════════════════════════════

class PortfolioTrainer:
    """
    Encapsulates the training loop, validation, and inference.

    Loss   : MSELoss  (targets are continuous probabilities in [0,1])
    Optim  : AdamW    (Adam + decoupled weight decay for regularisation)
    Sched  : OneCycleLR  (warm-up then cosine anneal over all epochs)
             Particularly effective on financial time-series where early
             gradient directions can be noisy.
    """

    def __init__(self, model: nn.Module,
                 lr: float   = 3e-4,
                 device: str = "cpu"):
        self.model     = model.to(device)
        self.device    = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(),
                                     lr=lr, weight_decay=1e-3)
        self.history   = {"train_loss": [], "val_loss": []}
        self.scheduler = None    # configured in fit() once steps are known

    # ── fit ───────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader,
            val_loader:   DataLoader,
            epochs:       int = 200,
            verbose_every: int = 20) -> None:
        """
        Full training loop with per-epoch validation.

        Best model weights (by val loss) are saved and restored
        automatically at the end of training.

        Parameters
        ----------
        train_loader  : DataLoader — chronologically first 80%
        val_loader    : DataLoader — chronologically last 20%
        epochs        : total training epochs
        verbose_every : print a log line every N epochs
        """
        total_steps = epochs * len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=3e-3,
            total_steps=total_steps,
            pct_start=0.15,          # 15% warm-up, then cosine decay
            anneal_strategy="cos",
        )

        print("─" * 65)
        print(f"{'Epoch':>8}  {'Train MSE':>12}  {'Val MSE':>10}  {'LR':>12}")
        print("─" * 65)

        best_val  = float("inf")
        best_ckpt = None

        for epoch in range(1, epochs + 1):

            # ── Training phase ────────────────────────────────────
            self.model.train()
            train_loss = 0.0

            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device)
                y_b = y_b.to(self.device)

                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_b), y_b)
                loss.backward()

                # Gradient clipping: prevents instability on volatile
                # market periods where gradients can spike sharply.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)

            # ── Validation phase ──────────────────────────────────
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(self.device)
                    y_b = y_b.to(self.device)
                    val_loss += self.criterion(self.model(X_b), y_b).item()
            avg_val = val_loss / len(val_loader)

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)

            # ── Checkpoint best model ─────────────────────────────
            if avg_val < best_val:
                best_val  = avg_val
                best_ckpt = {k: v.clone()
                             for k, v in self.model.state_dict().items()}

            if epoch % verbose_every == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"{epoch:>8}  {avg_train:>12.6f}  "
                      f"{avg_val:>10.6f}  {lr:>12.2e}")

        # Restore weights that gave the best validation loss
        if best_ckpt is not None:
            self.model.load_state_dict(best_ckpt)

        print("─" * 65)
        print(f"  Best val loss : {best_val:.6f}  (weights restored)\n")

    # ── predict ───────────────────────────────────────────────────
    def predict(self, X_tensor: torch.Tensor) -> np.ndarray:
        """
        Batch inference — no gradient computation.

        Parameters
        ----------
        X_tensor : (N, 1, 4, n_assets)

        Returns
        -------
        np.ndarray, shape (N, n_assets)
        """
        self.model.eval()
        preds = []
        CHUNK = 256          # avoids OOM on large datasets
        with torch.no_grad():
            for s in range(0, len(X_tensor), CHUNK):
                batch = X_tensor[s:s + CHUNK].to(self.device)
                preds.append(self.model(batch).cpu().numpy())
        return np.concatenate(preds, axis=0)


# ════════════════════════════════════════════════════════════════════
#  SECTION 4 — EVALUATION
# ════════════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Multi-output regression metrics on allocation weights.

    MSE   — primary training loss
    RMSE  — same unit as allocation weight; easier to interpret
    MAE   — robust to outlier days (e.g. crash / gap days)
    R2    — fraction of variance explained across all assets & days
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("=" * 45)
    print("  EVALUATION METRICS  (validation set)")
    print("=" * 45)
    print(f"  MSE   : {mse:.6f}")
    print(f"  RMSE  : {rmse:.6f}")
    print(f"  MAE   : {mae:.6f}")
    print(f"  R2    : {r2:.4f}  {'(good)' if r2 > 0.80 else '(needs tuning)'}")
    print("=" * 45 + "\n")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ════════════════════════════════════════════════════════════════════
#  SECTION 5 — VISUALISATION
# ════════════════════════════════════════════════════════════════════

def visualize(
    history:     dict,
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    asset_names: list,
    metrics:     dict,
    save_path:   str = "portfolio_cnn_results.png",
) -> str:
    """
    5-panel results dashboard for real-data scale.

    Layout  (3-row GridSpec)
    ────────────────────────
    Row 0 (full width) : Train vs Val loss curves
    Row 1 left         : Predicted vs Actual scatter  (val set)
    Row 1 right        : Per-asset MAE bar chart
    Row 2 left         : Last-N-day actual allocations  (stacked bar)
    Row 2 right        : Last-N-day predicted allocations (stacked bar)

    NOTE: The stacked-bar approach from the original synthetic version
    would be unreadable with hundreds of real samples. The time-series
    stacked bars in Row 2 show a N-day window instead.
    """
    N_SHOW  = len(y_true)          # показываем весь валидационный сет (~5 лет)
    palette = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373",
               "#CE93D8", "#80DEEA", "#FFCC80", "#EF9A9A"]
    bg, fg, gc = "#0D1117", "#E6EDF3", "#21262D"

    fig = plt.figure(figsize=(17, 13), facecolor=bg)
    fig.suptitle(
        f"Portfolio CNN  |  Real Market Data  ({', '.join(asset_names)})",
        fontsize=15, fontweight="bold", color=fg, y=0.99,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.52, wspace=0.30,
    )

    def _style(ax, title):
        ax.set_facecolor(bg)
        ax.set_title(title, color=fg, fontsize=11, pad=10, fontweight="bold")
        ax.tick_params(colors=fg, labelsize=8)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        for sp in ax.spines.values():
            sp.set_edgecolor(gc)
        ax.grid(color=gc, linestyle="--", linewidth=0.6, alpha=0.8)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Row 0 : Loss curves (full width) ─────────────────────────
    ax_loss = fig.add_subplot(gs[0, :])

    ax_loss.plot(epochs, history["train_loss"],
                 color="#7C83FD", lw=2, label="Train MSE")
    ax_loss.plot(epochs, history["val_loss"],
                 color="#F48FB1", lw=2, ls="--", label="Val MSE")
    ax_loss.fill_between(epochs,
                         history["train_loss"], history["val_loss"],
                         alpha=0.08, color="#7C83FD", label="Train-Val gap")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")

    ft = history["train_loss"][-1]
    fv = history["val_loss"][-1]
    ax_loss.annotate(f"Train: {ft:.5f}", xy=(len(epochs), ft),
                     xytext=(-90, 14), textcoords="offset points",
                     color="#7C83FD", fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="#7C83FD"))
    ax_loss.annotate(f"Val: {fv:.5f}", xy=(len(epochs), fv),
                     xytext=(-90, -18), textcoords="offset points",
                     color="#F48FB1", fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="#F48FB1"))
    _style(ax_loss, "Training vs Validation Loss")
    ax_loss.legend(facecolor=gc, labelcolor=fg, fontsize=9)

    n_assets = len(asset_names)

    # ── Row 1 left : Scatter predicted vs actual ─────────────────
    ax_sc = fig.add_subplot(gs[1, 0])
    for i in range(n_assets):
        ax_sc.scatter(
            y_true[:, i], y_pred[:, i],
            color=palette[i % len(palette)],
            s=6, alpha=0.45, label=asset_names[i],
        )
    lim = [min(y_true.min(), y_pred.min()) - 0.01,
           max(y_true.max(), y_pred.max()) + 0.01]
    ax_sc.plot(lim, lim, color="white", lw=1.2, ls="--",
               alpha=0.5, label="Perfect fit")
    ax_sc.set_xlabel("Actual weight")
    ax_sc.set_ylabel("Predicted weight")
    ax_sc.text(0.04, 0.93, f"R2 = {metrics['r2']:.4f}",
               transform=ax_sc.transAxes, color="#A5D6A7",
               fontsize=10, fontweight="bold")
    _style(ax_sc, "Predicted vs Actual  (val set)")
    ax_sc.legend(facecolor=gc, labelcolor=fg, fontsize=7,
                 markerscale=2.5, ncol=2)

    # ── Row 1 right : Per-asset MAE ───────────────────────────────
    ax_mae = fig.add_subplot(gs[1, 1])
    per_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    bars = ax_mae.bar(
        asset_names, per_mae,
        color=palette[:n_assets], width=0.5,
        edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, per_mae):
        ax_mae.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + per_mae.max() * 0.02,
            f"{val:.5f}", ha="center", va="bottom", color=fg, fontsize=8,
        )
    ax_mae.set_ylabel("Mean Absolute Error")
    ax_mae.set_ylim(0, per_mae.max() * 1.35)
    _style(ax_mae, "Per-Asset MAE  (val set)")

    # Legend shared by both stacked-bar panels
    leg_els = [Patch(facecolor=palette[i], label=asset_names[i])
               for i in range(n_assets)]

    # ── Row 2 left : Actual allocations — last N_SHOW days ────────
    ax_act = fig.add_subplot(gs[2, 0])
    y_show = y_true[-N_SHOW:]
    x_idx  = np.arange(N_SHOW)
    bottom = np.zeros(N_SHOW)
    for i in range(n_assets):
        ax_act.bar(x_idx, y_show[:, i], width=1.0,
                   bottom=bottom, color=palette[i % len(palette)],
                   alpha=0.85, label=asset_names[i])
        bottom += y_show[:, i]
    ax_act.set_xlabel(f"Last {N_SHOW} val samples  (day index)")
    ax_act.set_ylabel("Allocation weight")
    ax_act.set_ylim(0, 1.12)
    ax_act.set_xlim(-0.5, N_SHOW - 0.5)
    ax_act.legend(handles=leg_els, facecolor=gc, labelcolor=fg,
                  fontsize=7, ncol=2)
    _style(ax_act, f"Actual Allocations  (last {N_SHOW} days)")

    # ── Row 2 right : Predicted allocations — last N_SHOW days ───
    ax_pred = fig.add_subplot(gs[2, 1])
    p_show  = y_pred[-N_SHOW:]
    bottom  = np.zeros(N_SHOW)
    for i in range(n_assets):
        ax_pred.bar(x_idx, p_show[:, i], width=1.0,
                    bottom=bottom, color=palette[i % len(palette)],
                    alpha=0.85, label=asset_names[i])
        bottom += p_show[:, i]
    ax_pred.set_xlabel(f"Last {N_SHOW} val samples  (day index)")
    ax_pred.set_ylabel("Allocation weight")
    ax_pred.set_ylim(0, 1.12)
    ax_pred.set_xlim(-0.5, N_SHOW - 0.5)
    ax_pred.legend(handles=leg_els, facecolor=gc, labelcolor=fg,
                   fontsize=7, ncol=2)
    _style(ax_pred, f"Predicted Allocations  (last {N_SHOW} days)")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"  Visualization saved -> {save_path}\n")
    return save_path


# ════════════════════════════════════════════════════════════════════
#  SECTION 6 — MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  Portfolio Allocation CNN  |  PyTorch  |  Yahoo Finance")
    print("=" * 60)

    # ── Device selection ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    # ═══════════════════════════════════════════════════════════════
    #  [1/6]  Download & engineer real features from Yahoo Finance
    # ═══════════════════════════════════════════════════════════════
    print("\n[ 1/6 ]  Loading real market data ...")

    TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    START_DATE = "2019-01-01"          # ~6 лет истории
    # end_date defaults to today inside the function

    X, y, feature_names, asset_names = generate_portfolio_data_real(
        tickers    = TICKERS,
        start_date = START_DATE,
    )

    n_samples, n_features, n_assets = X.shape

    print(f"  X : {X.shape}  ->  (samples, features, assets)")
    print(f"  y : {y.shape}  ->  (samples, allocation weights)")
    print(f"\n  Feature normalisation check  (should be in [0, 1]):")
    for fi, fname in enumerate(feature_names):
        print(f"    {fname:12s}  min={X[:, fi, :].min():.4f}  "
              f"max={X[:, fi, :].max():.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  [2/6]  Build tensors + chronological train/val split
    # ═══════════════════════════════════════════════════════════════
    print("\n[ 2/6 ]  Building DataLoaders ...")

    # Add channel dimension: (N, 4, n_assets) -> (N, 1, 4, n_assets)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    print(f"  Input tensor : {X_tensor.shape}  <- (B, C=1, H=4, W={n_assets})")

    # Chronological split — avoids future-leakage into past training data
    # Фиксированный val: 5 лет (252 торг. дня × 5), минимум 1 год на train
    n_val   = min(252 * 5, n_samples - 252)
    n_train = n_samples - n_val

    train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_ds   = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])

    # Batch size: aim for ~32 samples/batch, min 8
    batch_size = min(32, max(8, n_train // 20))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    print(f"  Train : {n_train} samples   "
          f"Val : {n_val} samples   "
          f"Batch : {batch_size}")

    # ═══════════════════════════════════════════════════════════════
    #  [3/6]  Instantiate model
    # ═══════════════════════════════════════════════════════════════
    print("\n[ 3/6 ]  Instantiating PortfolioCNN ...\n")

    model    = PortfolioCNN(n_assets=n_assets)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\n  Trainable parameters : {n_params:,}\n")

    # ═══════════════════════════════════════════════════════════════
    #  [4/6]  Train
    # ═══════════════════════════════════════════════════════════════
    print("[ 4/6 ]  Training ...\n")

    trainer = PortfolioTrainer(model, lr=3e-4, device=device)
    trainer.fit(
        train_loader,
        val_loader,
        epochs        = 200,
        verbose_every = 20,
    )

    # ═══════════════════════════════════════════════════════════════
    #  [5/6]  Evaluate on held-out validation set
    # ═══════════════════════════════════════════════════════════════
    print("[ 5/6 ]  Evaluating on validation set ...\n")

    X_val  = X_tensor[n_train:]
    y_val  = y[n_train:]
    y_pred = trainer.predict(X_val)
    metrics = evaluate(y_val, y_pred)

    # Sample comparison table
    N_PRINT = min(5, n_val)
    print("  Sample predictions (first 5 validation days):")
    for i in range(N_PRINT):
        act  = "    Actual     " + "  ".join(f"{v:.4f}" for v in y_val[i])
        pred = "    Predicted  " + "  ".join(f"{v:.4f}" for v in y_pred[i])
        delta= "    |Delta|    " + "  ".join(f"{abs(a-p):.4f}"
               for a, p in zip(y_val[i], y_pred[i]))
        label = "               " + "  ".join(f"{a:>6}" for a in asset_names)
        print(f"\n  -- Day val-{i} --")
        print(label)
        print(act)
        print(pred)
        print(delta)
    print()

    # ═══════════════════════════════════════════════════════════════
    #  [6/6]  Visualise
    # ═══════════════════════════════════════════════════════════════
    print("[ 6/6 ]  Generating visualisation ...")

    visualize(
        history     = trainer.history,
        y_true      = y_val,
        y_pred      = y_pred,
        asset_names = asset_names,
        metrics     = metrics,
        save_path   = "portfolio_cnn_results.png",
    )

    print("All done!  ✓")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
