import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================== 配置 ==================
DATA_FILES = ["AM_L_all.csv", "EP_L_all.csv", "FAA_L_all.csv", "OE_L_all.csv"]
DATA_DIR = "."                

INTERVAL = 3.0
DQ_THRESH = 0.01
SEED = 42

# 每个阶段内的划分比例（20/30/20/30）
R_STAGE_RIDGE = 0.2
R_STAGE_LSTM_TRAIN = 0.3
R_STAGE_LSTM_VAL   = 0.2
R_STAGE_LSTM_TEST  = 0.3

RIDGE_ALPHA = 1.0
RIDGE_FIT_INTERCEPT = False

SEQ_LEN = 5
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 3


GYRO_COLS = [0, 1, 2, 3]
Q_COLS    = [4, 5, 6, 7]   # 星敏A


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_csv_auto(path: str) -> pd.DataFrame:
    """自动判断CSV是否有表头"""
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline().strip().split(",")
    has_header = False
    for tok in first:
        try:
            float(tok)
        except:
            has_header = True
            break
    return pd.read_csv(path, header=0 if has_header else None, encoding="utf-8-sig")

def normalize_quat(Q):
    n = np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-12
    return Q / n

def build_monomial_16(w4, q4):
    """16维单项式特征：omega_i * q_j"""
    return np.outer(w4, q4).reshape(-1)

def stage_split_indices(M, seed, r_ridge=0.2, r_tr=0.3, r_va=0.2, r_te=0.3):
    """
    在单个阶段内部划分：20% ridge, 30% train, 20% val, 30% test
    """
    if M <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    idx = np.arange(M)
    rng.shuffle(idx)

    n_ridge = int(round(M * r_ridge))
    n_tr    = int(round(M * r_tr))
    n_va    = int(round(M * r_va))

    n_used = n_ridge + n_tr + n_va
    if n_used > M:
        # 极端情况下修正
        overflow = n_used - M
        n_va = max(0, n_va - overflow)
        n_used = n_ridge + n_tr + n_va

    ridge_idx = idx[:n_ridge]
    tr_idx    = idx[n_ridge:n_ridge + n_tr]
    va_idx    = idx[n_ridge + n_tr:n_ridge + n_tr + n_va]
    te_idx    = idx[n_used:]  # 剩余全给test

    return ridge_idx, tr_idx, va_idx, te_idx

def build_sequence_samples(W_full, Q_full, interval, dq_thresh, seq_len):
    """
    在单个文件/阶段内部构造样本，避免跨断点：
      输入序列：{Q(t-seq_len+1..t), W(t-seq_len+1..t)}
      标签：dQ(t) = (Q(t+1)-Q(t))/interval
    dq超阈值的step不构造 => 等价于该 t+1 不预测
    """
    dQ_true = (Q_full[1:] - Q_full[:-1]) / interval  # (N-1,4)
    W_t = W_full[:-1]
    Q_t = Q_full[:-1]

    mask_step_ok = np.all((dQ_true <= dq_thresh) & (dQ_true >= -dq_thresh), axis=1)

    K = SEQ_LEN 
    if mask_step_ok.shape[0] > 0:
        mask_step_ok[:min(K, mask_step_ok.shape[0])] = False

    idx_valid = np.where(mask_step_ok)[0]
    bad_time_idx = np.where(~mask_step_ok)[0] + 1

    if len(idx_valid) == 0:
        return (np.zeros((0, seq_len, 8), np.float32),
                np.zeros((0, 16), np.float32),
                np.zeros((0, 4), np.float32),
                np.zeros((0, 4), np.float32),
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.int64),
                bad_time_idx.astype(np.int64))


    splits = np.where(np.diff(idx_valid) != 1)[0] + 1
    segments = np.split(idx_valid, splits)

    X_seq_list, X_ridge_list = [], []
    y_list, Qc_list, Qn_list, t_list = [], [], [], []

    for seg in segments:
        if len(seg) < seq_len:
            continue
        for k in range(seq_len - 1, len(seg)):
            t = seg[k]
            hist = seg[k - seq_len + 1:k + 1]

            Q_hist = Q_t[hist]                       # (seq,4)
            W_hist = W_t[hist]                       # (seq,4)
            X_seq = np.hstack([Q_hist, W_hist])      # (seq,8)

            X_ridge = build_monomial_16(W_t[t], Q_t[t])  # (16,)
            y = dQ_true[t]
            Qc = Q_t[t]
            Qn = Q_full[t + 1]

            X_seq_list.append(X_seq)
            X_ridge_list.append(X_ridge)
            y_list.append(y)
            Qc_list.append(Qc)
            Qn_list.append(Qn)
            t_list.append(t)

    X_seq  = np.asarray(X_seq_list, dtype=np.float32)
    X_ridge = np.asarray(X_ridge_list, dtype=np.float32)
    dQ = np.asarray(y_list, dtype=np.float32)
    Q_curr = np.asarray(Qc_list, dtype=np.float32)
    Q_next = np.asarray(Qn_list, dtype=np.float32)
    t_index = np.asarray(t_list, dtype=np.int64)

    return X_seq, X_ridge, dQ, Q_curr, Q_next, t_index, bad_time_idx.astype(np.int64)


# ================== Dataset / Model ==================
class SeqResidualDataset(Dataset):
    def __init__(self, x_seq, y_residual):
        self.x = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y_residual, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class ResidualLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=4, output_size=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]
        return self.fc(h)


def main():
    set_seed(SEED)

    stage_data = []  # 每个元素：dict(stage_name, X_seq, X_ridge, dQ, t_index, ...)

    for i, fn in enumerate(DATA_FILES):
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件：{path}")

        df = read_csv_auto(path)
        arr = df.to_numpy(dtype=float)

        W = arr[:, GYRO_COLS]
        Q = normalize_quat(arr[:, Q_COLS])

        X_seq, X_ridge, dQ, Qc, Qn, t_idx, bad_time_idx = build_sequence_samples(
            W, Q, INTERVAL, DQ_THRESH, SEQ_LEN
        )

        print(f"[{fn}] 样本数 M={len(dQ)} | 跳过点数(dq>{DQ_THRESH} 的 t+1)={len(bad_time_idx)}")

        stage_data.append({
            "stage": os.path.splitext(fn)[0],
            "file": fn,
            "X_seq": X_seq,
            "X_ridge": X_ridge,
            "dQ": dQ,
            "t_index": t_idx
        })

    # --------- 每阶段内分割：20% Ridge, 30/20/30 LSTM ---------
    ridge_X_list, ridge_y_list = [], []
    lstm_tr_X, lstm_tr_y = [], []
    lstm_va_X, lstm_va_y = [], []
    lstm_te_X, lstm_te_y = [], []

    test_meta = []  # list of dict with stage, t_index, dQ_true, dQ_ridge, dQ_final

    for k, sd in enumerate(stage_data):
        stage = sd["stage"]
        X_seq = sd["X_seq"]
        X_ridge = sd["X_ridge"]
        y = sd["dQ"]
        t_idx = sd["t_index"]
        M = len(y)

        if M == 0:
            continue

        # 每个阶段用不同seed，保证可复现又不完全相同
        sseed = SEED + 1000 * (k + 1)
        rid_i, tr_i, va_i, te_i = stage_split_indices(
            M, sseed, R_STAGE_RIDGE, R_STAGE_LSTM_TRAIN, R_STAGE_LSTM_VAL, R_STAGE_LSTM_TEST
        )

        # Ridge
        ridge_X_list.append(X_ridge[rid_i])
        ridge_y_list.append(y[rid_i])

        # LSTM（学 residual，先占位，residual后面算）
        lstm_tr_X.append(X_seq[tr_i])
        lstm_va_X.append(X_seq[va_i])
        lstm_te_X.append(X_seq[te_i])

        # 先把真值保存，等 ridge 出来再算 residual
        lstm_tr_y.append(y[tr_i])
        lstm_va_y.append(y[va_i])
        lstm_te_y.append(y[te_i])

        # 测试集 meta 信息（用于画图/保存）
        test_meta.append({
            "stage": stage,
            "t_index": t_idx[te_i],
            "dQ_true": y[te_i],
            "X_ridge": X_ridge[te_i],
            "X_seq": X_seq[te_i]
        })

        print(f"  - [{stage}] ridge/train/val/test = {len(rid_i)}/{len(tr_i)}/{len(va_i)}/{len(te_i)}")

    # 合并 Ridge 训练集
    X_ridge_train = np.concatenate(ridge_X_list, axis=0)
    y_ridge_train = np.concatenate(ridge_y_list, axis=0)

    print(f"\n[ALL] Ridge训练样本总数 = {len(y_ridge_train)}")

    # --------- 1) 训练 Ridge（四阶段20%合并）---------
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=RIDGE_FIT_INTERCEPT)
    ridge.fit(X_ridge_train, y_ridge_train)
    coef = ridge.coef_  # (4,16)

    # 保存 Ridge 系数
    out_dir = os.path.abspath(DATA_DIR)
    coef_path = os.path.join(out_dir, "ridge_coef_4stages.csv")
    np.savetxt(coef_path, coef, delimiter=",")
    print(f"已保存 Ridge 系数矩阵：{coef_path}")

    # --------- 2) 构造 LSTM residual 数据集（用 Ridge 预测再取 residual）---------
    X_tr = np.concatenate(lstm_tr_X, axis=0) if lstm_tr_X else np.zeros((0, SEQ_LEN, 8), np.float32)
    X_va = np.concatenate(lstm_va_X, axis=0) if lstm_va_X else np.zeros((0, SEQ_LEN, 8), np.float32)
    X_te = np.concatenate(lstm_te_X, axis=0) if lstm_te_X else np.zeros((0, SEQ_LEN, 8), np.float32)

    y_tr_true = np.concatenate(lstm_tr_y, axis=0) if lstm_tr_y else np.zeros((0, 4), np.float32)
    y_va_true = np.concatenate(lstm_va_y, axis=0) if lstm_va_y else np.zeros((0, 4), np.float32)
    y_te_true = np.concatenate(lstm_te_y, axis=0) if lstm_te_y else np.zeros((0, 4), np.float32)

    # LSTM 需要 ridge 的对应 X_ridge 来计算 residual：这里我们直接用“阶段内同索引”的真值已拼好，
    # 但 X_ridge 没拼好，所以我们重算：用 Seq 的最后一拍抽出 Q/W 也能重建 X_ridge。
    # 为简单可靠：在构造样本时本来就有 X_ridge，但我们没同时拼到这里；
    # 这里重新从 test_meta 里拿，同时对 train/val 我们也需要 X_ridge -> 直接从 stage_data 重跑更麻烦。
    #
    # 最稳的办法：在上面分割时，把 X_ridge 对应 tr/va/te 也分别 append。这里我们补上这个结构：
    Xr_tr_list, Xr_va_list, Xr_te_list = [], [], []
    for k, sd in enumerate(stage_data):
        stage = sd["stage"]
        X_ridge = sd["X_ridge"]
        y = sd["dQ"]
        M = len(y)
        if M == 0:
            continue
        sseed = SEED + 1000 * (k + 1)
        rid_i, tr_i, va_i, te_i = stage_split_indices(M, sseed, R_STAGE_RIDGE, R_STAGE_LSTM_TRAIN, R_STAGE_LSTM_VAL, R_STAGE_LSTM_TEST)
        Xr_tr_list.append(X_ridge[tr_i])
        Xr_va_list.append(X_ridge[va_i])
        Xr_te_list.append(X_ridge[te_i])

    Xr_tr = np.concatenate(Xr_tr_list, axis=0)
    Xr_va = np.concatenate(Xr_va_list, axis=0)
    Xr_te = np.concatenate(Xr_te_list, axis=0)

    dQ_ridge_tr = ridge.predict(Xr_tr)
    dQ_ridge_va = ridge.predict(Xr_va)
    dQ_ridge_te = ridge.predict(Xr_te)

    res_tr = y_tr_true - dQ_ridge_tr
    res_va = y_va_true - dQ_ridge_va
    res_te = y_te_true - dQ_ridge_te

    print(f"\n[ALL] LSTM train/val/test = {len(res_tr)}/{len(res_va)}/{len(res_te)}")

    # --------- 3) 训练 LSTM（学 residual）---------
    train_ds = SeqResidualDataset(X_tr, res_tr)
    val_ds   = SeqResidualDataset(X_va, res_va)
    test_ds  = SeqResidualDataset(X_te, res_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualLSTM(input_size=8, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=4).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = None
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                va_losses.append(loss_fn(model(xb), yb).item())

        tr = float(np.mean(tr_losses)) if tr_losses else np.nan
        va = float(np.mean(va_losses)) if va_losses else np.nan

        if best_val is None or va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | residual_train={tr:.6e} | residual_val={va:.6e} | best_val={best_val:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------- 4) 测试：dQ 误差（pred vs true）---------
    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
        res_hat_te = model(X_te_t).cpu().numpy()

    dQ_final_te = dQ_ridge_te + res_hat_te

    mse_ridge = float(np.mean((y_te_true - dQ_ridge_te) ** 2))
    mse_final = float(np.mean((y_te_true - dQ_final_te) ** 2))

    print("\n=== Test Metrics (dQ) ===")
    print(f"MSE dQ (ridge only) : {mse_ridge:.6e}")
    print(f"MSE dQ (ridge+LSTM) : {mse_final:.6e}")

    # --------- 5) 保存测试集结果 + 画误差图 ---------
    err = dQ_final_te - y_te_true
    err_norm = np.linalg.norm(err, axis=1)

    out_csv = os.path.join(out_dir, "test_dQ_true_pred_4stages.csv")
    out_df = pd.DataFrame({
        "dQ_true0": y_te_true[:, 0],
        "dQ_true1": y_te_true[:, 1],
        "dQ_true2": y_te_true[:, 2],
        "dQ_true3": y_te_true[:, 3],
        "dQ_pred0": dQ_final_te[:, 0],
        "dQ_pred1": dQ_final_te[:, 1],
        "dQ_pred2": dQ_final_te[:, 2],
        "dQ_pred3": dQ_final_te[:, 3],
        "err0": err[:, 0],
        "err1": err[:, 1],
        "err2": err[:, 2],
        "err3": err[:, 3],
        "err_norm": err_norm
    })
    out_df.to_csv(out_csv, index=False)
    print(f"已保存测试集 dQ 真值/预测/误差：{out_csv}")

    # 误差范数图
    fig1 = plt.figure(figsize=(12, 3.8))
    plt.plot(err_norm)
    plt.title("Test Set dQ Error Norm (ridge + LSTM residual)")
    plt.xlabel("test sample index")
    plt.ylabel("||dQ_pred - dQ_true||")
    plt.grid(True)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "test_dQ_error_norm_4stages.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    print(f"已保存误差范数图：{fig1_path}")

    # 4分量误差图
    fig2, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for k in range(4):
        axes[k].plot(err[:, k])
        axes[k].set_ylabel(f"err dQ{k}")
        axes[k].grid(True)
    axes[-1].set_xlabel("test sample index")
    fig2.suptitle("Test Set dQ Component Errors (pred - true)")
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "test_dQ_component_error_4stages.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close(fig2)
    print(f"已保存分量误差图：{fig2_path}")

    # 保存 LSTM 模型
    model_path = os.path.join(out_dir, "lstm_residual_4stages.pth")
    torch.save(model.state_dict(), model_path)
    print(f"已保存 LSTM residual 模型：{model_path}")


if __name__ == "__main__":
    main()
