import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from OpenMORe.utilities import *
from Feature_Extraction import *          # Regression_FE / prediction_regression / monomial_set
from quaternion_lib import *


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enforce_quat_continuity(Q):
    """
    时间连续化：若相邻内积<0，翻转当前四元数。
    """
    Q = np.asarray(Q, dtype=np.float64).copy()
    for k in range(1, len(Q)):
        if np.dot(Q[k], Q[k-1]) < 0:
            Q[k] *= -1.0
    return Q.astype(np.float64)

def make_seq_features(Q_series, S_series_sub, t, seq_len, Q_last_override=None, S_last_override=None):
    """
    组一个 LSTM 输入序列 (seq_len, 7) = [Q(4), gyro_sub(3)]
    """
    start = t - seq_len + 1
    seq_Q = np.asarray(Q_series[start:t+1], dtype=np.float32)
    seq_S = np.asarray(S_series_sub[start:t+1], dtype=np.float32)
    X = np.hstack([seq_Q, seq_S]).astype(np.float32)

    if Q_last_override is not None:
        X[-1, :4] = np.asarray(Q_last_override, dtype=np.float32).reshape(-1)
    if S_last_override is not None:
        X[-1, 4:] = np.asarray(S_last_override, dtype=np.float32).reshape(-1)
    return X


# =========================
# LSTM residual（只负责学 residual）
# =========================
class SeqResidualDataset(Dataset):
    def __init__(self, x_seq, y_residual):
        self.x = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y_residual, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class ResidualLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=4, dropout=0.0, out_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        h = out[:, -1, :]
        return self.fc(h)

def _compute_x_stats(X_seq_train):
    """
    X_seq_train: (N, seq_len, 7)
    return x_mu, x_std with broadcast shape (1,1,7)
    """
    X = np.asarray(X_seq_train, dtype=np.float64)
    x_mu = X.mean(axis=(0, 1), keepdims=True)
    x_std = X.std(axis=(0, 1), keepdims=True) + 1e-12
    return x_mu.astype(np.float32), x_std.astype(np.float32)

def _compute_y_stats(Y_train):
    """
    Y_train: (N,4)
    return y_mu, y_std with shape (1,4)
    """
    Y = np.asarray(Y_train, dtype=np.float64)
    y_mu = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True) + 1e-12
    return y_mu.astype(np.float32), y_std.astype(np.float32)

def train_residual_lstm(X_seq, Y_res, seed=42, epochs=120, lr=1e-3, batch_size=128, split_ratio=0.8):
    """
    严格按时间顺序切分：前 split_ratio 为 train，后面为 val
    + 标准化（仅用 train 部分统计）
    返回：model, device, tr_idx, va_idx, x_mu, x_std, y_mu, y_std
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_seq = np.asarray(X_seq, dtype=np.float32)   # (N, seq_len, 7) 已经按时间构造
    Y_res = np.asarray(Y_res, dtype=np.float32)   # (N, 4)

    n = X_seq.shape[0]
    if n < 20:
        return None, device, None, None, None, None, None, None

    n_tr = int(split_ratio * n)
    n_tr = max(1, min(n-1, n_tr))  # 保证 train/val 都非空
    tr = np.arange(0, n_tr)
    va = np.arange(n_tr, n)

    print(f"    [LSTM split TIME] total_seq={n} | train={len(tr)} | val={len(va)}")

    # ===== 标准化统计（只用 train）=====
    x_mu, x_std = _compute_x_stats(X_seq[tr])
    y_mu, y_std = _compute_y_stats(Y_res[tr])

    # ===== 标准化后的数据用于训练 =====
    Xn = (X_seq - x_mu) / x_std
    Yn = (Y_res - y_mu) / y_std

    # train loader 可以 shuffle（不影响“切分按时间”，也不会泄漏到val）
    dl_tr = DataLoader(SeqResidualDataset(Xn[tr], Yn[tr]), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(SeqResidualDataset(Xn[va], Yn[va]), batch_size=batch_size, shuffle=False)

    model = ResidualLSTM(input_size=X_seq.shape[-1], hidden_size=64, num_layers=4, dropout=0.0, out_dim=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = None
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in dl_tr:
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
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device)
                va_losses.append(loss_fn(model(xb), yb).item())

        tr_m = float(np.mean(tr_losses)) if tr_losses else np.nan
        va_m = float(np.mean(va_losses)) if va_losses else np.nan

        if best_val is None or va_m < best_val:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 20 == 0:
            print(f"    [LSTM] Epoch {ep:03d} | train={tr_m:.3e} | val={va_m:.3e} | best={best_val:.3e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, device, tr, va, x_mu, x_std, y_mu, y_std

def lstm_predict_one(model, device, x_seq_1, x_mu=None, x_std=None, y_mu=None, y_std=None):
    """
    x_seq_1: (seq_len,7) 原始尺度
    返回：residual_hat（真实尺度，已反标准化）
    """
    if model is None:
        return np.zeros(4, dtype=np.float32)

    x = np.asarray(x_seq_1, dtype=np.float32)

    if (x_mu is not None) and (x_std is not None):
        x = (x - x_mu.reshape(1, -1)) / x_std.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x[None, :, :], dtype=torch.float32).to(device)
        y_norm = model(xb).detach().cpu().numpy().reshape(-1).astype(np.float32)

    if (y_mu is not None) and (y_std is not None):
        y = y_norm * y_std.reshape(-1) + y_mu.reshape(-1)
    else:
        y = y_norm

    return y.astype(np.float32)

def lstm_predict_batch(model, device, X_seq, x_mu=None, x_std=None, y_mu=None, y_std=None, batch_size=1024):
    """
    X_seq: (N, seq_len,7) 原始尺度
    返回：residual_hat（真实尺度，已反标准化）
    """
    if model is None:
        return np.zeros((X_seq.shape[0], 4), dtype=np.float32)

    X = np.asarray(X_seq, dtype=np.float32)
    if (x_mu is not None) and (x_std is not None):
        X = (X - x_mu) / x_std

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            outs.append(model(xb).detach().cpu().numpy())
    y_norm = np.vstack(outs).astype(np.float32)

    if (y_mu is not None) and (y_std is not None):
        y = y_norm * y_std + y_mu
    else:
        y = y_norm

    return y.astype(np.float32)


# =========================
# Quaternion_update：加 residual
# =========================
def Quaternion_update(Q_p, S, Coef, sf, delta_T, dQ_residual=None):
    Q_pt = Q_p.reshape(1, -1)
    S_pt = S.reshape(1, -1)

    if dQ_residual is None:
        dQ_residual = np.zeros((1, 4), dtype=np.float64)
    else:
        dQ_residual = np.asarray(dQ_residual, dtype=np.float64).reshape(1, 4)

    M_monomial = monomial_set(Q_pt, S_pt) * sf

    K_1 = np.matmul(M_monomial, Coef.T) / sf + dQ_residual
    Q_2 = np.asarray(Q_pt + 0.5 * delta_T * K_1)
    K_2 = np.matmul(monomial_set(Q_2, S_pt) * sf, Coef.T) / sf + dQ_residual

    Q_3 = np.asarray(Q_pt + 0.5 * delta_T * K_2)
    K_3 = np.matmul(monomial_set(Q_3, S_pt) * sf, Coef.T) / sf + dQ_residual

    Q_4 = np.asarray(Q_pt + delta_T * K_3)
    K_4 = np.matmul(monomial_set(Q_4, S_pt) * sf, Coef.T) / sf + dQ_residual

    Q_u = Q_pt + delta_T * (K_1 + 2 * K_2 + 2 * K_3 + K_4) / 6
    return Q_u


# =========================
# probability_cal：保留你原结构，只加 residual（✅ 使用对应假设的标准化参数）
# =========================
def probability_cal(alpha, z_t, sigma_list, M_list, Coef_list, sf_list, delta_T,
                    lstm_models=None, lstm_devices=None,
                    Array_Q=None, Array_S=None, seq_len=10,
                    x_mu_list=None, x_std_list=None, y_mu_list=None, y_std_list=None):

    S_p = M_list[0].reshape(1, -1)
    Q_p = M_list[1].reshape(1, -1)
    Q_t = M_list[2].reshape(1, -1)

    sigma_g = sigma_list[0]
    sigma_theta = sigma_list[1]

    Coef_alpha = Coef_list[alpha]
    Coef_zt = Coef_list[z_t]
    sf_alpha = sf_list[alpha]
    sf_zt = sf_list[z_t]

    list_S_column = np.arange(0, 4)
    list_ag_column = np.delete(list_S_column, alpha)
    list_zg_column = np.delete(list_S_column, z_t)

    vector_ag_sigma = sigma_g * np.random.randn(1, 3)
    vector_zg_sigma = vector_ag_sigma.copy()

    S_a_sample = S_p[:, list_ag_column] - vector_ag_sigma
    S_z_sample = S_p[:, list_zg_column] - vector_zg_sigma

    vector_sigma_theta = sigma_theta * np.random.randn(1, 3)
    s2 = float(np.sum(vector_sigma_theta ** 2))
    q0_sigma = np.sqrt(max(0.0, 1.0 - s2))
    vector_sigma_q = np.append(vector_sigma_theta, q0_sigma)

    Q_h = qmul(Q_p.reshape(-1), qconj(vector_sigma_q.reshape(-1)))

    dQ_res_alpha = np.zeros(4, dtype=np.float32)
    dQ_res_zt = np.zeros(4, dtype=np.float32)

    if (lstm_models is not None) and (Array_Q is not None) and (Array_S is not None):
        i = int(M_list[3])
        if i >= seq_len - 1:
            # alpha
            model_a = lstm_models[alpha]
            dev_a = lstm_devices[alpha]
            S_sub_a = Array_S[:, list_ag_column]
            X_a = make_seq_features(Array_Q, S_sub_a, i, seq_len,
                                    Q_last_override=Q_h,
                                    S_last_override=S_a_sample.reshape(-1))

            if x_mu_list is not None:
                dQ_res_alpha = lstm_predict_one(model_a, dev_a, X_a,
                                               x_mu=x_mu_list[alpha], x_std=x_std_list[alpha],
                                               y_mu=y_mu_list[alpha], y_std=y_std_list[alpha])
            else:
                dQ_res_alpha = lstm_predict_one(model_a, dev_a, X_a)

            # zt
            model_z = lstm_models[z_t]
            dev_z = lstm_devices[z_t]
            S_sub_z = Array_S[:, list_zg_column]
            X_z = make_seq_features(Array_Q, S_sub_z, i, seq_len,
                                    Q_last_override=Q_h,
                                    S_last_override=S_z_sample.reshape(-1))

            if x_mu_list is not None:
                dQ_res_zt = lstm_predict_one(model_z, dev_z, X_z,
                                            x_mu=x_mu_list[z_t], x_std=x_std_list[z_t],
                                            y_mu=y_mu_list[z_t], y_std=y_std_list[z_t])
            else:
                dQ_res_zt = lstm_predict_one(model_z, dev_z, X_z)

    Qa_up = Quaternion_update(Q_h, S_a_sample, Coef_alpha, sf_alpha, delta_T, dQ_residual=dQ_res_alpha)
    Qz_up = Quaternion_update(Q_h, S_z_sample, Coef_zt, sf_zt, delta_T, dQ_residual=dQ_res_zt)

    Qa_mean = np.array([Qa_up[0, 0], Qa_up[0, 1], Qa_up[0, 2], Qa_up[0, 3]])
    Qz_mean = np.array([Qz_up[0, 0], Qz_up[0, 1], Qz_up[0, 2], Qz_up[0, 3]])

    Cov_Q = sigma_theta ** 2 / 4 * np.eye(4)

    p_alpha = st.multivariate_normal.pdf(Q_t, mean=Qa_mean, cov=Cov_Q)
    p_zt = st.multivariate_normal.pdf(Q_t, mean=Qz_mean, cov=Cov_Q)

    return p_alpha, p_zt


# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed(42)

    interval = 3
    d2r = np.arccos(-1.0) / 180

    # EP_L_all.csv / AM_L_all.csv / FAA_L_all.csv / OE_L_all.csv
    file_options = {
        "path_to_file": r".",
        "input_file_name": "data_maneuvering_up.csv"
    }
    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    X = X[100:, :] # 修改这里的行数，假装在做滑窗
    index_s = 4
    index_e = 16

    # ===== 差分 + 删除异常 dq 点 =====
    dq = (X[1:, index_s:index_e] - X[0:-1, index_s:index_e]) / interval
    index_delete = np.unique(np.where(np.abs(dq) > 0.01)[0])

    dq = np.delete(dq, index_delete, axis=0)
    dq = np.asarray(dq, dtype=np.float64)

    q1 = X[1:, index_s:index_s + 4]
    q1 = np.delete(q1, index_delete, axis=0)
    for i in range(q1.shape[0]):
        q1[i, :] = qnorm(q1[i, :])
    q1 = enforce_quat_continuity(q1)

    AR_m = X[1:, 0:4] * d2r
    AR_m = np.delete(AR_m, index_delete, axis=0)


    N_total = dq.shape[0]
    index_t = min(int(0.8 * N_total),400)    # train length
    if index_t < 20:
        raise ValueError(f"Train too short after cleaning: N_train={index_t}")
    print(f"[Split] cleaned_total={N_total} | train={index_t} | test={N_total-index_t}")

    dq_training = dq[:index_t, 0:4]
    q1_training = q1[:index_t, :]
    AR_training = AR_m[:index_t, :]

    dq_test = dq[index_t:index_t+100, 0:4]
    q1_test = q1[index_t:index_t+100, :]
    AR_test = AR_m[index_t:index_t+100, :]

    dq_test = np.asarray(dq_test, dtype=np.float64)
    Datalen_test = dq_test.shape[0]

    # =========================
    # Ridge：4种“删一路gyro”的假设
    # =========================
    boosting_status = 1

    Indices_s = [1, 2, 3]
    Coef_0, mse_0, _, _, sf_0 = Regression_FE(q1_training, dq_training, AR_training[:, Indices_s], boosting_status)

    Indices_s = [0, 2, 3]
    Coef_1, mse_1, _, _, sf_1 = Regression_FE(q1_training, dq_training, AR_training[:, Indices_s], boosting_status)

    Indices_s = [0, 1, 3]
    Coef_2, mse_2, _, _, sf_2 = Regression_FE(q1_training, dq_training, AR_training[:, Indices_s], boosting_status)

    Indices_s = [0, 1, 2]
    Coef_3, mse_3, _, _, sf_3 = Regression_FE(q1_training, dq_training, AR_training[:, Indices_s], boosting_status)

    Coef_list = [Coef_0, Coef_1, Coef_2, Coef_3]
    sf_list = [sf_0, sf_1, sf_2, sf_3]

    # =========================
    # Ridge 测试预测
    # =========================
    Indices_s = [1, 2, 3]
    y_pred0, _ = prediction_regression(q1_test, dq_test, AR_test[:, Indices_s], Coef_0, sf_0)
    y_pred0 = np.asarray(y_pred0, dtype=np.float64)

    Indices_s = [0, 2, 3]
    y_pred1, _ = prediction_regression(q1_test, dq_test, AR_test[:, Indices_s], Coef_1, sf_1)
    y_pred1 = np.asarray(y_pred1, dtype=np.float64)

    Indices_s = [0, 1, 3]
    y_pred2, _ = prediction_regression(q1_test, dq_test, AR_test[:, Indices_s], Coef_2, sf_2)
    y_pred2 = np.asarray(y_pred2, dtype=np.float64)

    Indices_s = [0, 1, 2]
    y_pred3, _ = prediction_regression(q1_test, dq_test, AR_test[:, Indices_s], Coef_3, sf_3)
    y_pred3 = np.asarray(y_pred3, dtype=np.float64)

    # =========================
    # 训练 LSTM residual（每个假设一套）
    # =========================
    SEQ_LEN = 10
    EPOCHS = 120
    LR = 1e-3
    BATCH = 128

    if index_t < SEQ_LEN:
        raise ValueError(f"N_train={index_t} < SEQ_LEN={SEQ_LEN}, cannot build sequences.")

    lstm_models = []
    lstm_devices = []
    x_mu_list, x_std_list = [], []
    y_mu_list, y_std_list = [], []

    for h in range(4):
        cols = list(np.delete(np.arange(0, 4), h))

        y_ridge_tr, _ = prediction_regression(q1_training, dq_training, AR_training[:, cols], Coef_list[h], sf_list[h])
        y_ridge_tr = np.asarray(y_ridge_tr, dtype=np.float64)

        dq_training_arr = np.asarray(dq_training, dtype=np.float64)
        res_tr = dq_training_arr - y_ridge_tr  # (N,4)

        X_seq_list = []
        y_seq_list = []
        t_list = []

        for t in range(SEQ_LEN - 1, res_tr.shape[0]):
            seq_Q = q1_training[t - SEQ_LEN + 1:t + 1, :]
            seq_S = AR_training[t - SEQ_LEN + 1:t + 1, :][:, cols]
            X_seq_list.append(np.hstack([seq_Q, seq_S]))
            y_seq_list.append(res_tr[t, :])
            t_list.append(t)

        X_seq = np.asarray(X_seq_list, dtype=np.float32)  # (Nseq, seq_len, 7)
        y_seq = np.asarray(y_seq_list, dtype=np.float32)  # (Nseq, 4)
        t_list = np.asarray(t_list, dtype=np.int64)

        print(f"\n=== Hypothesis {h} | Ridge mse={ [mse_0,mse_1,mse_2,mse_3][h]:.3e} | LSTM samples={len(y_seq)} ===")

        model_h, dev_h, tr_idx, va_idx, x_mu, x_std, y_mu, y_std = train_residual_lstm(
            X_seq, y_seq, seed=42, epochs=EPOCHS, lr=LR, batch_size=BATCH
        )

        lstm_models.append(model_h)
        lstm_devices.append(dev_h)
        x_mu_list.append(x_mu); x_std_list.append(x_std)
        y_mu_list.append(y_mu); y_std_list.append(y_std)

       
        if model_h is not None and tr_idx is not None:
            res_hat_all = lstm_predict_batch(model_h, dev_h, X_seq,
                                             x_mu=x_mu, x_std=x_std, y_mu=y_mu, y_std=y_std, batch_size=1024)

            def mse(a, b):
                a = np.asarray(a, dtype=np.float64)
                b = np.asarray(b, dtype=np.float64)
                return float(np.mean(np.square(a - b)))

            mse_res_tr = mse(y_seq[tr_idx], res_hat_all[tr_idx])
            mse_res_va = mse(y_seq[va_idx], res_hat_all[va_idx])

            t_tr = t_list[tr_idx]
            t_va = t_list[va_idx]

            dq_true_tr = dq_training_arr[t_tr]
            dq_true_va = dq_training_arr[t_va]
            dq_ridge_tr = y_ridge_tr[t_tr]
            dq_ridge_va = y_ridge_tr[t_va]

            dq_final_tr = dq_ridge_tr + res_hat_all[tr_idx]
            dq_final_va = dq_ridge_va + res_hat_all[va_idx]

            print(f"    [Residual MSE real] train={mse_res_tr:.3e} | val={mse_res_va:.3e}")
            print(f"    [dQ MSE] Ridge  train={mse(dq_true_tr, dq_ridge_tr):.3e} | val={mse(dq_true_va, dq_ridge_va):.3e}")
            print(f"    [dQ MSE] Final  train={mse(dq_true_tr, dq_final_tr):.3e} | val={mse(dq_true_va, dq_final_va):.3e}")

    # =========================
    # 用 LSTM 预测 test 残差 -> Ridge+LSTM 的 dq 预测（使用标准化参数）
    # =========================
    def predict_dq_ridge_lstm(h, y_ridge_test):
        cols = list(np.delete(np.arange(0, 4), h))
        model = lstm_models[h]
        dev = lstm_devices[h]
        y_ridge_test = np.asarray(y_ridge_test, dtype=np.float64)

        x_mu = x_mu_list[h]; x_std = x_std_list[h]
        y_mu = y_mu_list[h]; y_std = y_std_list[h]

        res_hat = np.zeros_like(y_ridge_test, dtype=np.float64)

        for t in range(Datalen_test):
            if t < SEQ_LEN - 1 or model is None:
                res_hat[t, :] = 0.0
            else:
                seq_Q = q1_test[t - SEQ_LEN + 1:t + 1, :]
                seq_S = AR_test[t - SEQ_LEN + 1:t + 1, :][:, cols]
                Xs = np.hstack([seq_Q, seq_S]).astype(np.float32)

                r = lstm_predict_one(model, dev, Xs,
                                     x_mu=x_mu, x_std=x_std, y_mu=y_mu, y_std=y_std)
                res_hat[t, :] = r.astype(np.float64)

        return y_ridge_test + res_hat

    y_pred0_nn = predict_dq_ridge_lstm(0, y_pred0)
    y_pred1_nn = predict_dq_ridge_lstm(1, y_pred1)
    y_pred2_nn = predict_dq_ridge_lstm(2, y_pred2)
    y_pred3_nn = predict_dq_ridge_lstm(3, y_pred3)

    # ====== Diagnostic: Ridge vs Residual vs Final (per hypothesis) ======
    pred_ridge_list = [y_pred0, y_pred1, y_pred2, y_pred3]
    pred_final_list = [y_pred0_nn, y_pred1_nn, y_pred2_nn, y_pred3_nn]

    for h in range(4):
        y_ridge = np.asarray(pred_ridge_list[h], dtype=np.float64)
        y_final = np.asarray(pred_final_list[h], dtype=np.float64)

        r_true = dq_test - y_ridge
        r_hat  = y_final - y_ridge

        mse_ridge    = np.mean(np.square(dq_test - y_ridge))
        mse_residual = np.mean(np.square(r_true - r_hat))
        mse_final    = np.mean(np.square(dq_test - y_final))
        res_energy   = np.mean(np.square(r_true))

        print(f"[Hypo {h}] mse_ridge={mse_ridge:.3e} | mse_residual={mse_residual:.3e} | "
              f"mse_final={mse_final:.3e} | res_energy={res_energy:.3e}")

    # =========================
    # 画 dq（你原来的写法：只画 Ridge+LSTM）
    # =========================
    for comp in range(4):
        fig = plt.figure()
        ax = plt.axes()

        ax.plot(y_pred0_nn[:, comp], 'r--',  label='R0+LSTM')
        ax.plot(y_pred1_nn[:, comp], 'b--',  label='R1+LSTM')
        ax.plot(y_pred2_nn[:, comp], 'c--',  label='R2+LSTM')
        ax.plot(y_pred3_nn[:, comp], 'y--',  label='R3+LSTM')

        ax.plot(dq_test[:, comp], 'k', label='differentiation')

        ax.set_xlabel('time')
        ax.set_ylabel(f'dq_{comp}')
        plt.title(f'Ridge vs Ridge+LSTM for dq{comp}')
        ax.legend()

    # =========================
    # Fault detection（parity）
    # =========================
    sigma_g = 1e-5 * d2r
    sigma_theta = 5 / 3600 * d2r
    sigma_list = [sigma_g, sigma_theta]

    # ✅ index_f 自适应，避免测试集变短越界；同时保证 index_f>=1（后面用 index_f-1）
    index_f = max(1, min(50, Datalen_test // 2))
    print(f"[Fault] index_f={index_f} (test_len={Datalen_test})")

    f_mag = 0 * sigma_g
    AR_test[index_f:, 3] += f_mag

    V_matrix = np.array([[0.40822003, 0.40822003, 0.40822003, 0.70715573]])
    vector_p = np.matmul(AR_test, V_matrix.T)

    T_u = 3 * sigma_g * np.ones(vector_p.shape)
    T_l = -3 * sigma_g * np.ones(vector_p.shape)

    fig = plt.figure()
    ax_1 = plt.axes()
    ax_1.plot(vector_p, 'b', label='parity vector')
    ax_1.plot(T_u, 'r--')
    ax_1.plot(T_l, 'r--')
    plt.title('statistic for detection')
    ax_1.legend()

    # =========================
    # Fault Isolation（M-H）✅ 使用标准化参数
    # =========================
    Array_S = AR_test[index_f:, :]
    Array_Q = q1_test[index_f - 1:, :]

    iter_out = min(Datalen_test - index_f,100)
    seq_pdf = np.zeros([iter_out, 4])

    iter_inner = 100

    for i in range(iter_out):
        Samples = np.zeros([iter_inner, 1])
        z_t = 0

        for j in range(iter_inner):
            alpha = np.random.randint(0, 4)
            while z_t == alpha:
                alpha = np.random.randint(0, 4)

            M_list = [Array_S[i, :], Array_Q[i, :], Array_Q[i + 1, :], i]

            p_alpha, p_zt = probability_cal(
                alpha, z_t, sigma_list, M_list,
                Coef_list, sf_list, interval,
                lstm_models=lstm_models,
                lstm_devices=lstm_devices,
                Array_Q=Array_Q,
                Array_S=Array_S,
                seq_len=SEQ_LEN,
                x_mu_list=x_mu_list, x_std_list=x_std_list,
                y_mu_list=y_mu_list, y_std_list=y_std_list
            )

            if p_alpha > p_zt * np.random.rand():
                z_t = alpha
            Samples[j] = z_t

        for k in range(4):
            seq_pdf[i, k] = (Samples == k).sum() / iter_inner

    fig = plt.figure()
    ax_1 = plt.axes()
    ax_1.plot(seq_pdf[:, 0], 'r', label='g_0')
    ax_1.plot(seq_pdf[:, 1], 'b', label='g_1')
    ax_1.plot(seq_pdf[:, 2], 'c', label='g_2')
    ax_1.plot(seq_pdf[:, 3], 'y', label='g_3')
    plt.title('Fault Probability for gyros (Ridge + LSTM residual)')
    ax_1.legend()
    plt.show()
