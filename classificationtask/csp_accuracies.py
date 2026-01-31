# -*- coding: utf-8 -*-
"""
CSP baseline replicated to match the "graph features" experiment pipeline.

Protocol (per subject):
1) Load OGFS .mat (L/R trials), bandpass 8–30 Hz, optional ASR
2) Crop/pad to per-subject nominal length (mode length)
3) Build per-trial CSP features:
      - Fit CSP on TRAIN split only
      - Transform train/test into log-variance CSP space
4) (OPTIONAL) replicate your controversial alternating reorder step
5) Temporal split: first 20% train, last 80% test (no shuffle)
6) Standardise (train-only), optional PCA for classification
7) GridSearchCV on training only, evaluate on test
8) Save CSV summaries

Labels:
- This script uses 0=Left, 1=Right internally (to match your graph script).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, filtfilt

from mne.decoding import CSP

from asrpy.asr import asr_calibrate, asr_process

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# Config
# -----------------------------
DATA_DIR = r"C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\OG_Full_Data"
SUBJECT_IDS = [1, 2, 5, 9, 21, 31, 34, 39]

FS = 256
BPF_LO, BPF_HI = 8, 30

USE_ASR = True
ASR_CUTOFF = 20.0

DROP_TOO_SHORT = True
MIN_LEN_FRACTION = 0.5

# CSP features (for classification)
N_CSP_FEATURES = 2  # number of CSP components used as features

# EXACT replication toggle (this is the “alternating order” step)
REPLICATE_ALTERNATING_REORDER = True

# Optional: plot CSP patterns trained on TRAIN split (interpretability)
PLOT_CSP_PATTERNS = False


# -----------------------------
# Data IO + preprocessing
# -----------------------------
def remove_last_entry_if_all_zeros(data_list):
    if len(data_list) > 0 and np.all(data_list[-1] == 0):
        return data_list[:-1]
    return data_list


def load_subjects_ogfs_mat(data_dir, subject_ids):
    out = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".mat"):
            continue

        subj_str = filename[len("S"):-len(".mat")]
        if not (subj_str.isdigit() and int(subj_str) in subject_ids):
            continue

        mat = loadmat(os.path.join(data_dir, filename))
        varname = f"Subject{subj_str}"
        if varname not in mat:
            raise KeyError(f"Expected variable '{varname}' in {filename}")

        void_array = mat[varname]
        sid = f"S{subj_str}"
        out.setdefault(sid, {"L": [], "R": []})

        for item in void_array[0]:
            out[sid]["L"].append(item["L"])
            out[sid]["R"].append(item["R"])

        out[sid]["L"] = remove_last_entry_if_all_zeros(out[sid]["L"])
        out[sid]["R"] = remove_last_entry_if_all_zeros(out[sid]["R"])

    return out


def bandpass_filter_trials(data_by_subject, low, high, sfreq):
    nyq = 0.5 * sfreq
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")

    out = {}
    for sid, d in data_by_subject.items():
        out[sid] = {"L": [], "R": []}
        for lab in ["L", "R"]:
            for tr in d[lab]:
                tr = tr[:, :19]  # keep first 19 channels
                y = np.zeros_like(tr, dtype=np.float64)
                for ch in range(tr.shape[1]):
                    y[:, ch] = filtfilt(b, a, tr[:, ch])
                out[sid][lab].append(y)
    return out


def run_asr_subjectwise(filtered_data_by_subject, sfreq=256.0, cutoff=20.0):
    out = {}
    for sid, d in filtered_data_by_subject.items():
        L_trials = d["L"]
        R_trials = d["R"]
        all_trials = L_trials + R_trials
        if len(all_trials) == 0:
            continue

        lengths = [tr.shape[0] for tr in all_trials]
        X = np.concatenate(all_trials, axis=0).T  # [n_ch x n_samp]

        M, T = asr_calibrate(X, sfreq=sfreq, cutoff=cutoff)
        Xc = asr_process(X, sfreq=sfreq, M=M, T=T).T

        cleaned = []
        idx = 0
        for L in lengths:
            cleaned.append(Xc[idx:idx + L, :])
            idx += L

        nL = len(L_trials)
        out[sid] = {"L": cleaned[:nL], "R": cleaned[nL:]}
        print(f"ASR done for {sid}: {len(out[sid]['L'])} L, {len(out[sid]['R'])} R")

    return out


def choose_target_length_mode(L_trials, R_trials):
    lengths = [tr.shape[0] for tr in (L_trials + R_trials)]
    lengths = np.asarray(lengths, dtype=int)
    vals, counts = np.unique(lengths, return_counts=True)
    return int(vals[np.argmax(counts)])


def crop_pad_trials(trials, target_len, drop_too_short=True, min_len_fraction=0.5):
    new_trials = []
    for tr in trials:
        n_samp, n_ch = tr.shape
        if n_samp > target_len:
            new_trials.append(tr[:target_len, :])
        elif n_samp < target_len:
            if drop_too_short and n_samp < min_len_fraction * target_len:
                continue
            pad = np.zeros((target_len - n_samp, n_ch), dtype=tr.dtype)
            new_trials.append(np.vstack([tr, pad]))
        else:
            new_trials.append(tr)
    return new_trials


def build_epochs_for_subject(d, target_len):
    """
    Returns:
      X: [n_trials, n_channels, n_times]
      y: [n_trials] with 0=Left, 1=Right  (matches your graph script)
    """
    L = crop_pad_trials(d["L"], target_len, DROP_TOO_SHORT, MIN_LEN_FRACTION)
    R = crop_pad_trials(d["R"], target_len, DROP_TOO_SHORT, MIN_LEN_FRACTION)

    if len(L) == 0 or len(R) == 0:
        return None, None

    X = []
    y = []

    for tr in L:
        X.append(tr.T)
        y.append(0)

    for tr in R:
        X.append(tr.T)
        y.append(1)

    X = np.stack(X, axis=0).astype(np.float64)
    y = np.asarray(y, dtype=int)
    return X, y


# -----------------------------
# Graph-script helpers replicated
# -----------------------------
def temporal_split(X, y, train_ratio=0.2):
    n_trials = X.shape[0]
    split_idx = int(np.floor(train_ratio * n_trials))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def make_alternating_order(labels, subj_id=None):
    labels = np.asarray(labels)
    idx0 = np.sort(np.where(labels == 0)[0])
    idx1 = np.sort(np.where(labels == 1)[0])

    if len(idx0) == 0 or len(idx1) == 0:
        if subj_id is not None:
            print(f"[WARN] {subj_id}: cannot alternate order – one class empty.")
        return np.arange(len(labels))

    m = min(len(idx0), len(idx1))
    order = []
    for k in range(m):
        order.append(idx0[k])
        order.append(idx1[k])

    if len(idx0) > m:
        order.extend(idx0[m:])
    if len(idx1) > m:
        order.extend(idx1[m:])

    order = np.asarray(order, dtype=int)
    assert len(order) == len(labels)
    return order


def get_models_and_param_grids():
    return {
        "LogReg_L2": (
            LogisticRegression(max_iter=2000, n_jobs=-1),
            {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"]},
        ),
        "LinearSVM": (
            LinearSVC(max_iter=5000),
            {"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "RBF_SVM": (
            SVC(kernel="rbf"),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.001]},
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "max_features": ["sqrt", "log2"]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
    }


def compute_binary_metrics(y_true, y_pred, positive_label=1):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_label], average="binary", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def evaluate_models_temporal(X, y, subj_id, feature_type="CSP features"):
    X_train, X_test, y_train, y_test = temporal_split(X, y, train_ratio=0.2)

    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    print(f"\n=== {feature_type} – Subject {subj_id} ===")
    print(f"Train trials: {len(y_train)}, Test trials: {len(y_test)}")
    print(f"  Train class dist: {dict(zip(train_classes, train_counts))}")
    print(f"  Test  class dist: {dict(zip(test_classes, test_counts))}")

    if len(train_classes) < 2:
        print(f"[WARN] {subj_id}: training split has only one class under temporal 20% rule.")
        return {}

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    use_pca = X_train_std.shape[1] > 5
    if use_pca:
        n_samples_train, n_features_train = X_train_std.shape
        max_components = min(n_samples_train - 1, n_features_train)
        n_components = min(30, max_components)

        if n_components >= 2:
            pca = PCA(n_components=n_components)
            X_train_clf = pca.fit_transform(X_train_std)
            X_test_clf = pca.transform(X_test_std)
            print(f"  [{feature_type}] Using PCA n_components={n_components}")
        else:
            X_train_clf, X_test_clf = X_train_std, X_test_std
            print(f"  [{feature_type}] Skipping PCA (insufficient rank)")
    else:
        X_train_clf, X_test_clf = X_train_std, X_test_std
        print(f"  [{feature_type}] Not using PCA (features <= 5)")

    models_and_grids = get_models_and_param_grids()
    results = {}

    for name, (base_est, param_grid) in models_and_grids.items():
        print(f"  -> Optimising {name} for {feature_type} – {subj_id}")

        grid = GridSearchCV(
            estimator=base_est,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            error_score="raise",
        )
        grid.fit(X_train_clf, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_clf)
        metrics = compute_binary_metrics(y_test, y_pred, positive_label=1)
        results[name] = metrics

        print(
            f"     Test acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}, "
            f"kappa={metrics['kappa']:.3f}, sens={metrics['sensitivity']:.3f}, spec={metrics['specificity']:.3f}"
        )

    return results


# -----------------------------
# CSP feature extraction that matches the experiment
# -----------------------------
def extract_csp_features_temporal(X_epochs, y, subj_id, n_components=4, plot_patterns=False):
    """
    IMPORTANT: Fit CSP on TRAIN ONLY (temporal split) to avoid test leakage.
    Returns:
      X_feat: full feature matrix aligned to original trial order (train+test)
    """
    X_train, X_test, y_train, y_test = temporal_split(X_epochs, y, train_ratio=0.2)

    if len(np.unique(y_train)) < 2:
        print(f"[WARN] {subj_id}: cannot fit CSP (train split has one class).")
        return None

    csp = CSP(
        n_components=n_components,
        log=True,
        norm_trace=True,
        component_order="mutual_info",
    )
    csp.fit(X_train, y_train)

    if plot_patterns:
        # Patterns are interpretable but not part of the predictive protocol
        fig = csp.plot_patterns(info=None, show=False)  # requires MNE info if you want topomaps
        fig.suptitle(f"{subj_id} — CSP patterns (fit on TRAIN split)", fontsize=14)
        plt.show()

    F_train = csp.transform(X_train)  # [n_train, n_components]
    F_test = csp.transform(X_test)    # [n_test, n_components]

    X_feat = np.vstack([F_train, F_test])
    return X_feat


# -----------------------------
# Main
# -----------------------------
data_by_subject = load_subjects_ogfs_mat(DATA_DIR, SUBJECT_IDS)
data_by_subject = bandpass_filter_trials(data_by_subject, BPF_LO, BPF_HI, FS)

if USE_ASR:
    data_by_subject = run_asr_subjectwise(data_by_subject, sfreq=FS, cutoff=ASR_CUTOFF)

all_results_csp = {}

for sid, d in data_by_subject.items():
    if len(d["L"]) == 0 or len(d["R"]) == 0:
        print(f"Skipping {sid}: missing L or R trials")
        continue

    target_len = choose_target_length_mode(d["L"], d["R"])
    X_epochs, y = build_epochs_for_subject(d, target_len)

    if X_epochs is None:
        print(f"Skipping {sid}: not enough usable trials after crop/pad")
        continue

    print(f"\n########## Subject {sid} ##########")
    print(f"  Epochs: {X_epochs.shape} (trials, ch, time) | y: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Replicate your alternating reorder step (if enabled)
    if REPLICATE_ALTERNATING_REORDER:
        order = make_alternating_order(y, subj_id=sid)
        X_epochs = X_epochs[order]
        y = y[order]
        print("  [Reorder] Applied alternating 0/1 order before temporal split (EXACT replication mode).")

    # Extract CSP features with CSP fit on TRAIN only
    X_csp = extract_csp_features_temporal(
        X_epochs, y, sid,
        n_components=N_CSP_FEATURES,
        plot_patterns=PLOT_CSP_PATTERNS
    )
    if X_csp is None:
        continue

    # Now run the exact same classifier pipeline you used for nodes/edges/concat
    res = evaluate_models_temporal(X_csp, y, sid, feature_type=f"CSP({N_CSP_FEATURES})")
    all_results_csp[sid] = res


# -----------------------------
# Save CSV like your graph script
# -----------------------------
import pandas as pd

def results_to_dataframe(all_results, feature_type):
    rows = []
    for subj_id, model_dict in all_results.items():
        for model_name, metrics in model_dict.items():
            row = {"Subject": subj_id, "Model": model_name, "FeatureType": feature_type}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)

df_csp = results_to_dataframe(all_results_csp, f"CSP({N_CSP_FEATURES})")
print(df_csp.head())

df_csp.to_csv("CSP_RESULTS_FULL.csv", index=False)
print("\nSaved CSV: CSP_RESULTS_FULL.csv")
