# -*- coding: utf-8 -*-
"""
PSD baseline replicating the CSP experiment pipeline exactly.

Protocol (per subject) — identical to csp_accuracies.py:
1) Load OGFS .mat (L/R trials), bandpass 8-30 Hz, ASR
2) Crop/pad to per-subject modal length
3) Alternating reorder (L0,R0,L1,R1,...) — same as CSP script
4) Temporal split: first 20% train, last 80% test (no shuffle)
5) Extract sub-band PSD features (alpha/mu 8-12, low-beta 13-20, high-beta 20-30)
   — feature extractor fit on TRAIN only (scaler, PCA)
6) GridSearchCV on training only, evaluate on test
7) Per-subject feature space plot (session colour, class marker)
8) Global accuracy chart (x=session, y=balanced accuracy, lines=subjects)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch

from asrpy.asr import asr_calibrate, asr_process

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support, confusion_matrix,
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
SUBBANDS = [
    ("alpha_mu",  8,  12),
    ("low_beta",  13, 20),
    ("high_beta", 20, 30),
]


# -----------------------------
# Data IO + preprocessing — identical to CSP script
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
                tr = tr[:, :19]
                y = np.zeros_like(tr, dtype=np.float64)
                for ch in range(tr.shape[1]):
                    y[:, ch] = filtfilt(b, a, tr[:, ch])
                out[sid][lab].append(y)
    return out


def run_asr_subjectwise(filtered_data_by_subject, sfreq=256.0, cutoff=20.0):
    out = {}
    for sid, d in filtered_data_by_subject.items():
        L_trials, R_trials = d["L"], d["R"]
        all_trials = L_trials + R_trials
        if not all_trials:
            continue
        lengths = [tr.shape[0] for tr in all_trials]
        X = np.concatenate(all_trials, axis=0).T
        M, T = asr_calibrate(X, sfreq=sfreq, cutoff=cutoff)
        Xc = asr_process(X, sfreq=sfreq, M=M, T=T).T
        cleaned, idx = [], 0
        for L in lengths:
            cleaned.append(Xc[idx:idx + L, :])
            idx += L
        nL = len(L_trials)
        out[sid] = {"L": cleaned[:nL], "R": cleaned[nL:]}
        print(f"ASR done for {sid}: {len(out[sid]['L'])} L, {len(out[sid]['R'])} R")
    return out


def build_trials_for_subject(d):
    """Returns flat lists: trials (each [n_times, n_ch]), y (0=L, 1=R)."""
    if not d["L"] or not d["R"]:
        return None, None
    trials, y = [], []
    for tr in d["L"]:
        trials.append(tr[:, :19].astype(np.float64)); y.append(0)
    for tr in d["R"]:
        trials.append(tr[:, :19].astype(np.float64)); y.append(1)
    return trials, np.asarray(y, dtype=int)


# -----------------------------
# Alternating reorder — exact copy from CSP script
# -----------------------------
def make_alternating_order(labels, subj_id=None):
    labels = np.asarray(labels)
    idx0 = np.sort(np.where(labels == 0)[0])
    idx1 = np.sort(np.where(labels == 1)[0])
    if len(idx0) == 0 or len(idx1) == 0:
        if subj_id is not None:
            print(f"[WARN] {subj_id}: cannot alternate order — one class empty.")
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


# -----------------------------
# 20/80 temporal split — exact copy from CSP script
# -----------------------------
def temporal_split(X, y, train_ratio=0.2):
    split_idx = int(np.floor(train_ratio * X.shape[0]))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


# -----------------------------
# PSD feature extraction
# -----------------------------
def bandpass_single(signal, low, high, sfreq):
    nyq = 0.5 * sfreq
    b, a = butter(N=4, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def trial_to_psd_feature(trial, sfreq=FS):
    """trial: [n_times, n_ch] — variable length ok."""
    n_times, n_ch = trial.shape
    feats = []
    for _, lo, hi in SUBBANDS:
        for ch in range(n_ch):
            sig = bandpass_single(trial[:, ch], lo, hi, sfreq)
            freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, n_times))
            mask = (freqs >= lo) & (freqs <= hi)
            band_power = np.mean(psd[mask]) if mask.any() else 1e-10
            feats.append(np.log(band_power + 1e-10))
    return np.array(feats)


def extract_psd_features(trials, sfreq=FS):
    """trials: list of [n_times, n_ch] arrays."""
    return np.stack([trial_to_psd_feature(tr, sfreq) for tr in trials], axis=0)


# -----------------------------
# Classifiers + metrics — identical to CSP script
# -----------------------------
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
        sensitivity = specificity = 0.0
    return {
        "accuracy": acc, "balanced_accuracy": bal_acc, "kappa": kappa,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "f1": f1,
    }


def evaluate_models_temporal(X, y, subj_id, feature_type="PSD"):
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

    results = {}
    for name, (base_est, param_grid) in get_models_and_param_grids().items():
        print(f"  -> Optimising {name} for {feature_type} – {subj_id}")
        grid = GridSearchCV(base_est, param_grid, cv=3, scoring="accuracy", n_jobs=-1, error_score="raise")
        grid.fit(X_train_clf, y_train)
        y_pred = grid.best_estimator_.predict(X_test_clf)
        metrics = compute_binary_metrics(y_test, y_pred)
        results[name] = metrics
        print(
            f"     Test acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}, "
            f"kappa={metrics['kappa']:.3f}, sens={metrics['sensitivity']:.3f}, spec={metrics['specificity']:.3f}"
        )
    return results


# -----------------------------
# Session-wise accuracy for global chart
# Train on first 20%, evaluate each session block separately using same LR
# Session boundaries inferred from the reordered trial sequence
# -----------------------------
def compute_session_accuracy(X_psd, y, n_sessions=4):
    """
    Mirrors CSP Figure 2C logic: train on first 20% (same split as classifier),
    evaluate balanced accuracy on each equal-sized quarter of the remaining 80%.
    """
    n_trials = X_psd.shape[0]
    split_idx = int(np.floor(0.2 * n_trials))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_psd[:split_idx])
    y_train = y[:split_idx]

    if len(np.unique(y_train)) < 2:
        return [np.nan] * n_sessions

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_train)

    # Session 1 = train block; sessions 2-4 are equal thirds of the test block
    X_test_all = X_psd[split_idx:]
    y_test_all = y[split_idx:]
    n_test = len(y_test_all)
    third = n_test // 3

    # Session accuracy: [train block, test third 1, test third 2, test third 3]
    accs = [balanced_accuracy_score(y_train, clf.predict(X_train))]
    for i in range(3):
        start = i * third
        end = n_test if i == 2 else (i + 1) * third
        X_s = scaler.transform(X_test_all[start:end])
        y_s = y_test_all[start:end]
        if len(np.unique(y_s)) < 2:
            accs.append(np.nan)
        else:
            accs.append(balanced_accuracy_score(y_s, clf.predict(X_s)))
    return accs


# -----------------------------
# Visualisation
# -----------------------------
def plot_feature_space(X_psd, y, subj_id):
    """
    One figure per subject. Projects to 2D via PCA.
    Session inferred by quartile of trial index. Session=colour, Class=marker.
    """
    from matplotlib.lines import Line2D

    session_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    class_markers = {0: "o", 1: "^"}

    n_trials = X_psd.shape[0]
    quarter = n_trials // 4
    # Assign session by trial index quartile
    sess_ids = np.minimum(np.arange(n_trials) // quarter, 3)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_psd)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_std)

    fig, ax = plt.subplots(figsize=(6, 5))
    for s_idx in range(4):
        for cls in [0, 1]:
            mask = (sess_ids == s_idx) & (y == cls)
            if not mask.any():
                continue
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=session_colours[s_idx],
                marker=class_markers[cls],
                alpha=0.6, s=25, linewidths=0,
            )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=session_colours[i],
               markersize=9, label=f"Session {i+1}")
        for i in range(4)
    ] + [
        Line2D([0], [0], marker="o", color="grey", markersize=9, label="Left MI"),
        Line2D([0], [0], marker="^", color="grey", markersize=9, label="Right MI"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_title(f"PSD Sub-band Features (PC1 vs PC2) — {subj_id}", fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    plt.tight_layout()

    fname = f"PSD_feature_space_{subj_id}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_global_accuracy(subject_session_acc, fname="PSD_accuracy_chart.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = cm.tab10(np.linspace(0, 1, len(subject_session_acc)))

    for (sid, accs), col in zip(sorted(subject_session_acc.items()), colours):
        ax.plot(np.arange(1, 5), accs, marker="o", label=sid, color=col, linewidth=1.5)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6, label="Chance")
    ax.set_xlabel("Session", fontsize=11)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title("PSD: Session-wise Balanced Accuracy (Train=S1 block, Test=S2–S4 blocks)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# -----------------------------
# Main
# -----------------------------
data_by_subject = load_subjects_ogfs_mat(DATA_DIR, SUBJECT_IDS)
data_by_subject = bandpass_filter_trials(data_by_subject, BPF_LO, BPF_HI, FS)
if USE_ASR:
    data_by_subject = run_asr_subjectwise(data_by_subject, sfreq=FS, cutoff=ASR_CUTOFF)

all_results_psd = {}
subject_session_acc = {}

for sid, d in data_by_subject.items():
    if not d["L"] or not d["R"]:
        print(f"Skipping {sid}: missing L or R trials")
        continue

    trials, y = build_trials_for_subject(d)
    if trials is None:
        print(f"Skipping {sid}: not enough usable trials")
        continue
        continue

    print(f"\n########## Subject {sid} ##########")
    print(f"  Trials: {len(trials)} | y: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # Alternating reorder — exact same step as CSP script
    order = make_alternating_order(y, subj_id=sid)
    trials = [trials[i] for i in order]
    y = y[order]
    print("  [Reorder] Applied alternating 0/1 order before temporal split.")

    # Extract PSD features (all trials, no leakage — Welch is per-trial, no fitting needed)
    X_psd = extract_psd_features(trials, sfreq=FS)
    print(f"  PSD feature matrix: {X_psd.shape}")

    # Classification: 20/80 temporal split, standardise+PCA on train only
    res = evaluate_models_temporal(X_psd, y, sid, feature_type="PSD_subband")
    all_results_psd[sid] = res

    # Feature space plot
    plot_feature_space(X_psd, y, sid)

    # Session-wise accuracy for global chart
    subject_session_acc[sid] = compute_session_accuracy(X_psd, y)

# Global accuracy chart
plot_global_accuracy(subject_session_acc)

# Save CSV
rows = []
for subj_id, model_dict in all_results_psd.items():
    for model_name, metrics in model_dict.items():
        row = {"Subject": subj_id, "Model": model_name, "FeatureType": "PSD_subband"}
        row.update(metrics)
        rows.append(row)
pd.DataFrame(rows).to_csv("PSD_RESULTS_FULL.csv", index=False)
print("\nSaved CSV: PSD_RESULTS_FULL.csv")