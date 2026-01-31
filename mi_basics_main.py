# -*- coding: utf-8 -*-
"""
mi_basics_main.py

Connectivity + stability + discriminability analysis for Left/Right MI cursor control.

Implements requested changes:
1) EXTRA_PLOT_METRIC selectable at top (instead of hardcoded Coh)
2) Reduced redundancies by splitting into functions + main, removing duplicated plot fns
3) Topomap rotation fixed via TOPO_ROTATION_DEG (default +90 to put FP1 "up")
4) Publication-level plots: MNE topomaps, high DPI PNG + vector PDF
5) Voting + final bar plot uses ONLY edges + node strength, combined into ONE grouped bar plot
"""

import os
import numpy as np
import scipy.io as sio
import pandas as pd
from tqdm import tqdm
import pickle

import torch
from torch_geometric.seed import seed_everything

from joblib import Parallel, delayed

import mne
import asrpy

from mi_basics_functions import (
    # dirs / plotting
    ensure_dir, rotate_xy, savefig,
    plot_topomap_mne, plot_scalp_graph,
    plot_metric_ns_trajectories, plot_metric_edge_trajectories,
    plot_metric_recurrence, plot_metric_cross_session_similarity,
    plot_stable_highkl_edges_nilearn,


    # filters / preprocessing
    design_bandpass, apply_filtfilt,
    preprocess_trial_after_asr, preprocess_trial_cfc_after_asr,

    # metrics
    plv_pli_wpli, coherence_msc, compute_cmi, compute_cfc,
    compute_node_strength, weighted_clustering_coef,

    # surrogates
    fourier_surrogate,

    # utils
    symmetric_kl_from_samples, cohen_d, flatten_upper_tri
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ttest_1samp


# ============================================================
# BASIC CONFIG
# ============================================================

seed_everything(12345)

data_dir = "C:/Users/uceerjp/Desktop/PhD/Multi-session Data/OG_Full_Data"
curr_dir = "C:/Users/uceerjp/Desktop/PhD/Year 2/DeepLearning-on-ALS-MI-Data/Graphs/Graph Basics"
output_dir = ensure_dir(os.path.join(curr_dir, "connectivity_stability_discriminability"))

subject_numbers = [1, 2, 5, 9, 21, 31, 34, 39]
N_JOBS = 8

FS = 256.0

MAIN_BAND = (8.0, 30.0)
LOW_BAND  = (8.0, 12.0)
HIGH_BAND = (13.0, 30.0)

USE_PRE_BANDPASS = True
USE_ASR          = True
USE_ICA          = False

APPLY_CAR    = False
APPLY_ZSCORE = False

ASR_CUTOFF = 20.0
N_SURR = 0

METRICS_TO_COMPUTE = [
    "PLV", "ImagPLV", "PLI", "wPLI",
    "Coh", "imCoh", "MSC", "CMI", "CFC",
]

# (1) SELECTABLE METRIC FOR EXTRA TRIAL-WISE PLOTS + SAVED FEATURE DICT
EXTRA_PLOT_METRIC = "Coh"   # <- change this to any of METRICS_TO_COMPUTE
SAVE_FEATURE_METRIC = EXTRA_PLOT_METRIC  # the metric saved into the per-trial pkl

STABILITY_PERCENTILE = 30.0
KL_N_BINS       = 20.0
KL_PERCENTILE   = 70.0
KL_EPS          = 1e-12

# (3) Topomap orientation
TOPO_ROTATION_DEG = 90.0  # puts FP1 "up" given your current coordinates

# (4) Publication output
PLOT_DPI = 600


# ============================================================
# ELECTRODE SETUP
# ============================================================

electrode_labels = [
    "FP1", "FP2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
numElectrodes = len(electrode_labels)

xyz_coords = np.array([
    [0.950,  0.309,  -0.0349],   # FP1
    [0.950, -0.309,  -0.0349],   # FP2
    [0.587,  0.809,  -0.0349],   # F7
    [0.673,  0.545,   0.500],    # F3
    [0.719,  0.000,   0.695],    # Fz
    [0.673, -0.545,   0.500],    # F4
    [0.587, -0.809,  -0.0349],   # F8
    [6.120e-17,  0.999,  -0.0349],   # T7
    [4.400e-17,  0.719,   0.695],    # C3
    [0.000,      0.000,  1.000],     # Cz
    [4.400e-17, -0.719,   0.695],    # C4
    [6.120e-17, -0.999,  -0.0349],   # T8
    [-0.587,  0.809,  -0.0349],      # P7
    [-0.673,  0.545,   0.500],       # P3
    [-0.719, -8.810e-17, 0.695],     # Pz
    [-0.673, -0.545,   0.500],       # P4
    [-0.587, -0.809,  -0.0349],      # P8
    [-0.950,  0.309,  -0.0349],      # O1
    [-0.950, -0.309,  -0.0349],      # O2
])

# 2D positions for topomaps/scalp graphs
xy_coords = xyz_coords[:, :2].copy()
xy_coords = rotate_xy(xy_coords, deg=TOPO_ROTATION_DEG)

# distance matrix based on xyz
dist_matrix = np.zeros((numElectrodes, numElectrodes))
for i in range(numElectrodes):
    for j in range(numElectrodes):
        dist_matrix[i, j] = np.linalg.norm(xyz_coords[i] - xyz_coords[j])


# ============================================================
# FILTER DESIGN
# ============================================================

bp_b, bp_a     = design_bandpass(MAIN_BAND, FS, order=4)
low_b, low_a   = design_bandpass(LOW_BAND, FS, order=4)
high_b, high_a = design_bandpass(HIGH_BAND, FS, order=4)

def apply_main_bandpass(x):
    return apply_filtfilt(x, bp_b, bp_a)


# ============================================================
# PER-SUBJECT WORKER
# ============================================================

def process_subject(subject_number):
    print(f"\nProcessing Subject S{subject_number}")

    # Collect per-trial features for SAVE_FEATURE_METRIC (parameterised)
    extra_edges_list = []
    extra_ns_list = []
    label_list = []

    metric_trial_mats_by_cls = {"L": None, "R": None}
    ns_trial_mats_by_cls = {"L": None, "R": None}
    clust_trial_mats_by_cls = {"L": None, "R": None}
    session_sizes_by_class = {"L": None, "R": None}

    mat_fname = os.path.join(data_dir, f"S{subject_number}.mat")
    mat_contents = sio.loadmat(mat_fname)
    subject_raw = mat_contents[f"Subject{subject_number}"]
    S1 = subject_raw[:, :-1]
    subject_data = S1

    stability_subj = {"L": {}, "R": {}}
    mean_metric_subj = {metric: {"L": None, "R": None} for metric in METRICS_TO_COMPUTE}
    mean_surrogate_subj = {metric: {"L": None, "R": None} for metric in METRICS_TO_COMPUTE}

    ns_summary_subj = {
        metric: {"L": {"mean": None, "cv": None}, "R": {"mean": None, "cv": None}}
        for metric in METRICS_TO_COMPUTE
    }
    clust_summary_subj = {
        metric: {"L": {"mean": None, "cv": None}, "R": {"mean": None, "cv": None}}
        for metric in METRICS_TO_COMPUTE
    }

    # --------------------------------------------------------
    # Subject-level preprocessing: bandpass -> ASR -> (optional) ICA
    # --------------------------------------------------------
    all_trials = []
    trial_meta = []  # (cls, t, length)
    cleaned_trials_by_cls = {}

    for cls in ["L", "R"]:
        trials_mat = subject_data[cls]
        num_trials_cls = trials_mat.shape[1]
        cleaned_trials_by_cls[cls] = [None] * num_trials_cls

        for t in range(num_trials_cls):
            x_raw = trials_mat[0, t][:, :numElectrodes]
            if USE_PRE_BANDPASS:
                x_raw = apply_main_bandpass(x_raw)
            all_trials.append(x_raw)
            trial_meta.append((cls, t, x_raw.shape[0]))

    concat_data = np.concatenate(all_trials, axis=0)

    if USE_ASR or USE_ICA:
        info = mne.create_info(
            ch_names=electrode_labels,
            sfreq=FS,
            ch_types=["eeg"] * numElectrodes,
        )
        raw_mne = mne.io.RawArray(concat_data.T, info, verbose=False)

        if USE_ASR:
            asr = asrpy.ASR(sfreq=FS, cutoff=ASR_CUTOFF)
            asr.fit(raw_mne)
            raw_mne = asr.transform(raw_mne)

        if USE_ICA:
            from mne.preprocessing import ICA, find_bads_eog
            ica = ICA(
                n_components=None,
                method="fastica",
                random_state=97,
                max_iter="auto",
            )
            ica.fit(raw_mne)

            eog_chs = [ch for ch in ["FP1", "FP2"] if ch in raw_mne.ch_names]
            if len(eog_chs) > 0:
                eog_inds, _ = find_bads_eog(raw_mne, ch_name=eog_chs)
                ica.exclude = list(eog_inds)

            raw_mne = ica.apply(raw_mne.copy())

        concat_clean = raw_mne.get_data().T
    else:
        concat_clean = concat_data

    start = 0
    for cls, t_idx, L in trial_meta:
        end = start + L
        cleaned_trials_by_cls[cls][t_idx] = concat_clean[start:end, :]
        start = end

    # --------------------------------------------------------
    # Class-wise compute trial matrices
    # --------------------------------------------------------
    n_sessions = 4

    for cls in ["L", "R"]:
        cleaned_trials = cleaned_trials_by_cls[cls]
        num_trials = len(cleaned_trials)

        if num_trials < n_sessions:
            raise ValueError(f"Subject {subject_number}, class {cls} has <{n_sessions} trials.")

        base = num_trials // n_sessions
        remainder = num_trials % n_sessions
        session_sizes = [base + 1 if s < remainder else base for s in range(n_sessions)]
        session_sizes_by_class[cls] = session_sizes

        metric_trial_mats = {metric: np.zeros((numElectrodes, numElectrodes, num_trials)) for metric in METRICS_TO_COMPUTE}
        surrogate_trial_mats = {metric: np.zeros((numElectrodes, numElectrodes, num_trials)) for metric in METRICS_TO_COMPUTE}
        ns_trial_mats = {metric: np.zeros((numElectrodes, num_trials)) for metric in METRICS_TO_COMPUTE}
        clust_trial_mats = {metric: np.zeros((numElectrodes, num_trials)) for metric in METRICS_TO_COMPUTE}

        for t in tqdm(range(num_trials), desc=f"S{subject_number} {cls} trials", leave=False):
            x_clean = cleaned_trials[t]

            x_main = preprocess_trial_after_asr(
                x_clean, numElectrodes, apply_car=APPLY_CAR, apply_zscore=APPLY_ZSCORE
            )
            x_cfc = preprocess_trial_cfc_after_asr(
                x_clean, numElectrodes, apply_car=APPLY_CAR, apply_zscore=APPLY_ZSCORE
            )

            plv_mat, imag_mat, pli_mat, wpli_mat = plv_pli_wpli(x_main, numElectrodes)
            coh_mat, imcoh_mat, msc_mat = coherence_msc(x_main, FS, MAIN_BAND, numElectrodes)
            cmi_mat = compute_cmi(x_main, numElectrodes)
            cfc_mat = compute_cfc(x_cfc, numElectrodes, low_b, low_a, high_b, high_a)

            # fill requested metrics
            if "PLV" in METRICS_TO_COMPUTE:
                metric_trial_mats["PLV"][:, :, t] = plv_mat
                ns_trial_mats["PLV"][:, t] = compute_node_strength(plv_mat)
                clust_trial_mats["PLV"][:, t] = weighted_clustering_coef(plv_mat)
            if "ImagPLV" in METRICS_TO_COMPUTE:
                metric_trial_mats["ImagPLV"][:, :, t] = imag_mat
                ns_trial_mats["ImagPLV"][:, t] = compute_node_strength(imag_mat)
                clust_trial_mats["ImagPLV"][:, t] = weighted_clustering_coef(imag_mat)
            if "PLI" in METRICS_TO_COMPUTE:
                metric_trial_mats["PLI"][:, :, t] = pli_mat
                ns_trial_mats["PLI"][:, t] = compute_node_strength(pli_mat)
                clust_trial_mats["PLI"][:, t] = weighted_clustering_coef(pli_mat)
            if "wPLI" in METRICS_TO_COMPUTE:
                metric_trial_mats["wPLI"][:, :, t] = wpli_mat
                ns_trial_mats["wPLI"][:, t] = compute_node_strength(wpli_mat)
                clust_trial_mats["wPLI"][:, t] = weighted_clustering_coef(wpli_mat)
            if "Coh" in METRICS_TO_COMPUTE:
                metric_trial_mats["Coh"][:, :, t] = coh_mat
                ns_trial_mats["Coh"][:, t] = compute_node_strength(coh_mat)
                clust_trial_mats["Coh"][:, t] = weighted_clustering_coef(coh_mat)
            if "imCoh" in METRICS_TO_COMPUTE:
                metric_trial_mats["imCoh"][:, :, t] = imcoh_mat
                ns_trial_mats["imCoh"][:, t] = compute_node_strength(imcoh_mat)
                clust_trial_mats["imCoh"][:, t] = weighted_clustering_coef(imcoh_mat)
            if "MSC" in METRICS_TO_COMPUTE:
                metric_trial_mats["MSC"][:, :, t] = msc_mat
                ns_trial_mats["MSC"][:, t] = compute_node_strength(msc_mat)
                clust_trial_mats["MSC"][:, t] = weighted_clustering_coef(msc_mat)
            if "CMI" in METRICS_TO_COMPUTE:
                metric_trial_mats["CMI"][:, :, t] = cmi_mat
                ns_trial_mats["CMI"][:, t] = compute_node_strength(cmi_mat)
                clust_trial_mats["CMI"][:, t] = weighted_clustering_coef(cmi_mat)
            if "CFC" in METRICS_TO_COMPUTE:
                metric_trial_mats["CFC"][:, :, t] = cfc_mat
                ns_trial_mats["CFC"][:, t] = compute_node_strength(cfc_mat)
                clust_trial_mats["CFC"][:, t] = weighted_clustering_coef(cfc_mat)

            # surrogates (optional)
            if N_SURR > 0:
                acc = {metric: None for metric in METRICS_TO_COMPUTE}
                for _ in range(N_SURR):
                    sur_main = fourier_surrogate(x_main, numElectrodes)
                    sur_cfc_in = fourier_surrogate(x_cfc, numElectrodes)

                    s_plv, s_imag, s_pli, s_wpli = plv_pli_wpli(sur_main, numElectrodes)
                    s_coh, s_imcoh, s_msc = coherence_msc(sur_main, FS, MAIN_BAND, numElectrodes)
                    s_cmi = compute_cmi(sur_main, numElectrodes)
                    s_cfc = compute_cfc(sur_cfc_in, numElectrodes, low_b, low_a, high_b, high_a)

                    def add(metric, mat):
                        acc[metric] = mat if acc[metric] is None else acc[metric] + mat

                    if "PLV" in METRICS_TO_COMPUTE: add("PLV", s_plv)
                    if "ImagPLV" in METRICS_TO_COMPUTE: add("ImagPLV", s_imag)
                    if "PLI" in METRICS_TO_COMPUTE: add("PLI", s_pli)
                    if "wPLI" in METRICS_TO_COMPUTE: add("wPLI", s_wpli)
                    if "Coh" in METRICS_TO_COMPUTE: add("Coh", s_coh)
                    if "imCoh" in METRICS_TO_COMPUTE: add("imCoh", s_imcoh)
                    if "MSC" in METRICS_TO_COMPUTE: add("MSC", s_msc)
                    if "CMI" in METRICS_TO_COMPUTE: add("CMI", s_cmi)
                    if "CFC" in METRICS_TO_COMPUTE: add("CFC", s_cfc)

                for metric in METRICS_TO_COMPUTE:
                    surrogate_trial_mats[metric][:, :, t] = acc[metric] / float(N_SURR)

        # ---- Collect trial-wise features for SAVE_FEATURE_METRIC (parametric, no hardcoded Coh) ----
        if SAVE_FEATURE_METRIC in METRICS_TO_COMPUTE:
            edges_cls = metric_trial_mats[SAVE_FEATURE_METRIC].transpose(2, 0, 1)  # [T, E, E]
            ns_cls = ns_trial_mats[SAVE_FEATURE_METRIC].T  # [T, E]
            label_val = 0 if cls == "L" else 1
            labels_cls = np.full((num_trials,), label_val, dtype=int)

            extra_edges_list.append(edges_cls)
            extra_ns_list.append(ns_cls)
            label_list.append(labels_cls)

        # ---- Summaries per metric ----
        for metric in METRICS_TO_COMPUTE:
            mats_trials = metric_trial_mats[metric]
            sur_trials  = surrogate_trial_mats[metric]
            ns_trials   = ns_trial_mats[metric]
            cl_trials   = clust_trial_mats[metric]

            real_mean = np.mean(mats_trials, axis=2)
            sur_mean  = np.mean(sur_trials, axis=2) if N_SURR > 0 else np.zeros_like(real_mean)
            mean_metric_subj[metric][cls]    = real_mean
            mean_surrogate_subj[metric][cls] = sur_mean

            mean_edge_trials = np.mean(mats_trials, axis=2)
            std_edge_trials  = np.std(mats_trials, axis=2)
            cv_edge_mat = np.zeros_like(mean_edge_trials)
            nonzero_edge = np.abs(mean_edge_trials) > 1e-6
            cv_edge_mat[nonzero_edge] = std_edge_trials[nonzero_edge] / (np.abs(mean_edge_trials[nonzero_edge]) + 1e-10)

            mean_ns_trials = np.mean(ns_trials, axis=1)
            std_ns_trials  = np.std(ns_trials, axis=1)
            cv_ns_vec = np.zeros_like(mean_ns_trials)
            nonzero_ns = np.abs(mean_ns_trials) > 1e-6
            cv_ns_vec[nonzero_ns] = std_ns_trials[nonzero_ns] / (np.abs(mean_ns_trials[nonzero_ns]) + 1e-10)

            mean_cl_trials = np.mean(cl_trials, axis=1)
            std_cl_trials  = np.std(cl_trials, axis=1)
            cv_cl_vec = np.zeros_like(mean_cl_trials)
            nonzero_cl = np.abs(mean_cl_trials) > 1e-6
            cv_cl_vec[nonzero_cl] = std_cl_trials[nonzero_cl] / (np.abs(mean_cl_trials[nonzero_cl]) + 1e-10)

            stability_subj[cls][metric] = {"std": std_edge_trials, "cv": cv_edge_mat}
            ns_summary_subj[metric][cls]["mean"] = mean_ns_trials
            ns_summary_subj[metric][cls]["cv"]   = cv_ns_vec
            clust_summary_subj[metric][cls]["mean"] = mean_cl_trials
            clust_summary_subj[metric][cls]["cv"]   = cv_cl_vec

        metric_trial_mats_by_cls[cls] = metric_trial_mats
        ns_trial_mats_by_cls[cls] = ns_trial_mats
        clust_trial_mats_by_cls[cls] = clust_trial_mats

    # --------------------------------------------------------
    # KL per metric per feature (L vs R)
    # --------------------------------------------------------
    d_edge_metrics, d_ns_metrics, d_cl_metrics = {}, {}, {}
    kl_edge_metrics, kl_ns_metrics, kl_cl_metrics = {}, {}, {}

    for metric in METRICS_TO_COMPUTE:
        mats_L = metric_trial_mats_by_cls["L"][metric]
        mats_R = metric_trial_mats_by_cls["R"][metric]
        ns_L   = ns_trial_mats_by_cls["L"][metric]
        ns_R   = ns_trial_mats_by_cls["R"][metric]
        cl_L   = clust_trial_mats_by_cls["L"][metric]
        cl_R   = clust_trial_mats_by_cls["R"][metric]

        n_elec = mats_L.shape[0]
        d_edge_mat  = np.zeros((n_elec, n_elec))
        kl_edge_mat = np.zeros((n_elec, n_elec))

        for i in range(n_elec):
            for j in range(i + 1, n_elec):
                vals_L = mats_L[i, j, :]
                vals_R = mats_R[i, j, :]

                d_val = cohen_d(vals_L, vals_R)
                d_edge_mat[i, j] = d_val
                d_edge_mat[j, i] = d_val

                kl_val = symmetric_kl_from_samples(vals_L, vals_R, n_bins=KL_N_BINS, eps=KL_EPS)
                kl_edge_mat[i, j] = kl_val
                kl_edge_mat[j, i] = kl_val

        d_edge_metrics[metric]  = d_edge_mat
        kl_edge_metrics[metric] = kl_edge_mat

        d_ns_vec = np.zeros(n_elec)
        d_cl_vec = np.zeros(n_elec)
        kl_ns_vec = np.zeros(n_elec)
        kl_cl_vec = np.zeros(n_elec)

        for k in range(n_elec):
            vals_L_ns = ns_L[k, :]
            vals_R_ns = ns_R[k, :]
            d_ns_vec[k] = cohen_d(vals_L_ns, vals_R_ns)
            kl_ns_vec[k] = symmetric_kl_from_samples(vals_L_ns, vals_R_ns, n_bins=KL_N_BINS, eps=KL_EPS)

            vals_L_cl = cl_L[k, :]
            vals_R_cl = cl_R[k, :]
            d_cl_vec[k] = cohen_d(vals_L_cl, vals_R_cl)
            kl_cl_vec[k] = symmetric_kl_from_samples(vals_L_cl, vals_R_cl, n_bins=KL_N_BINS, eps=KL_EPS)

        d_ns_metrics[metric] = d_ns_vec
        d_cl_metrics[metric] = d_cl_vec
        kl_ns_metrics[metric] = kl_ns_vec
        kl_cl_metrics[metric] = kl_cl_vec

    # --------------------------------------------------------
    # Finalise per-trial arrays for SAVE_FEATURE_METRIC
    # --------------------------------------------------------
    if len(extra_edges_list) > 0:
        extra_edges_all  = np.concatenate(extra_edges_list, axis=0)  # [N, E, E]
        extra_ns_all     = np.concatenate(extra_ns_list, axis=0)     # [N, E]
        extra_labels_all = np.concatenate(label_list, axis=0)        # [N]
    else:
        extra_edges_all = None
        extra_ns_all = None
        extra_labels_all = None

    return (
        subject_number,
        stability_subj,
        mean_metric_subj,
        mean_surrogate_subj,
        ns_summary_subj,
        clust_summary_subj,
        extra_edges_all,
        extra_ns_all,
        extra_labels_all,
        d_edge_metrics,
        d_ns_metrics,
        d_cl_metrics,
        kl_edge_metrics,
        kl_ns_metrics,
        kl_cl_metrics,
        session_sizes_by_class
    )


# ============================================================
# RUN ALL SUBJECTS
# ============================================================

results = Parallel(n_jobs=N_JOBS)(delayed(process_subject)(subj) for subj in subject_numbers)
print("All subjects processed.")

results_by_subj = {}
feature_data = {}

d_edge_per_subject = {metric: {} for metric in METRICS_TO_COMPUTE}
d_ns_per_subject   = {metric: {} for metric in METRICS_TO_COMPUTE}
d_cl_per_subject   = {metric: {} for metric in METRICS_TO_COMPUTE}
kl_edge_per_subject = {metric: {} for metric in METRICS_TO_COMPUTE}
kl_ns_per_subject   = {metric: {} for metric in METRICS_TO_COMPUTE}
kl_cl_per_subject   = {metric: {} for metric in METRICS_TO_COMPUTE}

for (
    subj,
    stability_subj,
    mean_metric_subj,
    mean_surrogate_subj,
    ns_summary_subj,
    clust_summary_subj,
    extra_edges_all,
    extra_ns_all,
    extra_labels_all,
    d_edge_metrics,
    d_ns_metrics,
    d_cl_metrics,
    kl_edge_metrics,
    kl_ns_metrics,
    kl_cl_metrics,
    session_sizes_by_class
) in results:

    results_by_subj[subj] = {
        "stability": stability_subj,
        "mean_metric": mean_metric_subj,
        "mean_surrogate": mean_surrogate_subj,
        "ns_summary": ns_summary_subj,
        "clust_summary": clust_summary_subj,
        "session_sizes": session_sizes_by_class,
        "extra_trials": {
            "metric": SAVE_FEATURE_METRIC,
            "Edges": extra_edges_all,
            "NS": extra_ns_all,
            "Labels": extra_labels_all,
        },
    }

    for metric in METRICS_TO_COMPUTE:
        d_edge_per_subject[metric][subj]  = d_edge_metrics[metric]
        d_ns_per_subject[metric][subj]    = d_ns_metrics[metric]
        d_cl_per_subject[metric][subj]    = d_cl_metrics[metric]

        kl_edge_per_subject[metric][subj] = kl_edge_metrics[metric]
        kl_ns_per_subject[metric][subj]   = kl_ns_metrics[metric]
        kl_cl_per_subject[metric][subj]   = kl_cl_metrics[metric]

    subj_key = f"S{subj:02d}"
    feature_data[subj_key] = {
        SAVE_FEATURE_METRIC: {
            "Edges": extra_edges_all.copy() if extra_edges_all is not None else None,
            "NS": extra_ns_all.copy() if extra_ns_all is not None else None,
            "Labels": extra_labels_all.copy() if extra_labels_all is not None else None,
            "edge_mask": None,
            "node_mask": None,
        }
    }


# ============================================================
# GLOBAL ANALYSES: DISTANCE VS METRIC + PER ELECTRODE CORR
# ============================================================

distance_spearman_summary = []
electrode_distance_corr_rows = []

def distance_vs_metric_analysis(metric_name):
    for cls in ["L", "R"]:
        all_d, all_m = [], []
        for subj in subject_numbers:
            mat = results_by_subj[subj]["mean_metric"][metric_name][cls]
            for i in range(numElectrodes):
                for j in range(i + 1, numElectrodes):
                    all_d.append(dist_matrix[i, j])
                    all_m.append(np.abs(mat[i, j]))
        all_d = np.array(all_d)
        all_m = np.array(all_m)

        if (np.std(all_d) < 1e-8) or (np.std(all_m) < 1e-8):
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(all_d, all_m)

        distance_spearman_summary.append({
            "metric": metric_name, "class": cls,
            "spearman_rho": rho, "spearman_p": pval,
        })

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(all_d, all_m, alpha=0.25, s=10)
        ax.set_xlabel("Inter-electrode distance (3D)")
        ax.set_ylabel(f"|{metric_name}|")
        ax.set_title(f"{metric_name} vs distance, class {cls}", fontsize=12)
        savefig(fig, os.path.join(output_dir, f"{metric_name}_vs_distance_class_{cls}"), dpi=PLOT_DPI, also_pdf=True)


def per_electrode_distance_corr(metric_name):
    for cls in ["L", "R"]:
        mean_rhos_per_electrode = []
        for i in range(numElectrodes):
            subj_rhos = []
            for subj in subject_numbers:
                mat = results_by_subj[subj]["mean_metric"][metric_name][cls]
                d_list, m_list = [], []
                for j in range(numElectrodes):
                    if j == i:
                        continue
                    d_list.append(dist_matrix[i, j])
                    m_list.append(np.abs(mat[i, j]))
                d_arr = np.array(d_list)
                m_arr = np.array(m_list)
                if (np.std(d_arr) < 1e-8) or (np.std(m_arr) < 1e-8):
                    continue
                rho, _ = spearmanr(d_arr, m_arr)
                if np.isfinite(rho):
                    subj_rhos.append(rho)

            if len(subj_rhos) == 0:
                mean_rho = np.nan
                std_rho = np.nan
                tstat = np.nan
                pval = np.nan
                n_subj = 0
            else:
                subj_rhos = np.array(subj_rhos)
                mean_rho = np.mean(subj_rhos)
                std_rho = np.std(subj_rhos)
                if len(subj_rhos) > 1:
                    tstat, pval = ttest_1samp(subj_rhos, 0.0)
                else:
                    tstat, pval = np.nan, np.nan
                n_subj = len(subj_rhos)

            mean_rhos_per_electrode.append(mean_rho)
            electrode_distance_corr_rows.append({
                "metric": metric_name, "class": cls,
                "electrode": electrode_labels[i],
                "mean_rho_distance_vs_metric": mean_rho,
                "std_rho_across_subjects": std_rho,
                "t_vs_0": tstat, "p_vs_0": pval,
                "n_subjects": n_subj,
            })

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(np.arange(numElectrodes), mean_rhos_per_electrode)
        ax.set_xticks(np.arange(numElectrodes))
        ax.set_xticklabels(electrode_labels, rotation=90)
        ax.set_ylabel(f"Mean Spearman rho (distance vs |{metric_name}|)")
        ax.set_title(f"{metric_name}: per-electrode distance correlation, class {cls}", fontsize=12)
        savefig(fig, os.path.join(output_dir, f"{metric_name}_per_electrode_distance_corr_class_{cls}"), dpi=PLOT_DPI, also_pdf=True)


print("\n--- Distance vs metric analysis ---")
for metric_name in METRICS_TO_COMPUTE:
    distance_vs_metric_analysis(metric_name)
    per_electrode_distance_corr(metric_name)


# ============================================================
# STABILITY + DISCRIMINABILITY SUMMARIES + S∩High-KL (edges & ns only)
# ============================================================

stability_summary_rows = []
discriminability_summary_rows = []
sd_feature_count_rows = []
sd_feature_detail_rows = []

FEATURE_TYPES_FOR_VOTING = ["edges", "ns"]  # (5) only these

for metric in METRICS_TO_COMPUTE:
    for subj in subject_numbers:
        res = results_by_subj[subj]

        # ---- STABILITY ----
        for cls in ["L", "R"]:
            cv_edge_mat = res["stability"][cls][metric]["cv"]
            cv_ns_vec   = res["ns_summary"][metric][cls]["cv"]
            cv_cl_vec   = res["clust_summary"][metric][cls]["cv"]

            cv_edges_flat = flatten_upper_tri(cv_edge_mat)

            stability_summary_rows.append({
                "subject": subj, "metric": metric, "class": cls,
                "mean_cv_edges": np.nanmean(cv_edges_flat),
                "std_cv_edges": np.nanstd(cv_edges_flat),
                "mean_cv_ns": np.nanmean(cv_ns_vec),
                "std_cv_ns": np.nanstd(cv_ns_vec),
                "mean_cv_clust": np.nanmean(cv_cl_vec),
                "std_cv_clust": np.nanstd(cv_cl_vec),
            })

        # ---- DISCRIMINABILITY (KL only) ----
        kl_edge_mat = kl_edge_per_subject[metric][subj]
        kl_ns_vec   = kl_ns_per_subject[metric][subj]
        kl_cl_vec   = kl_cl_per_subject[metric][subj]
        kl_edges_flat = flatten_upper_tri(kl_edge_mat)

        discriminability_summary_rows.append({
            "subject": subj, "metric": metric,
            "mean_kl_edges": np.nanmean(kl_edges_flat),
            "std_kl_edges": np.nanstd(kl_edges_flat),
            "mean_kl_ns": np.nanmean(kl_ns_vec),
            "std_kl_ns": np.nanstd(kl_ns_vec),
            "mean_kl_clust": np.nanmean(kl_cl_vec),
            "std_kl_clust": np.nanstd(kl_cl_vec),
        })

        # ---- S ∩ High-KL counts (edges + ns only) ----
        for feature_type in FEATURE_TYPES_FOR_VOTING:
            if feature_type == "edges":
                cv_L = flatten_upper_tri(res["stability"]["L"][metric]["cv"])
                cv_R = flatten_upper_tri(res["stability"]["R"][metric]["cv"])
                cv_vec = np.nanmean(np.stack([cv_L, cv_R], axis=0), axis=0)
                kl_vec = flatten_upper_tri(kl_edge_per_subject[metric][subj])
            else:  # ns
                cv_L = res["ns_summary"][metric]["L"]["cv"]
                cv_R = res["ns_summary"][metric]["R"]["cv"]
                cv_vec = np.nanmean(np.stack([cv_L, cv_R], axis=0), axis=0)
                kl_vec = kl_ns_per_subject[metric][subj]

            valid_mask = np.isfinite(cv_vec) & np.isfinite(kl_vec)
            cv_valid = cv_vec[valid_mask]
            kl_valid = kl_vec[valid_mask]

            if cv_valid.size < 10 or kl_valid.size < 10:
                n_valid = int(np.sum(valid_mask))
                stable_count = 0
                highkl_count = 0
                sdf_count = 0
                sdf_mask = np.zeros_like(kl_valid, dtype=bool)
            else:
                cv_thresh = np.percentile(cv_valid, STABILITY_PERCENTILE)
                kl_thresh = np.percentile(kl_valid, KL_PERCENTILE)
                stable_mask = cv_valid <= cv_thresh
                highkl_mask = kl_valid >= kl_thresh
                sdf_mask = stable_mask & highkl_mask

                n_valid = int(np.sum(valid_mask))
                stable_count = int(np.sum(stable_mask))
                highkl_count = int(np.sum(highkl_mask))
                sdf_count = int(np.sum(sdf_mask))

                # details
                valid_indices = np.where(valid_mask)[0]
                if feature_type == "edges":
                    iu, ju = np.triu_indices(numElectrodes, k=1)
                    for local_idx, global_idx in enumerate(valid_indices):
                        if sdf_mask[local_idx]:
                            i = iu[global_idx]
                            j = ju[global_idx]
                            sd_feature_detail_rows.append({
                                "subject": subj, "metric": metric, "feature_type": "edges",
                                "i_idx": int(i), "j_idx": int(j),
                                "i_label": electrode_labels[i], "j_label": electrode_labels[j],
                                "cv": float(cv_vec[global_idx]), "kl": float(kl_vec[global_idx]),
                            })
                else:
                    for local_idx, global_idx in enumerate(valid_indices):
                        if sdf_mask[local_idx]:
                            i = global_idx
                            sd_feature_detail_rows.append({
                                "subject": subj, "metric": metric, "feature_type": "ns",
                                "i_idx": int(i), "j_idx": np.nan,
                                "i_label": electrode_labels[i], "j_label": "",
                                "cv": float(cv_vec[global_idx]), "kl": float(kl_vec[global_idx]),
                            })

            sd_feature_count_rows.append({
                "subject": subj, "metric": metric, "feature_type": feature_type,
                "n_features_valid": int(np.sum(valid_mask)),
                "stability_percentile": STABILITY_PERCENTILE,
                "kl_percentile": KL_PERCENTILE,
                "stable_count": stable_count,
                "highKL_count": highkl_count,
                "stable_and_highKL_count": sdf_count,
            })


# ============================================================
# APPLY STABLE ∩ High-KL MASKS to SAVED METRIC FEATURES (edges + ns only)
# ============================================================

for subj in subject_numbers:
    subj_key = f"S{subj:02d}"
    metric_name = SAVE_FEATURE_METRIC
    md = feature_data[subj_key][metric_name]

    if md["Edges"] is None or md["NS"] is None:
        continue

    edges_all = md["Edges"]  # [N, E, E]
    ns_all    = md["NS"]     # [N, E]
    res = results_by_subj[subj]

    # --- Edge mask ---
    cv_L_edges = res["stability"]["L"][metric_name]["cv"]
    cv_R_edges = res["stability"]["R"][metric_name]["cv"]
    kl_edges   = kl_edge_per_subject[metric_name][subj]

    cv_vec_edges = np.nanmean(
        np.stack([flatten_upper_tri(cv_L_edges), flatten_upper_tri(cv_R_edges)], axis=0),
        axis=0,
    )
    kl_vec_edges = flatten_upper_tri(kl_edges)
    valid_edges = np.isfinite(cv_vec_edges) & np.isfinite(kl_vec_edges)

    edge_mask = np.zeros((numElectrodes, numElectrodes), dtype=bool)
    if np.sum(valid_edges) >= 10:
        cv_valid = cv_vec_edges[valid_edges]
        kl_valid = kl_vec_edges[valid_edges]
        cv_thresh = np.percentile(cv_valid, STABILITY_PERCENTILE)
        kl_thresh = np.percentile(kl_valid, KL_PERCENTILE)
        both = (cv_valid <= cv_thresh) & (kl_valid >= kl_thresh)

        iu, ju = np.triu_indices(numElectrodes, k=1)
        valid_indices = np.where(valid_edges)[0]
        for local_idx, global_idx in enumerate(valid_indices):
            if both[local_idx]:
                i = iu[global_idx]
                j = ju[global_idx]
                edge_mask[i, j] = True
                edge_mask[j, i] = True
    np.fill_diagonal(edge_mask, False)

    masked_edges_all = edges_all * edge_mask[np.newaxis, :, :] if np.any(edge_mask) else np.zeros_like(edges_all)

    # --- Node mask (based on NS CV + KL) ---
    cv_ns_L = res["ns_summary"][metric_name]["L"]["cv"]
    cv_ns_R = res["ns_summary"][metric_name]["R"]["cv"]
    kl_ns   = kl_ns_per_subject[metric_name][subj]

    cv_ns_combined = np.nanmean(np.stack([cv_ns_L, cv_ns_R], axis=0), axis=0)
    valid_nodes = np.isfinite(cv_ns_combined) & np.isfinite(kl_ns)
    node_mask = np.zeros_like(cv_ns_combined, dtype=bool)

    if np.sum(valid_nodes) >= 3:
        cv_valid_nodes = cv_ns_combined[valid_nodes]
        kl_valid_nodes = kl_ns[valid_nodes]
        cv_thr = np.percentile(cv_valid_nodes, STABILITY_PERCENTILE)
        kl_thr = np.percentile(kl_valid_nodes, KL_PERCENTILE)
        node_mask[valid_nodes] = (cv_valid_nodes <= cv_thr) & (kl_valid_nodes >= kl_thr)

    # recompute node strength from masked edges, and keep only masked nodes
    ns_recomputed = np.zeros_like(ns_all)
    for t in range(masked_edges_all.shape[0]):
        ns_recomputed[t] = compute_node_strength(masked_edges_all[t])

    masked_ns_all = np.zeros_like(ns_all)
    if np.any(node_mask):
        masked_ns_all[:, node_mask] = ns_recomputed[:, node_mask]

    md["Edges"] = masked_edges_all
    md["NS"] = masked_ns_all
    md["edge_mask"] = edge_mask
    md["node_mask"] = node_mask


# ============================================================
# PER-SUBJECT PLOTS (publication style)
# ============================================================

for subj in subject_numbers:
    subj_dir = ensure_dir(os.path.join(output_dir, f"S{subj:02d}"))

    for metric in METRICS_TO_COMPUTE:
        res = results_by_subj[subj]
        kl_ns = kl_ns_per_subject[metric][subj]
        # (3)+(4) Nice MNE topomap, rotated coords
        plot_topomap_mne(
            values=kl_ns,
            pos_xy=xy_coords,
            ch_names=electrode_labels,
            title=f"S{subj:02d} {metric} - KL(L,R) node strength",
            out_base=os.path.join(subj_dir, f"S{subj:02d}_{metric}_ns_KL_topomap"),
            cmap="viridis",
            dpi=PLOT_DPI,
            show_names=True,
        )

        # Stable ∩ High-KL edges scalp graph
        cv_L_edge = res["stability"]["L"][metric]["cv"]
        cv_R_edge = res["stability"]["R"][metric]["cv"]
        kl_edge = kl_edge_per_subject[metric][subj]

        cv_vec = np.nanmean(
            np.stack([flatten_upper_tri(cv_L_edge), flatten_upper_tri(cv_R_edge)], axis=0),
            axis=0
        )
        kl_vec = flatten_upper_tri(kl_edge)
        valid = np.isfinite(cv_vec) & np.isfinite(kl_vec)

        selected_edges = []
        if np.sum(valid) >= 10:
            cv_valid = cv_vec[valid]
            kl_valid = kl_vec[valid]
            cv_thr = np.percentile(cv_valid, STABILITY_PERCENTILE)
            kl_thr = np.percentile(kl_valid, KL_PERCENTILE)
            both = (cv_valid <= cv_thr) & (kl_valid >= kl_thr)

            iu, ju = np.triu_indices(numElectrodes, k=1)
            valid_indices = np.where(valid)[0]
            for local_idx, global_idx in enumerate(valid_indices):
                if both[local_idx]:
                    selected_edges.append((int(iu[global_idx]), int(ju[global_idx])))


        # --- ADD THIS CALL (Nilearn connectome) ---
        out_png = os.path.join(subj_dir, f"S{subj:02d}_{metric}_lowCV_highKL_edges_nilearn.png")
        
        plot_stable_highkl_edges_nilearn(
            subject=subj,
            metric=metric,
            selected_edges=selected_edges,
            edge_weights_mat=kl_edge,
            coords_xyz=xyz_coords,
            node_names=electrode_labels,
            out_png=out_png,
            title=f"S{subj:02d} {metric} — Stable ∩ High-KL edges (n={len(selected_edges)})",
            node_size=14,
            display_mode="lzr",
        )



        plot_scalp_graph(
            edges=selected_edges,
            pos_xy=xy_coords,
            ch_names=electrode_labels,
            title=f"S{subj:02d} {metric} - Stable ∩ High-KL edges (n={len(selected_edges)})",
            out_base=os.path.join(subj_dir, f"S{subj:02d}_{metric}_stable_highKL_edges_graph"),
            dpi=PLOT_DPI,
        )

    # (1) Extra trial-wise plots for chosen metric
    extra = results_by_subj[subj]["extra_trials"]
    extra_metric = extra["metric"]
    if extra["Edges"] is not None and extra["NS"] is not None:
        sess = results_by_subj[subj]["session_sizes"]
        plot_metric_ns_trajectories(
            subject=subj,
            metric_name=extra_metric,
            ns_all=extra["NS"],
            labels_all=extra["Labels"],
            session_sizes_by_class=sess,
            node_scores=kl_ns_per_subject[extra_metric][subj],
            ch_names=electrode_labels,
            out_base=os.path.join(subj_dir, f"S{subj:02d}"),
            top_k_nodes=8,
            dpi=PLOT_DPI,
        )
        plot_metric_edge_trajectories(
            subject=subj,
            metric_name=extra_metric,
            edges_all=extra["Edges"],
            labels_all=extra["Labels"],
            session_sizes_by_class=sess,
            edge_score_mat=kl_edge_per_subject[extra_metric][subj],
            ch_names=electrode_labels,
            out_base=os.path.join(subj_dir, f"S{subj:02d}"),
            top_k_edges=10,
            dpi=PLOT_DPI,
        )
        plot_metric_recurrence(
            subject=subj,
            metric_name=extra_metric,
            edges_all=extra["Edges"],
            labels_all=extra["Labels"],
            session_sizes_by_class=sess,
            out_base=os.path.join(subj_dir, f"S{subj:02d}"),
            dpi=PLOT_DPI,
        )
        plot_metric_cross_session_similarity(
            subject=subj,
            metric_name=extra_metric,
            edges_all=extra["Edges"],
            labels_all=extra["Labels"],
            session_sizes_by_class=sess,
            out_base=os.path.join(subj_dir, f"S{subj:02d}"),
            dpi=PLOT_DPI,
        )


# ============================================================
# SAVE CSVs
# ============================================================

pd.DataFrame(distance_spearman_summary).to_csv(os.path.join(output_dir, "distance_vs_metric_spearman.csv"), index=False)
pd.DataFrame(electrode_distance_corr_rows).to_csv(os.path.join(output_dir, "per_electrode_distance_correlations.csv"), index=False)

stability_df = pd.DataFrame(stability_summary_rows)
stability_df.to_csv(os.path.join(output_dir, "stability_cv_summary.csv"), index=False)

discriminability_df = pd.DataFrame(discriminability_summary_rows)
discriminability_df.to_csv(os.path.join(output_dir, "discriminability_KL_summary.csv"), index=False)

sd_counts_df = pd.DataFrame(sd_feature_count_rows)
sd_counts_df.to_csv(os.path.join(output_dir, "stable_highKL_feature_counts_edges_ns.csv"), index=False)

sd_detail_df = pd.DataFrame(sd_feature_detail_rows)
sd_detail_df.to_csv(os.path.join(output_dir, "stable_highKL_edges_ns_detail.csv"), index=False)


# ============================================================
# (5) VOTING: edges + ns only, and ONE combined grouped bar plot
# ============================================================

# Subject-level average stability metrics per metric
stab_grouped = (
    stability_df
    .groupby(["subject", "metric"], as_index=False)
    .agg({
        "mean_cv_edges": "mean",
        "mean_cv_ns": "mean",
    })
)

disc_grouped = (
    discriminability_df[["subject", "metric", "mean_kl_edges", "mean_kl_ns"]]
    .copy()
)

df_vote = sd_counts_df.merge(
    stab_grouped,
    on=["subject", "metric"],
    how="left",
    validate="many_to_one"
).merge(
    disc_grouped,
    on=["subject", "metric"],
    how="left",
    validate="many_to_one"
)

def pick_cv_kl(row):
    if row["feature_type"] == "edges":
        return row["mean_cv_edges"], row["mean_kl_edges"]
    elif row["feature_type"] == "ns":
        return row["mean_cv_ns"], row["mean_kl_ns"]
    else:
        raise ValueError("Only edges/ns expected here.")

df_vote["cv"], df_vote["kl"], df_vote["frac_sd"] = np.nan, np.nan, np.nan
for idx, r in df_vote.iterrows():
    cv_val, kl_val = pick_cv_kl(r)
    n_valid = r["n_features_valid"]
    n_sd = r["stable_and_highKL_count"]
    df_vote.loc[idx, "cv"] = cv_val
    df_vote.loc[idx, "kl"] = kl_val
    df_vote.loc[idx, "frac_sd"] = (n_sd / n_valid) if n_valid > 0 else 0.0

# rank within subject separately for edges and ns
def rank_group(g):
    g = g.copy()
    if g.shape[0] == 1:
        g["rank_frac_sd"] = 1.0
        g["rank_kl"] = 1.0
        g["rank_cv"] = 1.0
    else:
        g["rank_frac_sd"] = g["frac_sd"].rank(method="min", ascending=False)
        g["rank_kl"] = g["kl"].rank(method="min", ascending=False)
        g["rank_cv"] = g["cv"].rank(method="min", ascending=True)

    # Borda-like sum (lower is better because rank 1 is best)
    g["borda_score_subject"] = g["rank_frac_sd"] + g["rank_kl"] + g["rank_cv"]
    g["subject_rank"] = g["borda_score_subject"].rank(method="min", ascending=True)
    return g

df_ranked = df_vote.groupby(["subject", "feature_type"], group_keys=False).apply(rank_group)
df_ranked.to_csv(os.path.join(output_dir, "metric_voting_per_subject_edges_ns.csv"), index=False)

# "Top-2" counts per metric for edges and ns
top2 = df_ranked[df_ranked["subject_rank"] <= 2].copy()
counts_edges = top2[top2["feature_type"] == "edges"].groupby("metric")["subject"].nunique()
counts_ns = top2[top2["feature_type"] == "ns"].groupby("metric")["subject"].nunique()

metrics_all = sorted(set(counts_edges.index).union(set(counts_ns.index)))
vals_edges = np.array([counts_edges.get(m, 0) for m in metrics_all], dtype=float)
vals_ns = np.array([counts_ns.get(m, 0) for m in metrics_all], dtype=float)

# ONE combined grouped bar plot (edges vs ns)
x = np.arange(len(metrics_all))
width = 0.42

fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x - width/2, vals_edges, width=width, label="Edges (top-2 count)")
ax.bar(x + width/2, vals_ns, width=width, label="Node strength (top-2 count)")
ax.set_xticks(x)
ax.set_xticklabels(metrics_all, rotation=45, ha="right")
ax.set_ylabel("Number of subjects (ranked 1st or 2nd)")
ax.set_xlabel("Connectivity metric")
ax.set_title("KL+stability voting: top-2 frequency (edges vs node strength)", fontsize=12)
ax.legend()
savefig(fig, os.path.join(output_dir, "metric_top2_counts_edges_vs_ns_ONEPLOT"), dpi=PLOT_DPI, also_pdf=True)


# ============================================================
# SAVE PER-TRIAL FEATURE DICT (ONLY S∩High-KL NON-ZERO)
# ============================================================

pkl_path = os.path.join(output_dir, f"{SAVE_FEATURE_METRIC.lower()}_per_trial_features_by_subject.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Saved per-trial feature dict to:", pkl_path)
print("Analysis complete. Outputs in:", output_dir)
