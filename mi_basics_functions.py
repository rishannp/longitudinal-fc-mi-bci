# -*- coding: utf-8 -*-
"""
mi_basics_functions.py

All reusable functions for:
- filtering / preprocessing
- connectivity metrics
- graph metrics
- KL & effect utilities
- plotting (publication-style)
"""

import os
import numpy as np
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inspect
from scipy.stats import spearmanr, ttest_1samp

import mne


# ============================================================
# GEOMETRY / TOPO HELPERS
# ============================================================

def rotate_xy(xy, deg=90.0):
    theta = np.deg2rad(deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    return xy @ R.T


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def savefig(fig, path_base, dpi=600, also_pdf=True):
    """
    Save both PNG (high dpi) and PDF (vector) for publication.
    """
    png_path = path_base + ".png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        pdf_path = path_base + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_topomap_mne(
    values,
    pos_xy,
    ch_names,
    title,
    out_base,
    cmap="viridis",
    dpi=600,
    vmin=None,
    vmax=None,
    show_names=True,
):
    values = np.asarray(values, dtype=float).ravel()
    pos_xy = np.asarray(pos_xy, dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    # Build kwargs that your installed MNE actually supports
    sig = inspect.signature(mne.viz.plot_topomap)
    params = sig.parameters

    topo_kwargs = dict(
        pos=pos_xy,
        axes=ax,
        cmap=cmap,
        contours=6,
        sensors=True,
        show=False,
        outlines="head",
        sphere=0.10,
    )

    # Handle value limits in a version-safe way
    if vmin is not None and vmax is not None:
        if "vmin" in params and "vmax" in params:
            topo_kwargs["vmin"] = vmin
            topo_kwargs["vmax"] = vmax
        elif "clim" in params:
            topo_kwargs["clim"] = dict(kind="value", lim=(vmin, vmax))
        elif "vlim" in params:
            topo_kwargs["vlim"] = (vmin, vmax)
        elif "vscale" in params:
            # very old / uncommon API: fall back to no explicit limits
            pass

    im, _ = mne.viz.plot_topomap(values, **topo_kwargs)

    if show_names:
        for (x, y), nm in zip(pos_xy, ch_names):
            ax.text(x, y, nm, fontsize=7, ha="center", va="center")

    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # If you have a helper savefig() use it; otherwise:
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def plot_scalp_graph(
    edges,
    pos_xy,
    ch_names,
    title,
    out_base,
    edge_alpha=0.7,
    edge_lw=1.1,
    node_size=35,
    dpi=600,
):
    """
    Draw edges on scalp layout.
    edges: list[(i,j)]
    """
    pos_xy = np.asarray(pos_xy)
    xs, ys = pos_xy[:, 0], pos_xy[:, 1]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    # edges first
    for (i, j) in edges:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], alpha=edge_alpha, linewidth=edge_lw)

    ax.scatter(xs, ys, s=node_size)
    for k, nm in enumerate(ch_names):
        ax.text(xs[k], ys[k], nm, fontsize=7, ha="center", va="center")

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    savefig(fig, out_base, dpi=dpi, also_pdf=True)

from nilearn import plotting

def plot_stable_highkl_edges_nilearn(
    subject,
    metric,
    selected_edges,
    edge_weights_mat,
    coords_xyz,
    node_names,
    out_png,
    title="",
    node_size=12,
    display_mode="lzr",
):
    """
    Nilearn glass-brain connectivity plot for a selected edge list.

    Important:
    - coords_xyz here are your unit-sphere-ish EEG coordinates.
    - We map them into an MNI-like mm space so nilearn has something sensible to render.
    """
    import numpy as np

    try:
        from nilearn import plotting
    except Exception as e:
        raise ImportError(
            "Nilearn is required for plot_stable_highkl_edges_nilearn. "
            "Install with: pip install nilearn"
        ) from e

    n_nodes = coords_xyz.shape[0]

    # ----------------------------
    # Build adjacency (only selected edges)
    # ----------------------------
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    if selected_edges is not None and len(selected_edges) > 0:
        for (i, j) in selected_edges:
            w = float(edge_weights_mat[i, j]) if edge_weights_mat is not None else 1.0
            if np.isfinite(w):
                adj[i, j] = w
                adj[j, i] = w

    # If everything is zero, bail early with a clear message
    if not np.any(adj):
        # still make a node-only plot so you can see something
        coords = coords_xyz.copy()
        coords = _eeg_unit_sphere_to_mni_mm(coords)
        disp = plotting.plot_connectome(
            adj,
            coords,
            node_size=node_size,
            display_mode=display_mode,
            title=f"{title} (no nonzero edges)",
            edge_threshold="100%",
            node_color="k",
        )
        disp.savefig(out_png, dpi=300)
        disp.close()
        return

    # ----------------------------
    # Map EEG coords -> MNI-ish coords (mm)
    # ----------------------------
    coords = _eeg_unit_sphere_to_mni_mm(coords_xyz)

    # ----------------------------
    # Plot
    # ----------------------------
    # edge_threshold="0%" means draw all nonzero edges.
    # If it gets too cluttered later, set e.g. "80%" to keep strongest edges only.
    disp = plotting.plot_connectome(
        adj,
        coords,
        node_size=node_size,
        display_mode=display_mode,
        title=title,
        edge_threshold="0%",
    )

    disp.savefig(out_png, dpi=300)
    disp.close()


def _eeg_unit_sphere_to_mni_mm(coords_xyz, scale_mm=80.0):
    """
    Heuristic mapping:
    Your coords look like: x=anterior, y=left, z=up (unit sphere-ish).
    MNI convention: x=left-right, y=posterior-anterior, z=inferior-superior.

    So:
      mni_x = y
      mni_y = x
      mni_z = z
    Then scale to mm.
    """
    import numpy as np

    coords_xyz = np.asarray(coords_xyz, dtype=float)
    mni = np.zeros_like(coords_xyz)
    mni[:, 0] = coords_xyz[:, 1]  # left-right
    mni[:, 1] = coords_xyz[:, 0]  # post-ant
    mni[:, 2] = coords_xyz[:, 2]  # inf-sup
    return mni * float(scale_mm)

# ============================================================
# FILTERS
# ============================================================

def design_bandpass(band, fs, order=4):
    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = sig.butter(order, [low, high], btype="bandpass")
    return b, a


def apply_filtfilt(x, b, a):
    return sig.filtfilt(b, a, x, axis=0)


# ============================================================
# PREPROCESSING (PER TRIAL AFTER SUBJECT-LEVEL ASR/ICA)
# ============================================================

def preprocess_trial_after_asr(
    x,
    num_electrodes,
    apply_car=False,
    apply_zscore=False
):
    """
    x: [time, channels] after subject-level preprocessing.
    """
    x = x[:, :num_electrodes]

    if apply_car:
        x = x - np.mean(x, axis=1, keepdims=True)

    if apply_zscore:
        mean_ch = np.mean(x, axis=0, keepdims=True)
        std_ch = np.std(x, axis=0, keepdims=True) + 1e-10
        x = (x - mean_ch) / std_ch

    return x


def preprocess_trial_cfc_after_asr(
    x,
    num_electrodes,
    apply_car=False,
    apply_zscore=False
):
    """
    For CFC input: no extra bandpass here (LOW/HIGH done inside CFC).
    """
    x = x[:, :num_electrodes]

    if apply_car:
        x = x - np.mean(x, axis=1, keepdims=True)

    if apply_zscore:
        mean_ch = np.mean(x, axis=0, keepdims=True)
        std_ch = np.std(x, axis=0, keepdims=True) + 1e-10
        x = (x - mean_ch) / std_ch

    return x


# ============================================================
# CONNECTIVITY METRICS
# ============================================================

def plv_pli_wpli(eegData, num_electrodes):
    eegData = eegData[:, :num_electrodes]
    analytic = sig.hilbert(eegData, axis=0)
    phases = np.angle(analytic)

    n_ch = eegData.shape[1]
    plv_mat  = np.zeros((n_ch, n_ch))
    imag_mat = np.zeros((n_ch, n_ch))
    pli_mat  = np.zeros((n_ch, n_ch))
    wpli_mat = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            dphi = phases[:, j] - phases[:, i]
            complex_phase = np.exp(1j * dphi)
            mean_complex = np.mean(complex_phase)

            plv = np.abs(mean_complex)
            imag_plv = np.imag(mean_complex)
            pli = np.abs(np.mean(np.sign(np.sin(dphi))))

            im_cross = np.imag(analytic[:, i] * np.conj(analytic[:, j]))
            num = np.abs(np.mean(im_cross))
            den = np.mean(np.abs(im_cross)) + 1e-10
            wpli = num / den

            plv_mat[i, j]  = plv
            plv_mat[j, i]  = plv
            imag_mat[i, j] = imag_plv
            imag_mat[j, i] = imag_plv
            pli_mat[i, j]  = pli
            pli_mat[j, i]  = pli
            wpli_mat[i, j] = wpli
            wpli_mat[j, i] = wpli

    return plv_mat, imag_mat, pli_mat, wpli_mat


def coherence_msc(eegData, fs, main_band, num_electrodes):
    eegData = eegData[:, :num_electrodes]
    n_ch = eegData.shape[1]

    f_ref, _ = sig.welch(eegData[:, 0], fs=fs, nperseg=256)
    band_mask = (f_ref >= main_band[0]) & (f_ref <= main_band[1])

    coh_mat   = np.zeros((n_ch, n_ch))
    imcoh_mat = np.zeros((n_ch, n_ch))
    msc_mat   = np.zeros((n_ch, n_ch))

    Pxx = []
    for ch in range(n_ch):
        _, Pch = sig.welch(eegData[:, ch], fs=fs, nperseg=256)
        Pxx.append(Pch)
    Pxx = np.array(Pxx)

    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            _, Pxy = sig.csd(eegData[:, i], eegData[:, j], fs=fs, nperseg=256)

            coh_f   = np.abs(Pxy) ** 2 / (Pxx[i] * Pxx[j] + 1e-10)
            coh_band = coh_f[band_mask]
            msc = np.mean(coh_band)

            coh_complex      = Pxy / (np.sqrt(Pxx[i] * Pxx[j]) + 1e-10)
            coh_complex_band = coh_complex[band_mask]
            coh_complex_mean = np.mean(coh_complex_band)
            imcoh_val = np.imag(coh_complex_mean)
            coh_val   = np.abs(coh_complex_mean)

            coh_mat[i, j]   = coh_val
            coh_mat[j, i]   = coh_val
            imcoh_mat[i, j] = imcoh_val
            imcoh_mat[j, i] = imcoh_val
            msc_mat[i, j]   = msc
            msc_mat[j, i]   = msc

    return coh_mat, imcoh_mat, msc_mat


def compute_cmi(eegData, num_electrodes, n_bins=8):
    eegData = eegData[:, :num_electrodes]
    _, n_ch = eegData.shape
    cmi_mat = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        x = eegData[:, i]
        for j in range(i + 1, n_ch):
            y = eegData[:, j]
            H, _, _ = np.histogram2d(x, y, bins=n_bins)
            Pxy = H / (np.sum(H) + 1e-12)
            Px = np.sum(Pxy, axis=1, keepdims=True)
            Py = np.sum(Pxy, axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio     = Pxy / (Px @ Py + 1e-12)
                log_ratio = np.log(ratio + 1e-12)
                mi        = np.nansum(Pxy * log_ratio)
            cmi_mat[i, j] = mi
            cmi_mat[j, i] = mi

    return cmi_mat


def tort_modulation_index(phase, amp, n_bins=18):
    phase = np.angle(np.exp(1j * phase))
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    digitized = np.digitize(phase, bin_edges) - 1
    amp_means = np.zeros(n_bins)

    for b in range(n_bins):
        mask = digitized == b
        amp_means[b] = np.mean(amp[mask]) if np.any(mask) else 0.0

    P = amp_means / (np.sum(amp_means) + 1e-12)
    uniform = np.ones(n_bins) / n_bins
    with np.errstate(divide="ignore", invalid="ignore"):
        kld = np.nansum(P * np.log((P + 1e-12) / (uniform + 1e-12)))
    mi = kld / np.log(n_bins)
    return mi


def compute_cfc(eegData, num_electrodes, low_b, low_a, high_b, high_a):
    eegData = eegData[:, :num_electrodes]
    _, n_ch = eegData.shape

    low_filt = sig.filtfilt(low_b, low_a, eegData, axis=0)
    low_phase = np.angle(sig.hilbert(low_filt, axis=0))

    high_filt = sig.filtfilt(high_b, high_a, eegData, axis=0)
    high_amp = np.abs(sig.hilbert(high_filt, axis=0))

    cfc_mat = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            mi_ij = tort_modulation_index(low_phase[:, i], high_amp[:, j])
            mi_ji = tort_modulation_index(low_phase[:, j], high_amp[:, i])
            mi_sym = 0.5 * (mi_ij + mi_ji)
            cfc_mat[i, j] = mi_sym
            cfc_mat[j, i] = mi_sym

    return cfc_mat


# ============================================================
# GRAPH METRICS
# ============================================================

def compute_node_strength(W):
    W_abs = np.abs(W)
    np.fill_diagonal(W_abs, 0.0)
    return np.sum(W_abs, axis=1)


def weighted_clustering_coef(W):
    W = np.abs(W).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    if np.max(W) > 0:
        W = W / np.max(W)

    C = np.zeros(n)
    for i in range(n):
        neighbors = np.where(W[i, :] > 0)[0]
        k_i = len(neighbors)
        if k_i < 2:
            C[i] = 0.0
            continue
        tri_sum = 0.0
        for idx_j in range(k_i):
            j = neighbors[idx_j]
            for idx_k in range(idx_j + 1, k_i):
                k = neighbors[idx_k]
                tri_sum += (W[i, j] * W[i, k] * W[j, k]) ** (1.0 / 3.0)
        C[i] = (2.0 * tri_sum) / (k_i * (k_i - 1))
    return C


# ============================================================
# FOURIER SURROGATE
# ============================================================

def fourier_surrogate(eegData, num_electrodes):
    eegData = eegData[:, :num_electrodes]
    n_t, n_ch = eegData.shape
    surrogate = np.zeros_like(eegData)
    for ch in range(n_ch):
        x = eegData[:, ch]
        Xf = np.fft.rfft(x)
        amp = np.abs(Xf)
        phase = np.angle(Xf)
        random_phase = np.random.uniform(0, 2 * np.pi, size=phase.shape)
        random_phase[0] = phase[0]
        Xf_sur = amp * np.exp(1j * random_phase)
        surrogate[:, ch] = np.fft.irfft(Xf_sur, n=n_t)
    return surrogate


# ============================================================
# KL / EFFECT UTILS
# ============================================================

def symmetric_kl_from_samples(x, y, n_bins=20, eps=1e-12):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return 0.0

    data = np.concatenate([x, y])
    if np.allclose(data, data[0]):
        return 0.0

    data_min = np.min(data)
    data_max = np.max(data)
    if np.isclose(data_max, data_min):
        return 0.0

    bins = np.linspace(data_min, data_max, int(n_bins) + 1)
    Px, _ = np.histogram(x, bins=bins)
    Py, _ = np.histogram(y, bins=bins)

    Px = Px.astype(np.float64) + eps
    Py = Py.astype(np.float64) + eps
    Px /= Px.sum()
    Py /= Py.sum()

    kl_xy = np.sum(Px * np.log(Px / Py))
    kl_yx = np.sum(Py * np.log(Py / Px))
    return 0.5 * (kl_xy + kl_yx)


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / float(nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return (mx - my) / np.sqrt(pooled + 1e-12)


def flatten_upper_tri(mat):
    i, j = np.triu_indices(mat.shape[0], k=1)
    return mat[i, j]


# ============================================================
# EXTRA TRIAL-WISE PLOTS (PARAMETRISED BY METRIC)
# ============================================================

def _get_class_trial_indices(labels, cls_val):
    labels = np.asarray(labels)
    return np.where(labels == cls_val)[0]


def _plot_session_boundaries(ax, session_sizes_cls):
    if session_sizes_cls is None or len(session_sizes_cls) == 0:
        return
    cum_sizes = np.cumsum(session_sizes_cls)
    for s in cum_sizes[:-1]:
        ax.axvline(s - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)
        # if square, draw both
        if abs((ax.get_xbound()[1] - ax.get_xbound()[0]) - (ax.get_ybound()[1] - ax.get_ybound()[0])) < 1e-9:
            ax.axhline(s - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)


def plot_metric_ns_trajectories(
    subject,
    metric_name,
    ns_all,
    labels_all,
    session_sizes_by_class,
    node_scores,          # KL node scores used only for picking top nodes
    ch_names,
    out_base,
    top_k_nodes=8,
    dpi=600,
):
    if ns_all is None or labels_all is None:
        return

    labels = np.asarray(labels_all)
    ns_all = np.asarray(ns_all)

    node_scores = np.asarray(node_scores)
    order = np.argsort(np.abs(node_scores))[::-1][:min(top_k_nodes, len(node_scores))]
    node_labels = [ch_names[i] for i in order]

    for cls_val, cls_name in [(0, "L"), (1, "R")]:
        idx = _get_class_trial_indices(labels, cls_val)
        if idx.size < 2:
            continue
        ns_cls = ns_all[idx]
        T = ns_cls.shape[0]

        fig, ax = plt.subplots(figsize=(9, 4))
        for k, node_idx in enumerate(order):
            ax.plot(np.arange(T), ns_cls[:, node_idx], marker=".", linewidth=1.0, label=node_labels[k])

        sess_sizes = session_sizes_by_class.get(cls_name, None)
        if sess_sizes:
            for s in np.cumsum(sess_sizes)[:-1]:
                ax.axvline(s - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Trial index (within class)")
        ax.set_ylabel(f"{metric_name} node strength")
        ax.set_title(f"S{subject:02d} {metric_name} - NS trajectories ({cls_name})", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        savefig(fig, out_base + f"_{metric_name}_NS_trajectories_{cls_name}", dpi=dpi, also_pdf=True)


def plot_metric_edge_trajectories(
    subject,
    metric_name,
    edges_all,
    labels_all,
    session_sizes_by_class,
    edge_score_mat,  # KL edge mat used only for picking top edges
    ch_names,
    out_base,
    top_k_edges=10,
    dpi=600,
):
    if edges_all is None or labels_all is None:
        return

    labels = np.asarray(labels_all)
    edges_all = np.asarray(edges_all)

    d_flat = flatten_upper_tri(edge_score_mat)
    iu, ju = np.triu_indices(edge_score_mat.shape[0], k=1)
    order = np.argsort(d_flat)[::-1][:min(top_k_edges, len(d_flat))]
    selected_edges = [(int(iu[idx]), int(ju[idx])) for idx in order]
    edge_names = [f"{ch_names[i]}-{ch_names[j]}" for (i, j) in selected_edges]

    for cls_val, cls_name in [(0, "L"), (1, "R")]:
        idx = _get_class_trial_indices(labels, cls_val)
        if idx.size < 2:
            continue
        W_cls = edges_all[idx]
        T = W_cls.shape[0]

        fig, ax = plt.subplots(figsize=(9, 4))
        for k, (i, j) in enumerate(selected_edges):
            ax.plot(np.arange(T), W_cls[:, i, j], marker=".", linewidth=1.0, label=edge_names[k])

        sess_sizes = session_sizes_by_class.get(cls_name, None)
        if sess_sizes:
            for s in np.cumsum(sess_sizes)[:-1]:
                ax.axvline(s - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Trial index (within class)")
        ax.set_ylabel(f"{metric_name} edge weight")
        ax.set_title(f"S{subject:02d} {metric_name} - Top-KL edge trajectories ({cls_name})", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        savefig(fig, out_base + f"_{metric_name}_edge_trajectories_{cls_name}", dpi=dpi, also_pdf=True)


def plot_metric_recurrence(
    subject,
    metric_name,
    edges_all,
    labels_all,
    session_sizes_by_class,
    out_base,
    dpi=600,
):
    if edges_all is None or labels_all is None:
        return

    labels = np.asarray(labels_all)
    edges_all = np.asarray(edges_all)
    n_elec = edges_all.shape[1]
    iu, ju = np.triu_indices(n_elec, k=1)

    for cls_val, cls_name in [(0, "L"), (1, "R")]:
        idx = _get_class_trial_indices(labels, cls_val)
        if idx.size < 2:
            continue

        W_cls = edges_all[idx]
        T = W_cls.shape[0]
        F = iu.shape[0]
        V = np.zeros((T, F), dtype=np.float64)

        for t in range(T):
            V[t] = W_cls[t][iu, ju]

        V = V - np.mean(V, axis=1, keepdims=True)
        V = V / (np.std(V, axis=1, keepdims=True) + 1e-10)
        S = np.corrcoef(V)
        S = np.nan_to_num(S, nan=0.0)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        im = ax.imshow(S, vmin=-1, vmax=1, cmap="seismic", origin="upper")
        fig.colorbar(im, ax=ax, label=f"corr(flat({metric_name}))")
        _plot_session_boundaries(ax, session_sizes_by_class.get(cls_name, None))
        ax.set_xlabel("Trial index")
        ax.set_ylabel("Trial index")
        ax.set_title(f"S{subject:02d} {metric_name} - Recurrence ({cls_name})", fontsize=12)
        savefig(fig, out_base + f"_{metric_name}_recurrence_{cls_name}", dpi=dpi, also_pdf=True)


def plot_metric_cross_session_similarity(
    subject,
    metric_name,
    edges_all,
    labels_all,
    session_sizes_by_class,
    out_base,
    dpi=600,
):
    if edges_all is None or labels_all is None:
        return

    labels = np.asarray(labels_all)
    edges_all = np.asarray(edges_all)
    n_elec = edges_all.shape[1]
    iu, ju = np.triu_indices(n_elec, k=1)

    for cls_val, cls_name in [(0, "L"), (1, "R")]:
        idx = _get_class_trial_indices(labels, cls_val)
        if idx.size < 2:
            continue

        W_cls = edges_all[idx]
        T = W_cls.shape[0]
        sess_sizes = session_sizes_by_class.get(cls_name, None)
        if not sess_sizes:
            continue

        s1_end = int(np.cumsum(sess_sizes)[0])
        s1_end = min(max(s1_end, 1), T)

        W_ref = np.mean(W_cls[:s1_end], axis=0)
        ref_vec = W_ref[iu, ju]
        ref_vec = ref_vec - np.mean(ref_vec)
        ref_den = np.std(ref_vec) + 1e-10

        sim = np.zeros(T)
        for t in range(T):
            vec = W_cls[t][iu, ju]
            vec = vec - np.mean(vec)
            den = np.std(vec) + 1e-10
            sim[t] = 0.0 if (den < 1e-10 or ref_den < 1e-10) else (np.dot(vec, ref_vec) / (den * ref_den))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(T), sim, marker=".", linewidth=1.0)
        for s in np.cumsum(sess_sizes)[:-1]:
            if s <= T:
                ax.axvline(s - 0.5, linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Trial index (within class)")
        ax.set_ylabel("Similarity to Session-1 mean (corr)")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"S{subject:02d} {metric_name} - Cross-session similarity ({cls_name})", fontsize=12)
        savefig(fig, out_base + f"_{metric_name}_cross_session_similarity_{cls_name}", dpi=dpi, also_pdf=True)
