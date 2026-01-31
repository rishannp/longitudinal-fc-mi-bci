# -*- coding: utf-8 -*-
"""
CSP topomaps per subject (trained on the entire dataset).

Pipeline:
1) Load OGFS .mat per subject (L/R trials)
2) Bandpass filter (8–30 Hz)
3) OPTIONAL ASR (off by default)
4) Standardise trial length per subject (mode length, crop/pad)
5) Fit CSP on ALL trials for that subject
6) Plot CSP patterns (topomaps)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from mne.decoding import CSP

# Optional ASR (only used if USE_ASR=True)
from asrpy.asr import asr_calibrate, asr_process


# -----------------------------
# Config
# -----------------------------
CHANNELS = np.array([
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2'
])

DATA_DIR = r"C:\Users\uceerjp\Desktop\PhD\Multi-Session Data\OG_Full_Data"
SUBJECT_IDS = [1, 2, 5, 9, 21, 31, 34, 39]

FS = 256
BPF_LO, BPF_HI = 8, 30

USE_ASR = True
ASR_CUTOFF = 20.0

N_CSP_COMPONENTS = 4  # number of CSP maps to visualise per subject

DROP_TOO_SHORT = True
MIN_LEN_FRACTION = 0.5  # drop trial if < 50% of target_len


# -----------------------------
# Utilities
# -----------------------------
def remove_last_entry_if_all_zeros(data_list):
    if len(data_list) > 0 and np.all(data_list[-1] == 0):
        return data_list[:-1]
    return data_list


def load_subjects_ogfs_mat(data_dir, subject_ids):
    """
    Returns dict:
      data_by_subject['S1']['L'] = list of trials [samples x channels]
      data_by_subject['S1']['R'] = list of trials [samples x channels]
    """
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

        void_array = mat[varname]  # 1 x N array of structs
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
                tr = tr[:, :19]  # keep first 19 chans
                y = np.zeros_like(tr, dtype=np.float64)
                for ch in range(tr.shape[1]):
                    y[:, ch] = filtfilt(b, a, tr[:, ch])
                out[sid][lab].append(y)

    return out


def run_asr_subjectwise(filtered_data_by_subject, sfreq=256.0, cutoff=20.0):
    """
    Concatenate L+R trials per subject into "continuous" data, calibrate ASR,
    clean, then split back to trials (same order).
    """
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
        Xc = asr_process(X, sfreq=sfreq, M=M, T=T).T  # back to [n_samp x n_ch]

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


def build_csp_dataset_for_subject(d, target_len):
    """
    Returns X, y for CSP:
      X: [n_trials, n_channels, n_times]
      y: [n_trials] with 1=Left, 2=Right  (keeps your convention)
    """
    L = crop_pad_trials(d["L"], target_len, DROP_TOO_SHORT, MIN_LEN_FRACTION)
    R = crop_pad_trials(d["R"], target_len, DROP_TOO_SHORT, MIN_LEN_FRACTION)

    if len(L) == 0 or len(R) == 0:
        return None, None

    # stack, CSP expects [trials, channels, times]
    X = []
    y = []

    for tr in L:
        X.append(tr.T)   # [ch x time]
        y.append(1)

    for tr in R:
        X.append(tr.T)
        y.append(2)

    X = np.stack(X, axis=0).astype(np.float64)
    y = np.asarray(y, dtype=int)

    return X, y


def create_csp_info_and_montage(sfreq=256):
    ch_names = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'O1', 'O2'
    ]

    coords = [
        [0.950,  0.309, -0.0349],     # FP1
        [0.950, -0.309, -0.0349],     # FP2
        [0.587,  0.809, -0.0349],     # F7
        [0.673,  0.545,  0.500],      # F3
        [0.719,  0.000,  0.695],      # FZ
        [0.673, -0.545,  0.500],      # F4
        [0.587, -0.809, -0.0349],     # F8
        [6.120e-17,  0.999, -0.0349], # T7
        [4.400e-17,  0.719,  0.695],  # C3
        [3.750e-33, -6.120e-17, 1.0], # CZ
        [4.400e-17, -0.719,  0.695],  # C4
        [6.120e-17, -0.999, -0.0349], # T8
        [-0.587,  0.809, -0.0349],    # P7
        [-0.673,  0.545,  0.500],     # P3
        [-0.719, -8.810e-17, 0.695],  # PZ
        [-0.673, -0.545,  0.500],     # P4
        [-0.587, -0.809, -0.0349],    # P8
        [-0.950,  0.309, -0.0349],    # O1
        [-0.950, -0.309, -0.0349]     # O2
    ]

    montage = mne.channels.make_dig_montage(
        ch_pos={ch: xyz for ch, xyz in zip(ch_names, coords)},
        coord_frame="head"
    )
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    return info


# -----------------------------
# Main
# -----------------------------
data_by_subject = load_subjects_ogfs_mat(DATA_DIR, SUBJECT_IDS)
data_by_subject = bandpass_filter_trials(data_by_subject, BPF_LO, BPF_HI, FS)

if USE_ASR:
    data_by_subject = run_asr_subjectwise(data_by_subject, sfreq=FS, cutoff=ASR_CUTOFF)

info = create_csp_info_and_montage(sfreq=FS)

for sid, d in data_by_subject.items():
    if len(d["L"]) == 0 or len(d["R"]) == 0:
        print(f"Skipping {sid}: missing L or R trials")
        continue

    target_len = choose_target_length_mode(d["L"], d["R"])
    X, y = build_csp_dataset_for_subject(d, target_len)

    if X is None:
        print(f"Skipping {sid}: not enough usable trials after crop/pad")
        continue

    print(f"\n{sid}: X={X.shape} (trials, ch, time), y counts: L={(y==1).sum()}, R={(y==2).sum()}, target_len={target_len}")

    csp = CSP(
        n_components=N_CSP_COMPONENTS,
        log=True,
        norm_trace=True,
        component_order="mutual_info"
    )
    csp.fit(X, y)

    fig = csp.plot_patterns(
    info,
    ch_type="eeg",
    components=list(range(N_CSP_COMPONENTS)),
    show=False
    )
    
    fig.suptitle(f"{sid} — CSP patterns (trained on ALL trials)", fontsize=14)
    plt.show()

