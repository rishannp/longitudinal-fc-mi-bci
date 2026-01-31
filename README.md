# Stability of Functional Connectivity Features for Longitudinal Motor Imagery BCIs

This repository accompanies the paper:

**Stability and Neurophysiological Validity of Graph Connectivity Features for Non-stationary Motor Imagery BCIs**  
Rishan Patel, Barney Bryson, Tom Carlson, Andreas Demosthenous, Dai Jiang

---

## Motivation

Motor imagery (MI) EEG-based brain–computer interfaces (BCIs) are fundamentally limited by **longitudinal non-stationarity**. Feature distributions drift across sessions due to physiological, behavioural, and disease-related factors—particularly in clinical populations such as amyotrophic lateral sclerosis (ALS). As a result, models trained on early calibration data often fail to generalise over time.

Functional connectivity (FC) has been proposed as an alternative feature space capable of capturing distributed neural dynamics. However, the field lacks clear guidance on **which connectivity estimators produce features that are both stable over time and discriminative between MI classes**.

This work addresses that gap.

---

## What This Repository Is About

This repository implements a **feature-level evaluation framework** for EEG functional connectivity representations under realistic multi-session drift. Rather than proposing a new classifier or deep model, the focus is on **representation validity**.

Specifically, we ask:

- Which connectivity-derived features remain reproducible across sessions?
- Which preserve class separability despite non-stationarity?
- Which exhibit plausible sensorimotor organisation?
- Do stability-informed features improve cross-session decoding compared to CSP?

The emphasis is on **principled feature selection under drift**, not architectural novelty.

---

## Core Ideas

### Stability Is Not Optional  
A feature that is highly discriminative but unstable across sessions is unsuitable for longitudinal BCIs. This work explicitly quantifies **trial-wise reproducibility** before considering decoding performance.

### Discriminability Must Persist Under Drift  
Class separability is evaluated across the full empirical distributions of features, rather than relying on session-specific means or within-session optimisation.

### Neurophysiology Matters  
Connectivity features are validated against known MI-related sensorimotor organisation, rather than treated as abstract graph inputs.

### Decoding Is a Validation Step, Not the Goal  
Classification is used to verify whether stability-informed feature selection translates into improved temporal generalisation—nothing more.

---

## Methodological Summary

- EEG trials are transformed into weighted functional connectivity graphs using a broad family of estimators:
  - Phase-based (PLV, ImagPLV, PLI, wPLI)
  - Spectral (Coherence, Imaginary Coherence, Magnitude-Squared Coherence)
  - Information-theoretic (Cross Mutual Information)
  - Cross-frequency coupling

- From each graph, **node-level (strength)** and **edge-level** features are extracted.

- Feature quality is assessed along two independent axes:
  - **Stability**: quantified via coefficient of variation (CV)
  - **Discriminability**: quantified via symmetric Kullback–Leibler divergence
- Features that are simultaneously stable and discriminative are identified using percentile-based criteria.
- Neurophysiological plausibility is assessed using spatial topographies and distance-dependence controls.
- Selected feature sets are evaluated in a **strictly temporal train–test protocol** and compared against CSP.
---

## Key Findings

- Stability and discriminability can co-exist at the feature level, but only for a subset of connectivity features.
- Coherence-based metrics—particularly **magnitude-squared coherence (MSC)**—most consistently yield favourable stability–discriminability trade-offs across subjects.
- Node-strength representations derived from FC exhibit structured, lateralised sensorimotor patterns consistent with MI physiology.
- Stability-informed FC features demonstrate **more consistent cross-session decoding performance** than CSP for most subjects, under severe non-stationarity.
- Metrics explicitly designed to suppress zero-lag coupling (e.g. PLI, wPLI, imaginary coherence) are not necessarily more robust under longitudinal drift.

---

## Dataset

Experiments are conducted on a **longitudinal ALS motor imagery EEG dataset** comprising:

- 8 ALS participants  
- 4 sessions per subject over 1–2 months  
- Approximately 160 trials per class (LH / RH MI)  
- 19-channel 10–20 montage  

The dataset is publicly available via the UCL Research Data Repository.

---

## Design Philosophy

- Feature representations should be evaluated **before** model selection  
- Temporal integrity must be preserved (no session leakage)  
- Interpretability and physiological plausibility are first-class constraints  
- Stability under drift is a necessary condition for deployment  

---

## Citation

If you use this repository or the analysis framework, please cite:

```bibtex
@article{Patel2025FCStability,
  title   = {Stability and Neurophysiological Validity of Graph Connectivity Features for Non-stationary Motor Imagery BCIs},
  author  = {Patel, Rishan and Bryson, Barney and Carlson, Tom and Demosthenous, Andreas and Jiang, Dai},
  journal = {Journal of Neural Engineering},
  year    = {2025}
}
