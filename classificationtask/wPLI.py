import pickle
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
datadir = (
    "C:/Users/uceerjp/Desktop/PhD/Year 2/DeepLearning-on-ALS-MI-Data/"
    "Graphs/Graph Basics/connectivity_stability_discriminability/"
    "msc_per_trial_features_by_subject.pkl"
)
with open(datadir, "rb") as f:
    data = pickle.load(f)

electrode_labels = [
    "FP1", "FP2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

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

# ---------------------------------------------------------------------
# Helper: make sure I always work with index lists, not ambiguous masks
# ---------------------------------------------------------------------
def mask_to_indices(mask):
    """Convert boolean or integer mask into a clean integer index array."""
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return np.where(mask)[0]
    else:
        return mask.astype(int)

# ---------------------------------------------------------------------
# Helper: temporal 20% / 80% split with no shuffle
# ---------------------------------------------------------------------
def temporal_split(X, y, train_ratio=0.25):
    """Split trials in time: first train_ratio for train, rest for test."""
    n_trials = X.shape[0]
    split_idx = int(np.floor(train_ratio * n_trials))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------
def get_node_features(subj_dict):
    """Return node features and metadata for a subject using node_mask."""
    NS = subj_dict["NS"]  # (trials, n_electrodes_total)
    node_mask = mask_to_indices(subj_dict["node_mask"])

    X_nodes = NS[:, node_mask]  # (n_trials, n_used_nodes)
    node_names = [electrode_labels[i] for i in node_mask]
    node_xyz = xyz_coords[node_mask]

    return X_nodes, node_names, node_xyz, node_mask

# ---------------------------------------------------------------------
# Reordering labels: 0,1,0,1,...
# ---------------------------------------------------------------------
def make_alternating_order(labels, subj_id=None):
    """
    Build an order that interleaves class 0 and class 1 trials:
    [first 0, first 1, second 0, second 1, ...]
    while preserving the within-class order.

    If one class has more trials than the other, the leftovers are appended at the end.
    """
    labels = np.asarray(labels)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    # Sort indices to preserve their original relative order
    idx0 = np.sort(idx0)
    idx1 = np.sort(idx1)

    if len(idx0) == 0 or len(idx1) == 0:
        if subj_id is not None:
            print(f"[WARN] Subject {subj_id}: cannot build alternating order – one class is empty.")
        return np.arange(len(labels))

    # Interleave as much as possible
    m = min(len(idx0), len(idx1))
    order = []
    for k in range(m):
        order.append(idx0[k])  # class 0
        order.append(idx1[k])  # class 1

    # Append leftovers (if any)
    if len(idx0) > m:
        order.extend(idx0[m:])
    if len(idx1) > m:
        order.extend(idx1[m:])

    order = np.array(order, dtype=int)

    # Sanity check: length preserved
    assert len(order) == len(labels)

    if subj_id is not None:
        uniq_before, cnt_before = np.unique(labels, return_counts=True)
        uniq_after, cnt_after = np.unique(labels[order], return_counts=True)
        print(
            f"  [Reorder] Subject {subj_id}: class counts before: {dict(zip(uniq_before, cnt_before))}, "
            f"after: {dict(zip(uniq_after, cnt_after))}"
        )

    return order

# ---------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------
def get_edge_features(subj_dict):
    """
    Return edge features and metadata for a subject.

    - Uses ALL electrodes in Edges (no node_mask).
    - Uses only upper triangle (i < j).
    - Does NOT apply edge_mask (Edges already 0 for invalid edges).

    Returns:
        X_edges:        (n_trials, n_edges)
        edge_names:     list of "E1–E2" strings, len = n_edges
        edge_idx:       (n_edges, 2) array of (i,j) indices in [0..n_nodes-1]
        node_indices:   np.array of node indices used (0..n_nodes-1)
    """
    Edges = subj_dict["Edges"]  # (trials, n_electrodes, n_electrodes)
    Edges = np.asarray(Edges)
    n_trials, n_nodes, n_nodes2 = Edges.shape
    assert n_nodes == n_nodes2, "Edges must be square in last two dims."

    # Use upper triangle (no self-connections, i<j)
    tri_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)

    # Indices of all edges in upper triangle
    edge_idx = np.argwhere(tri_mask)  # shape (n_edges, 2)
    n_edges = edge_idx.shape[0]

    if n_edges == 0:
        print("[WARN] get_edge_features: no edges found in upper triangle.")
        X_edges = np.empty((n_trials, 0), dtype=Edges.dtype)
        edge_names = []
        node_indices = np.arange(n_nodes)
        return X_edges, edge_names, edge_idx, node_indices

    # Vectorized extraction: Edges[:, tri_mask] gives (n_trials, n_edges)
    X_edges = Edges[:, tri_mask]

    # Electrode labels for this subject:
    # assuming Edges is ordered exactly like electrode_labels[0:n_nodes]
    node_indices = np.arange(n_nodes)
    node_names = [electrode_labels[i] for i in node_indices]
    edge_names = [f"{node_names[i]}–{node_names[j]}" for i, j in edge_idx]

    return X_edges, edge_names, edge_idx, node_indices

# ---------------------------------------------------------------------
# Visualisation: node feature distributions
# ---------------------------------------------------------------------
def plot_node_feature_distributions(X_nodes, y, node_names, subj_id, max_plots=12):
    """
    Plot per-electrode feature distributions (Left vs Right) for this subject.
    """
    n_nodes = X_nodes.shape[1]
    if n_nodes == 0:
        print(f"[WARN] Skipping node distribution plots – no node features for Subject {subj_id}.")
        return

    n_plots = min(n_nodes, max_plots)
    n_rows = int(np.ceil(n_plots / 4))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(16, 3 * n_rows),
    )
    axes = np.ravel(axes)

    for idx_plot in range(n_plots):
        ax = axes[idx_plot]
        feat = X_nodes[:, idx_plot]
        left = feat[y == 0]
        right = feat[y == 1]

        # jitter x positions to show points per class
        x_left = np.zeros_like(left) + 0 + np.random.uniform(-0.05, 0.05, size=len(left))
        x_right = np.zeros_like(right) + 1 + np.random.uniform(-0.05, 0.05, size=len(right))

        ax.scatter(x_left, left, alpha=0.6, label="Left (0)")
        ax.scatter(x_right, right, alpha=0.6, label="Right (1)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"])
        ax.set_title(node_names[idx_plot])

    # Remove unused axes
    for k in range(n_plots, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"Node feature distributions by class – Subject {subj_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Visualisation: node discriminability topomap
# ---------------------------------------------------------------------
def plot_node_discriminability_topomap(X_nodes, y, node_xyz, node_names, subj_id):
    """Plot per-electrode |mean difference| as a 3D scalp scatter."""
    if X_nodes.shape[1] == 0:
        print(f"[WARN] Skipping node topomap – no node features for Subject {subj_id}.")
        return

    left_mask = (y == 0)
    right_mask = (y == 1)

    mean_left = X_nodes[left_mask].mean(axis=0)
    mean_right = X_nodes[right_mask].mean(axis=0)
    disc = np.abs(mean_right - mean_left)  # simple absolute difference

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
        c=disc, s=100, cmap="viridis"
    )

    for (x, y_coord, z, name) in zip(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], node_names):
        ax.text(x, y_coord, z + 0.03, name, fontsize=8)

    fig.colorbar(sc, ax=ax, label="|mean(Right) - mean(Left)|")
    ax.set_title(f"Node discriminability topomap – Subject {subj_id}")
    plt.show()

# ---------------------------------------------------------------------
# PCA feature space (2D or 3D) – visual only
# ---------------------------------------------------------------------
def plot_pca_feature_space(X, y, subj_id, feature_type="Node features"):
    """
    Project feature space using PCA and colour by class.
    - 3D PCA if possible (>=3 components)
    - 2D PCA if only 2 components possible
    - skip if <2 features or samples
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"{feature_type} for Subject {subj_id} is not 2D: shape={X.shape}")

    n_samples, n_features = X.shape
    if n_features == 0:
        print(f"[WARN] Skipping PCA for {feature_type} – Subject {subj_id}: no features (0 cols).")
        return

    max_components = min(n_samples, n_features)
    if max_components < 2:
        print(
            f"[WARN] Skipping PCA for {feature_type} – Subject {subj_id}: "
            f"need at least 2 samples and 2 features, got shape {X.shape}"
        )
        return

    # Standardise
    X_std = StandardScaler().fit_transform(X)

    n_components = min(3, max_components)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_std)

    if n_components >= 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], Z[y == 0, 2], alpha=0.7, label="Left (0)")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], Z[y == 1, 2], alpha=0.7, label="Right (1)")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{feature_type} PCA (3D) – Subject {subj_id}")
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        # 2D
        plt.figure(figsize=(6, 5))
        plt.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=0.7, label="Left (0)")
        plt.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=0.7, label="Right (1)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{feature_type} PCA (2D) – Subject {subj_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------
# Discriminability + edge visualisation
# ---------------------------------------------------------------------
def compute_feature_discriminability(X, y):
    """Simple |mean difference| discriminability per feature."""
    left_mask = (y == 0)
    right_mask = (y == 1)
    mean_left = X[left_mask].mean(axis=0)
    mean_right = X[right_mask].mean(axis=0)
    return np.abs(mean_right - mean_left)

def plot_top_edges_distributions(X_edges, y, edge_names, subj_id, top_k=12):
    """
    Plot distributions for top_k most discriminative edges.
    """
    if X_edges.size == 0 or X_edges.shape[1] == 0:
        print(f"[WARN] Skipping edge distribution plots – no edge features for Subject {subj_id}.")
        return

    disc = compute_feature_discriminability(X_edges, y)
    top_k = min(top_k, X_edges.shape[1])
    top_idx = np.argsort(disc)[::-1][:top_k]

    n_rows = int(np.ceil(top_k / 4))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=4,
        figsize=(18, 3 * n_rows),
    )
    axes = np.ravel(axes)

    for plot_i, feat_idx in enumerate(top_idx):
        ax = axes[plot_i]
        feat = X_edges[:, feat_idx]
        left = feat[y == 0]
        right = feat[y == 1]

        x_left = np.zeros_like(left) + 0 + np.random.uniform(-0.05, 0.05, size=len(left))
        x_right = np.zeros_like(right) + 1 + np.random.uniform(-0.05, 0.05, size=len(right))

        ax.scatter(x_left, left, alpha=0.6, label="Left (0)")
        ax.scatter(x_right, right, alpha=0.6, label="Right (1)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Left", "Right"])
        ax.set_title(edge_names[feat_idx], fontsize=9)

    for k in range(top_k, len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"Top-{top_k} discriminative edges – Subject {subj_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Top {top_k} edges for Subject {subj_id}:")
    for feat_idx in top_idx:
        print(f"  {edge_names[feat_idx]}: discriminability={disc[feat_idx]:.4f}")

def plot_edge_discriminability_graph(edge_idx, disc, node_indices, subj_id, top_k=30):
    """
    Plot top_k most discriminative edges as lines in 3D between electrodes.
    """
    if disc.size == 0 or edge_idx.size == 0:
        print(f"[WARN] Skipping edge discriminability graph – no edges for Subject {subj_id}.")
        return

    node_xyz = xyz_coords[node_indices]
    node_names = [electrode_labels[i] for i in node_indices]

    top_k = min(top_k, disc.size)
    top_edges = np.argsort(disc)[::-1][:top_k]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    ax.scatter(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], s=60, color="k")

    for (x, y_coord, z, name) in zip(node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2], node_names):
        ax.text(x, y_coord, z + 0.03, name, fontsize=8)

    # Normalise line width from discriminability
    disc_top = disc[top_edges]
    if disc_top.size > 0 and disc_top.max() > 0:
        widths = 1 + 4 * (disc_top - disc_top.min()) / (disc_top.max() - disc_top.min())
    else:
        widths = np.ones_like(disc_top)

    for w, e_idx in zip(widths, top_edges):
        i, j = edge_idx[e_idx]
        x_vals = [node_xyz[i, 0], node_xyz[j, 0]]
        y_vals = [node_xyz[i, 1], node_xyz[j, 1]]
        z_vals = [node_xyz[i, 2], node_xyz[j, 2]]
        ax.plot(x_vals, y_vals, z_vals, linewidth=w)

    ax.set_title(f"Top-{top_k} discriminative edges – Subject {subj_id}")
    plt.show()

# ---------------------------------------------------------------------
# Classical models + hyperparameter grids
# ---------------------------------------------------------------------
def get_models_and_param_grids():
    """
    Return dict of (name -> (estimator, param_grid)).
    If param_grid is None, the base estimator is used without GridSearch.
    """
    models = {
        "LogReg_L2": (
            LogisticRegression(max_iter=2000, n_jobs=-1),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        ),
        "LinearSVM": (
            LinearSVC(max_iter=5000),
            {
                "C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        "RBF_SVM": (
            SVC(kernel="rbf"),
            {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.01, 0.001],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
            },
        ),
    }
    return models

def compute_binary_metrics(y_true, y_pred, positive_label=1):
    """
    Compute accuracy, balanced accuracy, kappa, sensitivity, specificity,
    precision, F1 for a given positive class.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # precision, recall, f1 for positive class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_label], average="binary", zero_division=0
    )

    # specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm = [[TN, FP],
    #       [FN, TP]]
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

def evaluate_models_temporal(X, y, subj_id, feature_type="Nodes"):
    """
    Run temporal 20%/80% split and evaluate a bunch of classical ML models.
    Includes:
      - Scaling
      - Optional PCA for classification
      - Subject-specific hyperparameter optimisation (GridSearchCV)
      - Multiple metrics: accuracy, balanced accuracy, kappa, sensitivity, etc.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{feature_type} – Subject {subj_id}: X must be 2D, got shape {X.shape}")
    if X.shape[1] == 0:
        print(f"[WARN] Skipping {feature_type} – Subject {subj_id}: no features (0 columns).")
        return {}

    X_train, X_test, y_train, y_test = temporal_split(X, y, train_ratio=0.2)

    # Check train classes
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    print(f"\n=== {feature_type} – Subject {subj_id} ===")
    print(f"Train trials: {len(y_train)}, Test trials: {len(y_test)}")
    print(f"  Train class distribution: {dict(zip(train_classes, train_counts))}")
    print(f"  Test  class distribution: {dict(zip(test_classes,  test_counts))}")

    if len(train_classes) < 2:
        print(
            f"[WARN] Skipping {feature_type} – Subject {subj_id}: "
            f"training split has only one class {train_classes[0]} under the temporal 20% rule."
        )
        return {}

    # Standardise features based on training set only
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Decide whether to apply PCA for classification
    # I do PCA when there are "enough" features (here > 5).
    use_pca = X_train_std.shape[1] > 5
    if use_pca:
        n_samples_train, n_features_train = X_train_std.shape
        max_components = min(n_samples_train - 1, n_features_train)
        n_components = min(30, max_components)  # cap dimensionality a bit

        if n_components >= 2:
            pca = PCA(n_components=n_components)
            X_train_clf = pca.fit_transform(X_train_std)
            X_test_clf = pca.transform(X_test_std)
            print(f"  [{feature_type}] Using PCA with n_components={n_components} for classification.")
        else:
            # Fallback: no PCA if we can't get at least 2 components
            X_train_clf = X_train_std
            X_test_clf = X_test_std
            print(f"  [{feature_type}] Skipping PCA for classification (insufficient rank).")
    else:
        X_train_clf = X_train_std
        X_test_clf = X_test_std
        print(f"  [{feature_type}] Not using PCA for classification (features <= 5).")

    models_and_grids = get_models_and_param_grids()
    results = {}

    for name, (base_est, param_grid) in models_and_grids.items():
        print(f"\n  -> Optimising {name} for {feature_type} – Subject {subj_id}")

        if param_grid is not None and len(param_grid) > 0:
            # Subject-specific hyperparam optimisation on training set only
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
            print(f"     Best params: {grid.best_params_}")
            print(f"     Best CV accuracy: {grid.best_score_:.3f}")
        else:
            # No grid, just fit the base model
            best_model = base_est
            best_model.fit(X_train_clf, y_train)

        # Evaluate on held-out 80% test set
        y_pred = best_model.predict(X_test_clf)
        metrics = compute_binary_metrics(y_test, y_pred, positive_label=1)
        results[name] = metrics

        print(
            f"     Test accuracy={metrics['accuracy']:.3f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.3f}, "
            f"kappa={metrics['kappa']:.3f}, "
            f"sens={metrics['sensitivity']:.3f}, "
            f"spec={metrics['specificity']:.3f}, "
            f"prec={metrics['precision']:.3f}, "
            f"f1={metrics['f1']:.3f}"
        )

    return results

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
all_results_nodes = {}
all_results_edges = {}
all_results_concat = {}

for subj_id in data:
    print(f"\n########## Subject {subj_id} ##########")
    subj_dict = data[subj_id]["MSC"]

    # Raw labels
    y_raw = np.asarray(subj_dict["Labels"]).astype(int)

    # Build alternating order 0,1,0,1,...
    order = make_alternating_order(y_raw, subj_id=subj_id)

    # Reorder everything per trial
    NS_reordered = subj_dict["NS"][order]
    Edges_reordered = subj_dict["Edges"][order]
    y = y_raw[order]

    print(f"  After reordering: NS shape = {NS_reordered.shape}, Edges shape = {Edges_reordered.shape}")
    uniq_y, cnt_y = np.unique(y, return_counts=True)
    print(f"  Label distribution (reordered): {dict(zip(uniq_y, cnt_y))}")

    # ---------------- Nodes ----------------
    X_nodes, node_names, node_xyz, node_mask = get_node_features(
        {"NS": NS_reordered, "node_mask": subj_dict["node_mask"]}
    )

    # Visualisations – nodes
    plot_node_feature_distributions(X_nodes, y, node_names, subj_id, max_plots=12)
    plot_node_discriminability_topomap(X_nodes, y, node_xyz, node_names, subj_id)
    plot_pca_feature_space(X_nodes, y, subj_id, feature_type="Node features")

    # Classification – node strength only
    res_nodes = evaluate_models_temporal(X_nodes, y, subj_id, feature_type="Node features")
    all_results_nodes[subj_id] = res_nodes

    # ---------------- Edges ----------------
    X_edges, edge_names, edge_idx, node_indices_edges = get_edge_features(
        {"Edges": Edges_reordered}
    )

    # Visualisations – edges
    plot_top_edges_distributions(X_edges, y, edge_names, subj_id, top_k=12)
    disc_edges = compute_feature_discriminability(X_edges, y)
    plot_edge_discriminability_graph(edge_idx, disc_edges, node_indices_edges, subj_id, top_k=30)
    plot_pca_feature_space(X_edges, y, subj_id, feature_type="Edge features")

    # Classification – edges only
    res_edges = evaluate_models_temporal(X_edges, y, subj_id, feature_type="Edge features")
    all_results_edges[subj_id] = res_edges

    # -------------- Concatenated -----------
    X_concat = np.concatenate([X_nodes, X_edges], axis=1)
    res_concat = evaluate_models_temporal(X_concat, y, subj_id, feature_type="Node+Edge concat")
    all_results_concat[subj_id] = res_concat

#%%

import pandas as pd

# -------------------------------------------------------------
# Helper: flatten metrics into a dataframe
# -------------------------------------------------------------
def results_to_dataframe(all_results, feature_type):
    """
    Convert nested dict:
        {subj_id: {model_name: {metric: value, ...}}}
    into a DataFrame with multi-index.
    """
    rows = []
    for subj_id, model_dict in all_results.items():
        for model_name, metrics in model_dict.items():
            row = {"Subject": subj_id, "Model": model_name, "FeatureType": feature_type}
            row.update(metrics)  # accuracy, kappa, etc.
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Convert everything into stacked dataframes
# -------------------------------------------------------------
df_nodes = results_to_dataframe(all_results_nodes, "Nodes")
df_edges = results_to_dataframe(all_results_edges, "Edges")
df_concat = results_to_dataframe(all_results_concat, "Concat")

df_all = pd.concat([df_nodes, df_edges, df_concat], ignore_index=True)

print("\n============================================================")
print("FULL RESULTS DATAFRAME (all subjects, all models, all metrics)")
print("============================================================")
print(df_all.head())


# -------------------------------------------------------------
# A. Per-subject best model summary
# -------------------------------------------------------------
def best_model_per_subject(df):
    """
    For each subject + feature type,
    pick the model with highest *balanced accuracy* (more robust metric).
    """
    best_rows = []
    grouped = df.groupby(["Subject", "FeatureType"])

    for (subj, ft), subdf in grouped:
        best_idx = subdf["balanced_accuracy"].idxmax()
        best_rows.append(subdf.loc[best_idx])

    return pd.DataFrame(best_rows)

df_best_per_subject = best_model_per_subject(df_all)

print("\n============================================================")
print("BEST MODEL PER SUBJECT (based on balanced accuracy)")
print("============================================================")
print(df_best_per_subject)


# -------------------------------------------------------------
# B. Average metrics across subjects (generalisation indicators)
# -------------------------------------------------------------
df_grouped = df_all.groupby("FeatureType").mean(numeric_only=True)

print("\n============================================================")
print("AVERAGED METRICS ACROSS SUBJECTS (GENERALISATION SCORE)")
print("============================================================")
print(df_grouped[[
    "accuracy", "balanced_accuracy", "kappa",
    "sensitivity", "specificity", "precision", "f1"
]])


# -------------------------------------------------------------
# C. (Optional) Save to CSVs for paper/report
# -------------------------------------------------------------
df_all.to_csv("ALL_RESULTS_FULL.csv", index=False)
df_best_per_subject.to_csv("BEST_MODEL_PER_SUBJECT.csv", index=False)
df_grouped.to_csv("FEATURETYPE_GENERALISATION_SUMMARY.csv")

print("\nSaved CSV files:")
print(" - ALL_RESULTS_FULL.csv")
print(" - BEST_MODEL_PER_SUBJECT.csv")
print(" - FEATURETYPE_GENERALISATION_SUMMARY.csv")
