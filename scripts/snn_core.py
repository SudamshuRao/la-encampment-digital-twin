"""
Shared additive-RBF-subnetwork training machinery, used internally by
train_snn2.py to fit both the base-feature layer and the interaction-term
layer (same architecture applied to different column sets -- see
train_snn2.py's module docstring). Not exposed as a standalone model; the
app only ever loads the finished SNN-2 artifact.
"""
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RBFSubNet:
    """Fixed radial-basis feature map + ridge-regressed linear readout for
    a single input column. n_rbf centers spread across the standardized
    range with a shared bandwidth tuned to the center spacing; the
    readout is solved in closed form each fit() call (no autodiff/torch
    dependency needed)."""

    def __init__(self, n_rbf: int = 10, ridge: float = 1e-2):
        self.n_rbf = n_rbf
        self.ridge = ridge
        self.centers = np.linspace(-2.5, 2.5, n_rbf)
        spacing = self.centers[1] - self.centers[0]
        self.width = 1.0 / (2 * (spacing * 0.9) ** 2)
        self.coef = np.zeros(n_rbf + 1)  # [bias, w_1..w_n_rbf]

    def _phi(self, x):
        d = x[:, None] - self.centers[None, :]
        rbf = np.exp(-self.width * d ** 2)
        return np.concatenate([np.ones((len(x), 1)), rbf], axis=1)

    def fit(self, x, target):
        Phi = self._phi(x)
        A = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        b = Phi.T @ target
        self.coef = np.linalg.solve(A, b)
        return self

    def forward(self, x):
        return self._phi(x) @ self.coef

    def get_curve(self, x_range=(-3, 3), n_pts=200):
        xs = np.linspace(*x_range, n_pts)
        return xs, self.forward(xs)

    def state_dict(self):
        return {"centers": self.centers.tolist(), "width": float(self.width), "coef": self.coef.tolist()}


def train_additive_snn(X_np, soft_targets, hard_labels, n_rbf=10, n_rounds=6, verbose=True, **_ignored):
    """Fractional-knowledge-distillation backfitting: round 1 fits each
    column sequentially against the running residual; subsequent rounds
    refit each column against the target minus every other column's
    current output, until convergence."""
    n, n_feat = X_np.shape
    ts0 = np.asarray(soft_targets, dtype=float)
    nets = [RBFSubNet(n_rbf) for _ in range(n_feat)]
    fo = [np.zeros(n) for _ in range(n_feat)]

    residual = ts0.copy()
    for j, net in enumerate(nets):
        net.fit(X_np[:, j], residual)
        fo[j] = net.forward(X_np[:, j])
        residual = residual - fo[j]

    for rnd in range(1, n_rounds):
        for j, net in enumerate(nets):
            others = sum(fo[k] for k in range(n_feat) if k != j)
            target_j = ts0 - others
            net.fit(X_np[:, j], target_j)
            fo[j] = net.forward(X_np[:, j])
        if verbose:
            auc = roc_auc_score(hard_labels, _sigmoid(np.stack(fo, 1).sum(1)))
            print(f"    round {rnd+1}/{n_rounds}  AUC={auc:.4f}")

    contribs = np.stack(fo, 1)
    probs = _sigmoid(contribs.sum(1))
    return nets, contribs, probs


class BivariateRBFSubNet:
    """True 2D joint-RBF interaction subnetwork, matching the reference
    notebook's SubNet2: an n_rbf2 x n_rbf2 grid of basis points over the
    two standardized input dimensions (xj, xk), each basis a product of
    two Gaussians. Readout is closed-form ridge regression (same approach
    as RBFSubNet, extended to the 2D basis expansion) rather than the
    reference's gradient descent -- no torch dependency, same architecture."""

    def __init__(self, n_rbf2: int = 6, ridge: float = 1e-2):
        self.n_rbf2 = n_rbf2
        self.ridge = ridge
        pts = np.linspace(-2.5, 2.5, n_rbf2)
        cj, ck = np.meshgrid(pts, pts, indexing="ij")
        self.cj = cj.reshape(-1)
        self.ck = ck.reshape(-1)
        spacing = pts[1] - pts[0]
        self.width = 1.0 / (2 * (spacing * 0.9) ** 2)
        self.coef = np.zeros(n_rbf2 * n_rbf2 + 1)

    def _phi(self, xj, xk):
        dj = xj[:, None] - self.cj[None, :]
        dk = xk[:, None] - self.ck[None, :]
        basis = np.exp(-self.width * dj ** 2) * np.exp(-self.width * dk ** 2)
        return np.concatenate([np.ones((len(xj), 1)), basis], axis=1)

    def fit(self, xj, xk, target):
        Phi = self._phi(xj, xk)
        A = Phi.T @ Phi + self.ridge * np.eye(Phi.shape[1])
        b = Phi.T @ target
        self.coef = np.linalg.solve(A, b)
        return self

    def forward(self, xj, xk):
        return self._phi(xj, xk) @ self.coef

    def state_dict(self):
        return {"cj": self.cj.tolist(), "ck": self.ck.tolist(),
                "width": float(self.width), "coef": self.coef.tolist()}


def train_interaction_snn(X_np, soft_targets, hard_labels, base_contribs_sum,
                           top_pairs, n_rbf2=6, n_rounds=3, verbose=True):
    """Freezes the (already-trained) SNN-1 main-effect layer and fits ONLY
    the interaction subnets on the leftover residual, matching the
    reference's train_snn2: base = snn1_contribs.sum(1) stays fixed
    throughout; only pair_nets are trained round over round."""
    n = X_np.shape[0]
    ts0 = np.asarray(soft_targets, dtype=float)
    base = np.asarray(base_contribs_sum, dtype=float)

    pair_nets = {jk: BivariateRBFSubNet(n_rbf2) for jk in top_pairs}
    pair_fo = {jk: np.zeros(n) for jk in top_pairs}

    residual = ts0 - base
    for jk in top_pairs:
        j, k = jk
        net = pair_nets[jk]
        net.fit(X_np[:, j], X_np[:, k], residual)
        pair_fo[jk] = net.forward(X_np[:, j], X_np[:, k])
        residual = residual - pair_fo[jk]

    for rnd in range(1, n_rounds):
        for jk in top_pairs:
            j, k = jk
            others = sum(pair_fo[p] for p in top_pairs if p != jk)
            target = ts0 - base - others
            pair_nets[jk].fit(X_np[:, j], X_np[:, k], target)
            pair_fo[jk] = pair_nets[jk].forward(X_np[:, j], X_np[:, k])
        if verbose:
            total = base + sum(pair_fo[p] for p in top_pairs)
            auc = roc_auc_score(hard_labels, _sigmoid(total))
            print(f"    interaction round {rnd+1}/{n_rounds}  AUC={auc:.4f}")

    pair_contribs = np.column_stack([pair_fo[jk] for jk in top_pairs])
    total = base + pair_contribs.sum(1)
    probs = _sigmoid(total)
    return pair_nets, pair_contribs, probs


def select_top_pairs(X_np, hard_labels, n_feat, top_k=20):
    """Matches the reference's select_top_pairs exactly: build every
    pairwise product column, fit a shallow RF directly on those products
    against the hard labels (not soft labels, not combined with base
    features), rank by importance."""
    from sklearn.ensemble import RandomForestClassifier
    import itertools
    pairs = list(itertools.combinations(range(n_feat), 2))
    products = np.column_stack([X_np[:, j] * X_np[:, k] for j, k in pairs])
    rf_screen = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
    rf_screen.fit(products, hard_labels)
    ranked = sorted(zip(pairs, rf_screen.feature_importances_), key=lambda x: -x[1])
    return [p for p, _ in ranked[:top_k]]
