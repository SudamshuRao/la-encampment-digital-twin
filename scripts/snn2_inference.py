"""
Standalone inference for the trained SNN-2 model: a frozen additive
main-effect layer (one RBFSubNet per base feature) plus a true 2D
joint-RBF interaction layer (one BivariateRBFSubNet per selected pair,
taking both raw feature values as separate inputs -- not a collapsed
product column). Loads the JSON artifact written by scripts/train_snn2.py.
"""
import json
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class SNN2Model:
    def __init__(self, artifact_path: str):
        with open(artifact_path) as f:
            art = json.load(f)
        self.base_features = art["base_features"]
        self.interaction_pairs = art["interaction_pairs"]  # list of [a, b] names
        self.interaction_names = [f"{a}__x__{b}" for a, b in self.interaction_pairs]

        self.scaler_base_mean = np.array(art["scaler_base_mean"])
        self.scaler_base_scale = np.array(art["scaler_base_scale"])
        self.threshold = art["threshold"]

        self.main_nets = []
        for nd in art["main_nets"]:
            self.main_nets.append({
                "centers": np.array(nd["centers"]),
                "width": nd["width"],
                "coef": np.array(nd["coef"]),
            })

        self.pair_nets = []
        for nd in art["pair_nets"]:
            self.pair_nets.append({
                "cj": np.array(nd["cj"]), "ck": np.array(nd["ck"]),
                "width": nd["width"], "coef": np.array(nd["coef"]),
            })

        self._base_idx = {f: i for i, f in enumerate(self.base_features)}
        self._interactions_by_base_feature = {f: [] for f in self.base_features}
        for k, (a, b) in enumerate(self.interaction_pairs):
            self._interactions_by_base_feature[a].append(k)
            self._interactions_by_base_feature[b].append(k)

    def _main_forward_one(self, net, x_std_scalar: float) -> float:
        d = x_std_scalar - net["centers"]
        rbf = np.exp(-net["width"] * d ** 2)
        phi = np.concatenate([[1.0], rbf])
        return float(phi @ net["coef"])

    def _pair_forward_one(self, net, xj_std: float, xk_std: float) -> float:
        dj = xj_std - net["cj"]
        dk = xk_std - net["ck"]
        basis = np.exp(-net["width"] * dj ** 2) * np.exp(-net["width"] * dk ** 2)
        phi = np.concatenate([[1.0], basis])
        return float(phi @ net["coef"])

    def predict(self, raw_base: dict):
        """raw_base: {base_feature_name: raw_value}. Returns (risk, contribs)
        where contribs has one entry per base feature (main effect) and
        one per selected interaction pair (named 'a__x__b')."""
        base_vec = np.array([raw_base.get(f, 0.0) for f in self.base_features])
        base_std = (base_vec - self.scaler_base_mean) / self.scaler_base_scale

        contribs = {}
        total = 0.0
        for j, f in enumerate(self.base_features):
            sj = self._main_forward_one(self.main_nets[j], base_std[j])
            contribs[f] = sj
            total += sj

        for idx, (a, b) in enumerate(self.interaction_pairs):
            ja, jb = self._base_idx[a], self._base_idx[b]
            ijk = self._pair_forward_one(self.pair_nets[idx], base_std[ja], base_std[jb])
            contribs[self.interaction_names[idx]] = ijk
            total += ijk

        risk = float(_sigmoid(np.array([total]))[0])
        return risk, contribs

    def predict_batch(self, raw_base_df) -> np.ndarray:
        n = len(raw_base_df)
        base_mat = raw_base_df[self.base_features].fillna(0).to_numpy(dtype=float)
        base_std = (base_mat - self.scaler_base_mean) / self.scaler_base_scale

        total = np.zeros(n)
        for j, net in enumerate(self.main_nets):
            d = base_std[:, j][:, None] - net["centers"][None, :]
            rbf = np.exp(-net["width"] * d ** 2)
            phi = np.concatenate([np.ones((n, 1)), rbf], axis=1)
            total += phi @ net["coef"]

        for idx, (a, b) in enumerate(self.interaction_pairs):
            ja, jb = self._base_idx[a], self._base_idx[b]
            net = self.pair_nets[idx]
            dj = base_std[:, ja][:, None] - net["cj"][None, :]
            dk = base_std[:, jb][:, None] - net["ck"][None, :]
            basis = np.exp(-net["width"] * dj ** 2) * np.exp(-net["width"] * dk ** 2)
            phi = np.concatenate([np.ones((n, 1)), basis], axis=1)
            total += phi @ net["coef"]

        return _sigmoid(total)

    def batch_top_interactions(self, raw_base_df):
        """Vectorized version of top_interaction_for_hex for every row --
        used to precompute a 'headline interaction' column per hex for the map."""
        n = len(raw_base_df)
        base_mat = raw_base_df[self.base_features].fillna(0).to_numpy(dtype=float)
        base_std = (base_mat - self.scaler_base_mean) / self.scaler_base_scale

        if not self.interaction_pairs:
            return [None] * n, np.zeros(n)

        all_contribs = np.zeros((n, len(self.interaction_pairs)))
        for idx, (a, b) in enumerate(self.interaction_pairs):
            ja, jb = self._base_idx[a], self._base_idx[b]
            net = self.pair_nets[idx]
            dj = base_std[:, ja][:, None] - net["cj"][None, :]
            dk = base_std[:, jb][:, None] - net["ck"][None, :]
            basis = np.exp(-net["width"] * dj ** 2) * np.exp(-net["width"] * dk ** 2)
            phi = np.concatenate([np.ones((n, 1)), basis], axis=1)
            all_contribs[:, idx] = phi @ net["coef"]

        best_idx = np.argmax(np.abs(all_contribs), axis=1)
        best_val = all_contribs[np.arange(n), best_idx]
        best_pairs = [tuple(self.interaction_pairs[i]) for i in best_idx]
        return best_pairs, best_val
