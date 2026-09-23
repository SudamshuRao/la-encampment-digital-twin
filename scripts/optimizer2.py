"""
Greedy optimizer for SNN-2. Unlike SNN-1's optimizer, a slider move here
can affect multiple interaction subnets (any interaction pair involving
the changed feature), not just one independent main-effect subnet. Rather
than track which interactions are affected and patch deltas by hand, each
candidate move just calls model.predict() fresh -- with ~65 tiny RBF
subnets this is still cheap (well under a millisecond per call), and it
guarantees correctness without duplicating the interaction-composition
logic that already lives in SNN2Model.
"""
import numpy as np


def greedy_optimize_snn2(model, current_base_features: dict, candidate_features: list,
                          bounds: dict, k: int = 5, step_size: float = 1.0,
                          allow_decrease: bool = False):
    values = dict(current_base_features)
    baseline_risk, _ = model.predict(values)
    risk_trajectory = [baseline_risk]
    steps = []

    for _ in range(k):
        best_f, best_val, best_risk, best_dir = None, None, risk_trajectory[-1], None

        for f in candidate_features:
            cur_val = values.get(f, 0.0)
            lo, hi = bounds.get(f, (-np.inf, np.inf))

            trials = [("up", cur_val + step_size, hi, "hi")]
            if allow_decrease:
                trials.append(("down", cur_val - step_size, lo, "lo"))

            for direction, new_val, limit, kind in trials:
                if kind == "hi" and new_val > limit:
                    continue
                if kind == "lo" and new_val < limit:
                    continue
                trial = dict(values)
                trial[f] = new_val
                risk, _ = model.predict(trial)
                if risk < best_risk:
                    best_risk, best_f, best_val, best_dir = risk, f, new_val, direction

        if best_f is None:
            break

        old_val = values[best_f]
        values[best_f] = best_val
        steps.append({
            "feature": best_f,
            "direction": best_dir,
            "from_value": old_val,
            "to_value": best_val,
            "risk_before": risk_trajectory[-1],
            "risk_after": best_risk,
            "risk_delta": best_risk - risk_trajectory[-1],
        })
        risk_trajectory.append(best_risk)

    return {
        "baseline_risk": baseline_risk,
        "optimized_risk": risk_trajectory[-1],
        "optimized_features": values,
        "steps": steps,
        "risk_trajectory": risk_trajectory,
    }


def is_actionable_step_l2(feature: str, direction: str, feature_groups: dict, actionable_groups: set) -> bool:
    return feature_groups.get(feature) in actionable_groups and direction == "up"


def actionable_subset_risk_l2(model, baseline_features: dict, steps: list, feature_groups: dict, actionable_groups: set):
    subset_features = dict(baseline_features)
    applied = []
    for s in steps:
        if is_actionable_step_l2(s["feature"], s["direction"], feature_groups, actionable_groups):
            subset_features[s["feature"]] = s["to_value"]
            applied.append(s)
    risk, _ = model.predict(subset_features)
    return risk, applied
