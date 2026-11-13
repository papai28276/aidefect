"""Assumption-driven cost of poor quality and threshold sensitivity."""
from __future__ import annotations
import numpy as np
import pandas as pd


def cost_curve(y_true, probability, missed_cost=1000., false_alarm_cost=100., inspection_cost=0., threshold_count=101):
    """Return threshold costs and the cost-minimizing operating point."""
    y = np.asarray(y_true, dtype=int); p = np.asarray(probability, dtype=float)
    thresholds = np.linspace(0, 1, threshold_count)
    rows = []
    for t in thresholds:
        pred = p >= t
        fn = int(((y == 1) & ~pred).sum()); fp = int(((y == 0) & pred).sum())
        inspected = int(pred.sum())
        rows.append({"threshold": t, "missed_defects": fn, "false_alarms": fp,
                     "inspected_parts": inspected,
                     "copq": fn * missed_cost + fp * false_alarm_cost + inspected * inspection_cost})
    return pd.DataFrame(rows)


def compare_scenarios(y_true, predictions: dict, costs: dict, production_volume=10000):
    """Scale held-out scenario costs to a declared production volume."""
    rows = []
    for name, pred in predictions.items():
        if isinstance(pred, pd.DataFrame):
            y, p = pred.actual.to_numpy(), pred.probability.to_numpy()
            if "threshold" in pred:
                threshold = float(pred.threshold.iloc[0])
                mask = p >= threshold
                fn = int(((y == 1) & ~mask).sum()); fp = int(((y == 0) & mask).sum())
                inspected = int(mask.sum())
                value = fn * costs.get("missed_cost", 1000.) + fp * costs.get("false_alarm_cost", 100.) + inspected * costs.get("inspection_cost", 0.)
                scale = production_volume / max(1, len(y))
                total=value*scale; baseline=costs.get("inspection_cost",0.)*production_volume
                rows.append({"scenario": name, "threshold": threshold, "copq": total,
                             "net_savings_vs_manual": baseline-total,
                             "roi": (baseline-total)/max(total,1e-9),
                             "missed_defects": int(fn * scale), "false_alarms": int(fp * scale)})
                continue
        else:
            y, p = np.asarray(y_true), np.asarray(pred)
        curve = cost_curve(y, p, **costs)
        best = curve.loc[curve.copq.idxmin()]
        scale = production_volume / max(1, len(y))
        total=best.copq*scale; baseline=costs.get("inspection_cost",0.)*production_volume
        rows.append({"scenario": name, "threshold": best.threshold, "copq": total,
                     "net_savings_vs_manual": baseline-total,
                     "roi": (baseline-total)/max(total,1e-9),
                     "missed_defects": int(best.missed_defects * scale), "false_alarms": int(best.false_alarms * scale)})
    return pd.DataFrame(rows)


def sensitivity(y, p, missed_costs=(100, 250, 500, 1000, 2500), false_alarm_costs=(10, 25, 50, 100, 250)):
    rows=[]
    for miss in missed_costs:
        for false in false_alarm_costs:
            c=cost_curve(y,p,missed_cost=miss,false_alarm_cost=false)
            rows.append({"missed_defect_cost":miss,"false_alarm_cost":false,"minimum_cost":c.copq.min(),"optimal_threshold":c.loc[c.copq.idxmin(),"threshold"]})
    return pd.DataFrame(rows)
