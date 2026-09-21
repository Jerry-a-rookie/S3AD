"""
Shared utilities for experiments: evaluation metrics, result formatting,
dataset selection, and LaTeX table generation.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_f1_score import best_f1, best_f1_without_pointadjust, delay_f1


def evaluate_scores(scores: np.ndarray, labels: np.ndarray,
                    delay_k: int = 3) -> Dict[str, float]:
    """Compute evaluation metrics from anomaly scores and binary labels."""
    if len(np.unique(labels)) < 2:
        return {k: np.nan for k in [
            'AUC', 'AUPRC', 'Best_F1', 'Best_Precision', 'Best_Recall',
            'Best_Threshold', 'Delay_F1', 'Delay_Precision', 'Delay_Recall',
            'Best_F1_noPA', 'Precision_noPA', 'Recall_noPA',
        ]}

    auc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    bf1, bp, br, _, thres = best_f1(scores, labels)
    df1, dp, dr, _ = delay_f1(scores, labels, k=delay_k)
    bf1n, bpn, brn, _ = best_f1_without_pointadjust(scores, labels)

    return {
        'AUC': auc, 'AUPRC': auprc,
        'Best_F1': bf1, 'Best_Precision': bp,
        'Best_Recall': br, 'Best_Threshold': thres,
        'Delay_F1': df1, 'Delay_Precision': dp, 'Delay_Recall': dr,
        'Best_F1_noPA': bf1n, 'Precision_noPA': bpn, 'Recall_noPA': brn,
    }


def evaluate_multiple_runs(all_scores: List[np.ndarray],
                           all_labels: List[np.ndarray],
                           delay_k: int = 3) -> Dict[str, Tuple[float, float]]:
    """Evaluate multiple random-seed runs. Returns (mean, std) per metric."""
    results_per_run = [
        evaluate_scores(scores, labels, delay_k)
        for scores, labels in zip(all_scores, all_labels)
    ]
    metrics = list(results_per_run[0].keys())
    aggregated = {}
    for metric in metrics:
        vals = [r[metric] for r in results_per_run if not np.isnan(r[metric])]
        aggregated[metric] = (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)
    return aggregated


def write_csv_result(filepath: str, model_name: str, dataset: str,
                     exp_type: str, config: Dict, metrics: Dict):
    """Append a result row to a CSV file. Creates the header for new files."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    new_file = not os.path.exists(filepath)
    row = {
        'model': model_name,
        'dataset': dataset,
        'exp_type': exp_type,
        **config,
        **metrics,
    }
    with open(filepath, 'a', newline='') as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def resolve_experiment_datasets(default_datasets: Dict[str, str],
                                available_datasets: Dict[str, str]) -> Dict[str, str]:
    """Resolve experiment datasets from the EXPERIMENT_DATASETS environment variable.

    Examples:
        EXPERIMENT_DATASETS=TODS
        EXPERIMENT_DATASETS=KPI,TODS
        EXPERIMENT_DATASETS=all
    """
    selected = os.environ.get('EXPERIMENT_DATASETS', '').strip()
    if not selected:
        return default_datasets

    if selected.lower() == 'all':
        return available_datasets

    result = {}
    for name in selected.split(','):
        key = name.strip()
        if not key:
            continue
        if key not in available_datasets:
            raise ValueError(
                f"Unknown dataset '{key}'. Available: {', '.join(available_datasets)}"
            )
        result[key] = available_datasets[key]
    return result or default_datasets


def generate_latex_table(results: Dict[str, Dict[str, float]],
                         caption: str = "Results",
                         label: str = "tab:results",
                         metrics: List[str] = None) -> str:
    """Generate a LaTeX table from per-model results."""
    if metrics is None:
        metrics = ['Best_F1', 'Delay_F1', 'AUC', 'AUPRC']

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{l' + 'c' * len(metrics) + '}',
        r'\toprule',
        'Model & ' + ' & '.join(m.replace('_', r'\_') for m in metrics) + r' \\',
        r'\midrule',
    ]

    for model, mdict in results.items():
        vals = [f"{mdict.get(m, np.nan):.4f}" for m in metrics]
        lines.append(f'{model} & ' + ' & '.join(vals) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    return '\n'.join(lines)


def dict_to_csv_row(model: str, dataset: str, exp_type: str,
                    config: Dict, metrics: Dict) -> Dict:
    """Combine metadata and metrics into a single flat dict for CSV output."""
    return {
        'model': model,
        'dataset': dataset,
        'exp_type': exp_type,
        **config,
        **metrics,
    }
