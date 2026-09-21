"""
Exp 0 — Clean Baseline.
Evaluate on clean (non-injected) data + collect efficiency stats.

Usage: python experiments/exp_clean.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters import get_adapter
from experiments.data_pipeline import load_dataset, split_data
from experiments.utils import (
    evaluate_scores,
    write_csv_result,
    resolve_experiment_datasets,
)

# ── Config ──────────────────────────────────────────────────────────
MODELS = ['S3AD', 'CSLSTMs', 'KANAD', 'DLinear']

AVAILABLE_DATASETS = {
    # 'NAB':   './data/NAB',
    'KPI':   './data/KPI',
    'WSD':   './data/WSD',
    'TODS':  './data/TODS',
    'AIOPS': './data/AIOPS',
}
DATASETS = resolve_experiment_datasets({'NAB': './data/NAB'}, AVAILABLE_DATASETS)

DATASET_HP = {
    'KPI':   {'window': 240, 'seasonal': 48,  'cycle': 48,  'd_model': 128},
    'NAB':   {'window': 768, 'seasonal': 128, 'cycle': 128, 'd_model': 256},
    'WSD':   {'window': 147, 'seasonal': 24,  'cycle': 24,  'd_model': 128},
    'TODS':  {'window': 240, 'seasonal': 48,  'cycle': 48,  'd_model': 128},
    'AIOPS': {'window': 768, 'seasonal': 128, 'cycle': 128, 'd_model': 128},
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'clean')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda:0'

    for ds_name, ds_path in DATASETS.items():
        if not os.path.isdir(ds_path):
            print(f"[SKIP] Dataset not found: {ds_path}")
            continue

        hp = DATASET_HP.get(ds_name, {'window': 240})
        window = hp['window']
        csv_path = os.path.join(OUTPUT_DIR, f'{ds_name}_results.csv')

        print(f"\n{'='*60}\n  Clean Baseline | {ds_name} (window={window})\n{'='*60}")

        values, labels = load_dataset(ds_path)
        (train_v, train_l), (valid_v, valid_l), (test_v, test_l) = split_data(values, labels)

        config = {}
        for model_name in MODELS:
            print(f"  Model: {model_name}")
            try:
                adapter = get_adapter(model_name, window=window, device=device)
                adapter.fit(train_v, train_l, valid_v, valid_l)
                scores, lbls = adapter.predict(test_v, test_l)
                metrics = evaluate_scores(scores, lbls)
                eff = adapter.get_efficiency_stats()
                all_metrics = {**metrics, **eff}

                write_csv_result(csv_path, model_name, ds_name, 'clean', config, all_metrics)
                print(f"    => Best_F1={metrics['Best_F1']:.4f}  AUC={metrics['AUC']:.4f}  "
                      f"params={eff['n_params']:,}  train={eff['train_time_s']:.1f}s")
            except Exception as e:
                print(f"    ERROR: {e}")

        # LaTeX table for this dataset
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            from experiments.utils import generate_latex_table
            results = {}
            for _, row in df.iterrows():
                results[row['model']] = {}
                for c in ['Best_F1', 'Delay_F1', 'AUC', 'AUPRC']:
                    if c in row:
                        results[row['model']][c] = row[c]
            tex = generate_latex_table(results,
                                       caption=f'Clean baseline on {ds_name}',
                                       label=f'tab:clean_{ds_name}')
            tex_path = os.path.join(OUTPUT_DIR, f'{ds_name}_clean.tex')
            with open(tex_path, 'w') as f:
                f.write(tex)
        except Exception as e:
            print(f"    LaTeX: {e}")

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
