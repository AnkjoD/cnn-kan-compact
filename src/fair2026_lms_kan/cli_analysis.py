"""Analyze saved per-image predictions without retraining or touching image data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from .metrics import binary_metrics


def analyze(frame, bins=10):
    output = {}
    for name, group in frame.groupby('dataset', sort=True):
        y = group.label.to_numpy(dtype=int)
        p = group.probability.to_numpy(dtype=float)
        if not set(np.unique(y)) == {0, 1}:
            raise ValueError(f'{name}: both classes are required')
        counts, edges = np.histogram(p, bins=np.linspace(0, 1, bins + 1))
        indices = np.minimum(np.searchsorted(edges, p, side='right') - 1, bins - 1)
        ece = 0.0
        calibration = []
        for i, count in enumerate(counts):
            if count:
                selected = indices == i
                confidence = float(p[selected].mean())
                frequency = float(y[selected].mean())
                ece += count / len(p) * abs(confidence - frequency)
                calibration.append({'lower': float(edges[i]), 'upper': float(edges[i+1]),
                                    'n': int(count), 'mean_probability': confidence, 'positive_rate': frequency})
        precision, recall, _ = precision_recall_curve(y, p)
        output[name] = {**binary_metrics(y, p), 'average_precision': float(average_precision_score(y, p)),
                        'brier_score': float(brier_score_loss(y, p)), 'ece': float(ece),
                        'calibration_bins': calibration,
                        'precision_recall': [{'recall': float(r), 'precision': float(q)}
                                             for r, q in zip(recall, precision)]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='CSV with dataset,sample_id,label,probability')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--bins', type=int, default=10)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error('--bins must be at least 2')
    frame = pd.read_csv(args.predictions, dtype={'sample_id': str})
    required = {'dataset', 'sample_id', 'label', 'probability'}
    if not required <= set(frame):
        parser.error(f'missing columns: {sorted(required - set(frame))}')
    if frame.duplicated(['dataset', 'sample_id']).any():
        parser.error('duplicate dataset/sample_id pairs')
    if frame[list(required)].isna().any().any() or not frame.label.isin([0, 1]).all():
        parser.error('null values or invalid binary labels')
    if not np.isfinite(frame.probability.to_numpy(dtype=float)).all() or not frame.probability.between(0, 1).all():
        parser.error('probabilities must be finite in [0,1]')
    result = analyze(frame, args.bins)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.predictions).parent / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: {m: v for m, v in values.items() if m not in ('calibration_bins', 'precision_recall')}
                      for k, values in result.items()}, indent=2))
    print(f'Full curves and calibration bins: {output_dir / "analysis.json"}')

if __name__ == '__main__':
    main()
