"""Validate a frozen RSNA split and report provenance without creating a new split."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(path, root=None, patient_column='patient_id', check_files=False):
    path = Path(path)
    data = pd.read_csv(path, dtype={'path': str, 'split': str})
    required = {'path', 'label', 'split'}
    if not required <= set(data):
        raise ValueError(f'Missing columns: {sorted(required - set(data))}')
    if data[list(required)].isna().any().any() or not data.label.isin([0,1]).all():
        raise ValueError('Null required values or nonbinary labels')
    if not data.split.isin(['train','val','test']).all():
        raise ValueError('Only train/val/test splits are allowed')
    if set(data.split) != {'train','val','test'}:
        raise ValueError('All train/val/test splits must be present')
    if data.path.duplicated().any():
        raise ValueError('Duplicate image paths (including across splits)')
    if patient_column in data:
        if data[patient_column].isna().any():
            raise ValueError('Null patient identifiers')
        leakage = data.groupby(patient_column).split.nunique()
        if (leakage > 1).any():
            raise ValueError(f'{int((leakage>1).sum())} patients appear in multiple splits')
    if check_files:
        missing=[]
        for raw in data.path:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                if root is None:
                    raise ValueError('Relative paths require --data-root for --check-files')
                p = Path(root).expanduser() / p
            if not p.is_file():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing files; first: {missing[0]}')
    counts = data.groupby(['split','label']).size().unstack(fill_value=0).reindex(['train','val','test'], fill_value=0)
    result = {'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n_images': int(len(data)),
              'split_label_counts': {str(split): {str(k): int(v) for k,v in row.items()}
                                     for split,row in counts.iterrows()},
              'patient_column_present': patient_column in data,
              'n_patients': int(data[patient_column].nunique()) if patient_column in data else None,
              'files_checked': bool(check_files)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--patient-column', default='patient_id')
    parser.add_argument('--check-files', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = audit(args.manifest, args.data_root, args.patient_column, args.check_files)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
