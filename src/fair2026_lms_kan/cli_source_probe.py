"""Label-balanced source probe of classifier features; exploratory only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import LungCropCache, XRayDataset, build_transforms, discover_kermany, load_manifest
from .models.classifier import build_model


@torch.inference_mode()
def features(model, samples, transform, crop, device, batch_size, workers):
    if crop:
        crop.ensure(samples)
    loader = DataLoader(XRayDataset(samples, transform, crop), batch_size=batch_size,
                        num_workers=workers, shuffle=False)
    output=[]
    model.eval()
    for x, _ in tqdm(loader, desc='features', leave=False):
        output.append(model.forward_features(x.to(device)).cpu().numpy())
    return np.concatenate(output)


def probe(feature_vectors, labels, sources, seed=42, test_fraction=0.25):
    if len(np.unique(sources)) != 2 or len(np.unique(labels)) != 2:
        raise ValueError('Two sources and two disease labels required')
    strata = sources * 2 + labels
    counts = np.bincount(strata, minlength=4)
    if counts.min() < 4:
        raise ValueError('Each source/label cell needs at least four samples for the stratified holdout')
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(np.flatnonzero(strata == group), size=counts.min(), replace=False)
                               for group in range(4)])
    train, test = train_test_split(selected, test_size=test_fraction, stratify=strata[selected], random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    model.fit(feature_vectors[train], sources[train])
    scores = model.predict_proba(feature_vectors[test])[:,1]
    return {'auc': float(roc_auc_score(sources[test], scores)), 'n_train': int(len(train)),
            'n_test': int(len(test)), 'n_per_source_label_cell': int(counts.min()), 'seed': int(seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--head', choices=['fastkan','linear','mlp'], help='for legacy checkpoints')
    parser.add_argument('--no-crop', action='store_true', help='for legacy checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='source_probe.json')
    args = parser.parse_args()
    cfg = load_config(args.config); paths = cfg['paths']; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = ckpt.get('metadata', {})
    if args.head and meta.get('head') and args.head != meta['head']:
        parser.error('--head conflicts with checkpoint metadata')
    if args.no_crop and meta.get('use_lung_crop') is True:
        parser.error('--no-crop conflicts with checkpoint metadata')
    model_cfg = dict(cfg['model']); default_head = model_cfg.pop('head')
    model = build_model(head=args.head or meta.get('head') or default_head, **model_cfg).to(device)
    model.load_state_dict(ckpt.get('model_state', ckpt))
    crop = None
    if meta.get('use_lung_crop', not args.no_crop):
        crop = LungCropCache(paths['segmentation_checkpoint'], paths['crop_cache'], device)
    rsna = load_manifest(paths['manifest'], paths.get('rsna_root'))['test']
    external = discover_kermany(paths['kermany_root'])
    _, transform = build_transforms(cfg['data']['image_size'])
    sizes = cfg['training']
    x = np.concatenate([features(model, rsna, transform, crop, device, sizes['batch_size'], sizes['num_workers']),
                        features(model, external, transform, crop, device, sizes['batch_size'], sizes['num_workers'])])
    labels = np.array([y for _,y in rsna+external]); sources = np.array([0]*len(rsna)+[1]*len(external))
    result = probe(x, labels, sources, seed=args.seed)
    result['interpretation'] = 'Source is decodable from frozen features; this does not prove a disease shortcut or its cause.'
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
