"""Run the controlled classifier experiments in priority order, with resume."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .cli_prepare_data import SOURCES, download
from .cli_prepare_rsna import prepare as prepare_rsna
from .config import load_config
from .preflight import check_data

ROOT = Path(__file__).resolve().parents[2]


def jobs(heads, seeds, include_no_crop=True):
    plan=[]
    for seed in seeds:
        plan.extend((head, seed, True) for head in heads)
        if include_no_crop and 'fastkan' in heads:
            plan.append(('fastkan', seed, False))
    return plan


def _invoke(command, seconds=None):
    print('[RUN]', ' '.join(map(str, command)), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True, timeout=seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('Time budget reached during a job. The last completed epoch is saved; rerun the same command to resume.') from error


def _check_checkpoint(path, head, seed, crop, manifest_hash, model_config):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    meta = checkpoint.get('metadata', {})
    expected = {'head':head, 'seed':seed, 'use_lung_crop':crop,
                'manifest_sha256':manifest_hash,'model':model_config}
    if any(meta.get(key) != value for key,value in expected.items()):
        raise ValueError(f'Existing run has incompatible checkpoint metadata: {path}. Use a different output directory.')


def run(args):
    config = load_config(args.config)
    paths=config['paths']
    seeds=args.seeds if args.seeds else config['training']['seeds']
    if len(seeds)!=len(set(seeds)) or len(args.heads)!=len(set(args.heads)):
        raise ValueError('Repeated head or seed in requested jobs')
    plan=jobs(args.heads,seeds,not args.skip_no_crop)
    if args.max_jobs is not None:
        if args.max_jobs<1: raise ValueError('--max-jobs must be positive')
    if args.dry_run:
        for head,seed,crop in (plan[:args.max_jobs] if args.max_jobs else plan):
            print(f'[PLAN] {head}_{"crop" if crop else "no_crop"}_seed{seed}')
        return
    if args.download_missing:
        # The raw competition download is only necessary for a new conversion.
        if not Path(paths['manifest']).exists() and args.rebuild_rsna:
            download('rsna',Path(paths['rsna_raw_root']).expanduser().resolve())
        if paths.get('kermany_root'):
            download('kermany',Path(paths['kermany_root']).expanduser().resolve())
        if args.segmentation_manifest and not Path(paths['segmentation_checkpoint']).exists():
            download('masks',Path(paths['lung_masks_root']).expanduser().resolve())
    if args.rebuild_rsna and not Path(paths['manifest']).exists():
        prepare_rsna(paths['rsna_raw_root'],paths['rsna_root'],paths['manifest'],args.rebuild_seed)
    if args.segmentation_manifest and not Path(paths['segmentation_checkpoint']).exists():
        _invoke([sys.executable,str(ROOT/'train_segmentation.py'),str(Path(args.segmentation_manifest).resolve()),
                 '--data-root',args.segmentation_root or paths['lung_masks_root'],
                 '--output',paths['segmentation_checkpoint']], None)
    report=check_data(paths,use_crop=any(crop for _,_,crop in plan))
    model_cfg={k:v for k,v in config['model'].items() if k!='head'}
    deadline=time.monotonic()+args.max_hours*3600 if args.max_hours else None
    new_jobs=0
    for head,seed,crop in plan:
        name=f'{head}_{"crop" if crop else "no_crop"}_seed{seed}'
        run_dir=Path(paths['output_dir'])/name
        checkpoint=run_dir/'checkpoint.pt';last=run_dir/'last.pt'
        if checkpoint.exists():
            _check_checkpoint(checkpoint,head,seed,crop,report['manifest_sha256'],model_cfg)
        completed=checkpoint.exists() and (run_dir/'metrics.json').exists() and (run_dir/'predictions.csv').exists()
        if completed:
            print(f'[SKIP] {name}: checkpoint and evaluation already exist',flush=True)
        else:
            if args.max_jobs is not None and new_jobs>=args.max_jobs:
                print('[STOP] Job limit reached; rerun the same command to continue',flush=True)
                return
            if deadline and time.monotonic()>=deadline:
                print(f'[STOP] Time budget before {name}; rerun pipeline to continue',flush=True)
                return
            if checkpoint.exists() and not last.exists():
                raise RuntimeError(f'{name} has a checkpoint but no last.pt or completed evaluation. '
                                   f'Run evaluate.py {checkpoint} manually; do not overwrite training.')
            command=[sys.executable,str(ROOT/'train.py'),'--config',str(Path(args.config).resolve()),
                     '--head',head,'--seed',str(seed)]
            if not crop: command.append('--no-crop')
            if last.exists(): command.append('--resume')
            try:
                _invoke(command, max(deadline-time.monotonic(),0.01) if deadline else None)
                new_jobs+=1
            except RuntimeError as error:
                if 'Time budget reached' in str(error):
                    print(f'[STOP] {name}: rerun pipeline to resume from last completed epoch',flush=True)
                    return
                raise
        if not args.skip_analysis:
            prediction=run_dir/'predictions.csv'
            if prediction.exists() and not (run_dir/'analysis'/'analysis.json').exists():
                _invoke([sys.executable,str(ROOT/'analyze_predictions.py'),str(prediction),
                         '--output-dir',str(run_dir/'analysis')])
    if not args.skip_analysis and paths.get('kermany_root'):
        selected={(head,seed,crop) for head,seed,crop in plan}
        for seed in seeds:
            fast=Path(paths['output_dir'])/f'fastkan_crop_seed{seed}'/'predictions.csv'
            for head in ('linear','mlp'):
                other=Path(paths['output_dir'])/f'{head}_crop_seed{seed}'/'predictions.csv'
                if fast.exists() and other.exists() and (head,seed,True) in selected and ('fastkan',seed,True) in selected:
                    output=Path(paths['output_dir'])/f'{head}_vs_fastkan_seed{seed}.json'
                    _invoke([sys.executable,str(ROOT/'compare_heads.py'),str(other),str(fast),
                             '--dataset','Kermany','--output',str(output)])
    print('[DONE] Selected jobs and available analyses completed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--heads',nargs='+',choices=['fastkan','linear','mlp'],default=['fastkan','linear','mlp'])
    parser.add_argument('--seeds',nargs='+',type=int,help='default: training.seeds in YAML')
    parser.add_argument('--skip-no-crop',action='store_true',help='omit the three FastKAN no-crop jobs')
    parser.add_argument('--download-missing',action='store_true',help='download missing Kaggle data using your account')
    parser.add_argument('--rebuild-rsna',action='store_true',help='create a NEW split only if manifest is absent')
    parser.add_argument('--rebuild-seed',type=int,default=42)
    parser.add_argument('--segmentation-manifest',help='explicit image/mask train/val CSV when checkpoint is absent')
    parser.add_argument('--segmentation-root',help='base directory for relative image/mask paths')
    parser.add_argument('--max-hours',type=float,help='soft deadline; interrupt current job, resume later')
    parser.add_argument('--max-jobs',type=int,help='run only the first N jobs from the priority list')
    parser.add_argument('--skip-analysis',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='show job plan without reading or writing data')
    args=parser.parse_args()
    if args.max_hours is not None and args.max_hours<=0: parser.error('--max-hours must be positive')
    run(args)

if __name__=='__main__':main()
