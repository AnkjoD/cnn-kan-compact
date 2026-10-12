"""Download missing research datasets through the user's own Kaggle CLI account."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from .config import load_config

SOURCES = {
    'rsna': ('competitions', 'rsna-pneumonia-detection-challenge', 'rsna_raw_root'),
    'kermany': ('datasets', 'paultimothymooney/chest-xray-pneumonia', 'kermany_root'),
    'masks': ('datasets', 'nikhilpandey360/chest-xray-masks-and-labels', 'lung_masks_root'),
}


def _has_rsna(root: Path) -> bool:
    label_file = root / 'stage_2_detailed_class_info.csv'
    images = root / 'stage_2_train_images'
    if not label_file.is_file() or not images.is_dir():
        return False
    try:
        frame = pd.read_csv(label_file, usecols=['patientId'])
    except (OSError, ValueError):
        return False
    ids = frame.patientId.dropna().astype(str).unique()
    return len(ids) > 0 and all((images / f'{pid}.dcm').is_file() for pid in ids)


def _has_kermany(root: Path) -> bool:
    bases = [root / 'chest_xray', root / 'chest_xray' / 'chest_xray', root]
    return any(all(any((base / split / label).glob('*'))
                   for split in ('train', 'val', 'test') for label in ('NORMAL', 'PNEUMONIA'))
               for base in bases)


def _has_masks(root: Path) -> bool:
    image_dirs = list(root.glob('**/CXR_png'))
    mask_dirs = list(root.glob('**/masks'))
    return any(any(d.glob('*.png')) for d in image_dirs) and any(any(d.glob('*.png')) for d in mask_dirs)

CHECKS = {'rsna': _has_rsna, 'kermany': _has_kermany, 'masks': _has_masks}


def _extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError(f'Unsafe path in {zip_path.name}: {member.filename}')
        archive.extractall(destination)


def download(source: str, destination: Path, kaggle: str = 'kaggle') -> bool:
    if CHECKS[source](destination):
        print(f'[READY] {source}: {destination}', flush=True)
        return False
    if shutil.which(kaggle) is None:
        raise RuntimeError('Kaggle CLI not installed: pip install kaggle. Log in with kaggle auth login or set your token.')
    category, slug, _ = SOURCES[source]
    destination.mkdir(parents=True, exist_ok=True)
    # Kaggle CLI 2.x takes the competition/dataset slug as a positional value.
    command = [kaggle, category, 'download', slug, '-p', str(destination)]
    print(f'[DOWNLOAD] {source} -> {destination}', flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if source == 'rsna':
            raise RuntimeError('RSNA download failed. Sign in to Kaggle and accept the competition rules on its Data page.') from error
        raise RuntimeError(f'{source} download failed. Check Kaggle authentication and network.') from error
    processed = set()
    while True:
        archives = [p for p in sorted(destination.glob('*.zip')) if p not in processed]
        if not archives:
            break
        for archive in archives:
            print(f'[EXTRACT] {archive.name}', flush=True)
            _extract(archive, destination)
            processed.add(archive)
    if not CHECKS[source](destination):
        raise RuntimeError(f'{source} downloaded, but expected folders/files were not found under {destination}. '
                           'Inspect the archive and paths; do not mark it as complete.')
    print(f'[READY] {source}: {destination}', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=['rsna', 'kermany'],
                        help='default: rsna kermany; masks only when training MobileUNet')
    parser.add_argument('--kaggle-command', default='kaggle', help='Kaggle CLI executable')
    args = parser.parse_args()
    cfg = load_config(args.config)
    for source in args.sources:
        key = SOURCES[source][2]
        if not cfg['paths'].get(key):
            parser.error(f'paths.{key} must be set in {args.config}')
        download(source, Path(cfg['paths'][key]).expanduser().resolve(), args.kaggle_command)
    print('Download check finished. Classifier train still requires the frozen RSNA manifest, '
          'PNG images, and MobileUNet checkpoint (unless --no-crop).')

if __name__ == '__main__':
    main()
