"""Fail early with specific data preparation steps before GPU work begins."""
from pathlib import Path

from .cli_audit_manifest import audit
from .data import discover_kermany


def check_data(paths: dict, use_crop: bool = True) -> dict:
    manifest=Path(paths['manifest']).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f'RSNA manifest missing: {manifest}. Use your existing frozen manifest. '
                                'For a NEW split only, run prepare_data.py --sources rsna and then prepare_rsna.py.')
    rsna_root=paths.get('rsna_root')
    result=audit(manifest, rsna_root, check_files=True)
    if not result['patient_column_present']:
        print('[WARNING] No patient_id column: cannot verify patient-level split.',flush=True)
    print(f'[DATA] RSNA manifest: {result["n_images"]} images, sha256={result["manifest_sha256"]}',flush=True)
    if use_crop:
        checkpoint=Path(paths['segmentation_checkpoint']).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'MobileUNet checkpoint missing: {checkpoint}. Restore the original '
                                    'checkpoint, train a NEW one with train_segmentation.py, or use --no-crop.')
    if paths.get('kermany_root'):
        root=Path(paths['kermany_root']).expanduser()
        if not root.exists():
            raise FileNotFoundError(f'Kermany dataset missing: {root}. Run prepare_data.py --sources kermany.')
        samples=discover_kermany(root)
        if {label for _,label in samples}!={0,1}:
            raise ValueError('Kermany requires both NORMAL and PNEUMONIA images')
        print(f'[DATA] Kermany: {len(samples)} images',flush=True)
        if len(samples)<5856:
            print('[WARNING] Kermany image count is below the 5,856 images reported in the paper; '
                  'verify completeness before using these metrics.',flush=True)
    return result
