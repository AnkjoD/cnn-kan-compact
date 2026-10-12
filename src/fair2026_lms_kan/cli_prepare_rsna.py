"""Convert downloaded RSNA DICOMs to PNG and create a NEW patient-level manifest."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .config import load_config
from .cli_prepare_data import _has_rsna
from .cli_audit_manifest import audit


def prepare(raw_root, image_root, manifest, seed=42):
    import pydicom
    raw_root=Path(raw_root).expanduser().resolve()
    image_root=Path(image_root).expanduser().resolve()
    manifest=Path(manifest).expanduser().resolve()
    if not _has_rsna(raw_root):
        raise FileNotFoundError(f'Raw RSNA files missing at {raw_root}; run prepare_data.py --sources rsna')
    if manifest.exists():
        raise FileExistsError(f'{manifest} exists; never overwrite a frozen manifest')
    classes=pd.read_csv(raw_root/'stage_2_detailed_class_info.csv')
    if not {'patientId','class'}<=set(classes):
        raise ValueError('RSNA class file missing patientId/class columns')
    classes=classes.drop_duplicates('patientId')
    classes=classes[classes['class'].isin(['Normal','Lung Opacity'])]
    labels={'Normal':0,'Lung Opacity':1}
    items={0:[],1:[]}
    image_root.mkdir(parents=True,exist_ok=True)
    for pid, class_name in tqdm(zip(classes['patientId'], classes['class']),total=len(classes),desc='RSNA DICOM -> PNG'):
        pid=str(pid); label=labels[class_name]
        dicom=raw_root/'stage_2_train_images'/f'{pid}.dcm'
        if not dicom.is_file():
            raise FileNotFoundError(f'Missing selected DICOM {dicom}')
        destination=image_root/('Normal' if label==0 else 'Pneumonia')/f'{pid}.png'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_file():
            pixels=pydicom.dcmread(dicom).pixel_array.astype(np.float32)
            pixels-=pixels.min()
            maximum=pixels.max()
            if maximum>0: pixels=pixels/maximum*255.0
            temporary=destination.with_suffix('.tmp.png')
            Image.fromarray(pixels.astype(np.uint8)).save(temporary)
            temporary.replace(destination)
        items[label].append((str(destination.relative_to(image_root)), label, pid))
    rng=random.Random(seed); splits={'train':[],'val':[],'test':[]}
    for label in (0,1):
        group=sorted(items[label]);rng.shuffle(group)
        n=int(len(group)*0.15)
        splits['val']+=group[:n];splits['test']+=group[n:2*n];splits['train']+=group[2*n:]
    for group in splits.values(): rng.shuffle(group)
    manifest.parent.mkdir(parents=True,exist_ok=True)
    rows=[{'path':path,'label':label,'split':split,'patient_id':pid}
          for split,group in splits.items() for path,label,pid in group]
    temporary=manifest.with_suffix('.tmp.csv')
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(manifest)
    report=audit(manifest,image_root,check_files=True)
    report['origin']='new reconstruction from raw RSNA; not the archived paper split'
    report['raw_class_csv_sha256']=hashlib.sha256((raw_root/'stage_2_detailed_class_info.csv').read_bytes()).hexdigest()
    manifest.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    cfg=load_config(args.config)['paths']
    prepare(cfg['rsna_raw_root'],cfg['rsna_root'],cfg['manifest'],args.seed)

if __name__=='__main__':main()
