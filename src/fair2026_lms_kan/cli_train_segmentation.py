"""Train a NEW MobileUNet from an explicit image/mask train/val manifest."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from .data import IMAGENET_MEAN, IMAGENET_STD
from .models.mobileunet import build_mobileunet
from .training import seed_everything


class SegmentationPairs(Dataset):
    def __init__(self, frame, root=None, train=False, size=256):
        self.rows = frame.to_dict('records'); self.root = Path(root) if root else None
        self.train=train; self.size=size
        self.normalize=transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row=self.rows[index]
        def resolve(raw):
            p=Path(raw).expanduser()
            return p if p.is_absolute() else self.root / p
        image=Image.open(resolve(row['image'])).convert('RGB').resize((self.size,self.size),Image.Resampling.BICUBIC)
        mask=Image.open(resolve(row['mask'])).convert('L').resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.train:
            if random.random()<0.5:
                image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT); mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle=random.uniform(-10,10)
            image=image.rotate(angle,resample=Image.Resampling.BICUBIC)
            mask=mask.rotate(angle,resample=Image.Resampling.NEAREST)
        target=torch.from_numpy((np.asarray(mask)>127).astype(np.float32).copy()).unsqueeze(0)
        return self.normalize(image), target


def dice(logits, target):
    pred=(torch.sigmoid(logits)>0.5).float()
    intersection=(pred*target).sum((1,2,3)); total=pred.sum((1,2,3))+target.sum((1,2,3))
    return (2*intersection+1e-6)/(total+1e-6)


def loss_fn(logits, target):
    prob=torch.sigmoid(logits)
    intersection=(prob*target).sum((1,2,3)); total=prob.sum((1,2,3))+target.sum((1,2,3))
    return nn.functional.binary_cross_entropy_with_logits(logits,target)+1-((2*intersection+1)/(total+1)).mean()


def validate_manifest(frame, root):
    required={'image','mask','split'}
    if not required<=set(frame):
        raise ValueError(f'Missing columns: {sorted(required-set(frame))}')
    if frame[list(required)].isna().any().any() or set(frame.split)!={'train','val'}:
        raise ValueError('Nonempty train and val splits with complete rows required')
    if frame.image.duplicated().any() or frame['mask'].duplicated().any():
        raise ValueError('Duplicate image or mask across splits')
    if 'patient_id' in frame and (frame.patient_id.isna().any() or (frame.groupby('patient_id').split.nunique()>1).any()):
        raise ValueError('Patient missing or shared between train and val')
    for column in ('image','mask'):
        for raw in frame[column]:
            p=Path(raw).expanduser()
            if not p.is_absolute():
                if root is None: raise ValueError('Relative paths need --data-root')
                p=Path(root)/p
            if not p.is_file(): raise FileNotFoundError(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='CSV image,mask,split[,patient_id]; train and val specified in advance')
    parser.add_argument('--data-root')
    parser.add_argument('--output', default='runs/mobileunet_new.pt')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--num-workers',type=int,default=0)
    args=parser.parse_args()
    manifest=Path(args.manifest); frame=pd.read_csv(manifest,dtype={'image':str,'mask':str,'split':str})
    validate_manifest(frame,args.data_root)
    out=Path(args.output)
    if out.exists(): parser.error(f'{out} already exists; choose a new --output')
    out.parent.mkdir(parents=True,exist_ok=True)
    seed_everything(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders={split:DataLoader(SegmentationPairs(frame[frame.split==split],args.data_root,split=='train'),
                              batch_size=args.batch_size,shuffle=split=='train',num_workers=args.num_workers)
             for split in ('train','val')}
    model=build_mobileunet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=GradScaler(device.type,enabled=device.type=='cuda'); best=-1.0; history=[]
    for epoch in range(args.epochs):
        model.train(); loss_sum=0.0
        for image,mask in tqdm(loaders['train'],desc=f'MobileUNet {epoch+1}/{args.epochs}'):
            image=image.to(device);mask=mask.to(device);optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(image);loss=loss_fn(logits,mask)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.item())*len(image)
        model.eval(); score_sum=0.0; total=0
        with torch.inference_mode():
            for image,mask in loaders['val']:
                image=image.to(device);mask=mask.to(device)
                per_sample=dice(model(image),mask);score_sum+=float(per_sample.sum());total+=len(image)
        score=score_sum/total; history.append({'epoch':epoch+1,'train_loss':loss_sum/len(loaders['train'].dataset),'val_dice':score})
        print(history[-1],flush=True)
        if score>best:
            best=score
            torch.save({'model_state':model.state_dict(),'val_dice':score,'epoch':epoch+1,
                        'metadata':{'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                    'n_train':len(loaders['train'].dataset),'n_val':len(loaders['val'].dataset),
                                    'seed':args.seed}},out)
    out.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(f'Best validation Dice={best:.4f}; checkpoint={out} (new experiment)')

if __name__=='__main__':
    main()
