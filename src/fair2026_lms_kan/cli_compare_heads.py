"""Paired stratified bootstrap for AUC differences on identical test images."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compare(left, right, dataset, seed=42, iterations=2000):
    required={'dataset','sample_id','label','probability'}
    for name,frame in [('left',left),('right',right)]:
        if not required<=set(frame): raise ValueError(f'{name} missing {sorted(required-set(frame))}')
        if frame.duplicated(['dataset','sample_id']).any(): raise ValueError(f'{name}: duplicate image IDs')
    a=left[left.dataset==dataset][['sample_id','label','probability']]
    b=right[right.dataset==dataset][['sample_id','label','probability']]
    if len(a)==0 or len(a)!=len(b): raise ValueError('Empty or unequal dataset sizes')
    data=a.merge(b,on='sample_id',suffixes=('_left','_right'),validate='one_to_one')
    if len(data)!=len(a) or not (data.label_left==data.label_right).all():
        raise ValueError('Image IDs or labels do not match exactly')
    y=data.label_left.to_numpy(dtype=int)
    if set(y)!={0,1}: raise ValueError('Both classes required')
    l=data.probability_left.to_numpy(dtype=float);r=data.probability_right.to_numpy(dtype=float)
    if not np.isfinite(l).all() or not np.isfinite(r).all() or not ((0<=l)&(l<=1)).all() or not ((0<=r)&(r<=1)).all():
        raise ValueError('Probabilities must be finite and within [0,1]')
    rng=np.random.default_rng(seed);neg=np.flatnonzero(y==0);pos=np.flatnonzero(y==1)
    differences=np.empty(iterations)
    for i in range(iterations):
        indices=np.concatenate([rng.choice(neg,len(neg)),rng.choice(pos,len(pos))])
        differences[i]=roc_auc_score(y[indices],l[indices])-roc_auc_score(y[indices],r[indices])
    return {'dataset':dataset,'n':int(len(data)),'auc_left':float(roc_auc_score(y,l)),
            'auc_right':float(roc_auc_score(y,r)),
            'difference_left_minus_right':float(roc_auc_score(y,l)-roc_auc_score(y,r)),
            'ci95':np.quantile(differences,[0.025,0.975]).tolist(),
            'iterations':iterations,'seed':seed,
            'scope':'paired image-level uncertainty for these two checkpoints; not variability across training seeds'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left',help='predictions.csv of the first head')
    parser.add_argument('right',help='predictions.csv of the second head')
    parser.add_argument('--dataset',default='Kermany')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--iterations',type=int,default=2000)
    parser.add_argument('--output',default='head_comparison.json')
    args=parser.parse_args()
    if args.iterations<100: parser.error('--iterations must be >=100')
    result=compare(pd.read_csv(args.left,dtype={'sample_id':str}),
                   pd.read_csv(args.right,dtype={'sample_id':str}),args.dataset,args.seed,args.iterations)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
