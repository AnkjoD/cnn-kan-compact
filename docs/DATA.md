# Data preparation

This repository does not redistribute medical images, DICOM files, patient
identifiers, derived crop caches, or dataset archives.

## RSNA

Prepare the RSNA Pneumonia Detection Challenge images locally and provide a
frozen CSV manifest with these columns:

```text
path,label,split
relative/or/absolute/path.png,0,train
relative/or/absolute/path.png,1,val
relative/or/absolute/path.png,0,test
```

Allowed split values are `train`, `val`, and `test`. Relative paths are resolved
under `paths.rsna_root`. The public code never recreates the split automatically.
This prevents a reconstructed split from being mistaken for the training split.

## Kermany

Point `paths.kermany_root` to a directory containing the standard
`chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}` layout. All images are combined
for zero-shot external evaluation because no Kermany image is used to train the
models in the primary protocol.

## Lung segmentation

Set `paths.segmentation_checkpoint` to the trained MobileUNet checkpoint. The
crop cache is generated locally and excluded by `.gitignore`.

Follow the original dataset licenses and terms of access. Do not commit images,
patient identifiers, manifest rows with identifying paths, or prediction files
that expose patient information.


Download and reconstruction commands: [DOWNLOAD_AND_PREFLIGHT.md](DOWNLOAD_AND_PREFLIGHT.md).
