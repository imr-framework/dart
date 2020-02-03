# DART - Deep learning Assisted Radiology reporT

## Datasets
1. Multimodal Brain Tumor Segmentation Challenge  2018 - https://www.med.upenn.edu/sbia/brats2018/data.html
2. Total Intracranial Vault (TICV) BC2 Atlases - https://my.vanderbilt.edu/masi/about-us/resources-data/

## Generating neuroanatomy labels
[Freesurfer's](freesurfer) `recon-all` pipeline is performed on `1000_3.nii` atlas from the TICVBC2 dataset. For example:
`recon-all -i <path to 1000_3.nii> -s bert -all`. Locate the automatic segmentation masks inside `bert/mri/` as a file called `aseg.mgz`.

## Data preprocessing
1. Atlas `1000_3.nii` from TICVBC2 dataset is of size `256x256x287`. Crop to size `256x256x256` (center-out).
2. Resample this cropped atlas to `240x240x155` using `resample_vol_fix_orientation.py`.
3. Similarly, resample `aseg.mgz` to `240x240x155`. Both atlas and segmentation masks should match in size now.

[freesurfer]: https://surfer.nmr.mgh.harvard.edu 
