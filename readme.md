## NeuralDiffuser: Controllable fMRI Reconstruction with Primary Visual Feature Guided Diffusion

### Environment

1. download ``nsddata``, ``nsddata_betas``, and ``nsddata_stimuli`` from `Natural Scenes Dataset` website: https://naturalscenesdataset.org/
2. download ``stable-diffusion-v-4`` from https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main
3. run `conda install -c conda-forge diffusers` to install `stable-diffusion-pipeline`

### Preprocessing

**!!! please modify the path in the src.**

```shell
cd src_icml24/data_prepare
python fmri_prepare.py --subject <subject> --mask_name nsdgeneral
python mask_prepare.py --subject <subject>
python img2feat_sd_pipe.py  --subject <subject> --gpu 0
python img2feat_guidance.py --subject <subject> --gpu 0
python make_subjstim.py --featname z --use_stim each --subject <subject>
python make_subjstim.py --featname z --use_stim ave --subject <subject>
python make_subjstim.py --featname c --use_stim each --subject <subject>
python make_subjstim.py --featname c --use_stim ave --subject <subject>
python make_subjstim.py --featname g --use_stim each --subject <subject>
python make_subjstim.py --featname g --use_stim ave --subject <subject>
```

### Training

```
cd src_icml24/model_train
python train.py --method neuraldiffuser --fetch_gt c --fetch_fmri nsdgeneral --subject <subject>
python file.py --subject <subject> --file tr_each
python file.py --subject <subject> --file te_ave
python train_guidance.py --fetch_gt g --fetch_fmri nsdgeneral --subject <subject> --method neuraldiffuser --Layers Linear-2, Linear-4, Linear-6, Linear-8, Linear-10, Linear-12
```

please download mindeye's pretrain model `autoencoder_<subject>_4x_locont_no_reconst` from https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/mindeye_models

```
python mindeye_scorez.py --method neuraldiffuser --fetch_gt z --fetch_fmri nsdgeneral --subject <subject>
```

### Reconstruction

```
cd src_icml24/Image_Reconstructe
python generate.py --method neuraldiffuser --subject <subject> --imgidx <0-981> --guidance_scale <500000 for paper> --guidance_strength <0.2 for paper> --niter <default=5> --ma <use momentum alignment?>
```

### Results

We show the subj01's results in `recon/0.0` for without guidance and `recon/500000.0/0.2` for with guidance. The first column of each figure represents the test image, and the subsequent 5 columns are repeatedly reconstructed images. Note that these results are $4\times$ downsampled due to file size. It would lead to blurry and unclear.





### Acknowledgement

We would like to thank the authors.

https://github.com/tknapen/nsd_access

https://github.com/huggingface/diffusers

https://github.com/openai/guided-diffusion

https://github.com/yu-takagi/StableDiffusionReconstruction

https://github.com/ReedOnePeck/MindDiffuser

https://github.com/MedARC-AI/fMRI-reconstruction-NSD