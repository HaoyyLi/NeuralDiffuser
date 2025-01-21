## NeuralDiffuser: Neuroscience-inspired Diffusion Guidance for fMRI Visual Reconstruction

---

Reconstructing visual stimuli from functional Magnetic Resonance Imaging (fMRI) enables fine-grained retrieval of brain activity. However, the accurate reconstruction of diverse details, including structure, background, texture, color, and more, remains challenging. The stable diffusion models inevitably result in the variability of reconstructed images, even under identical conditions. To address this challenge, we first uncover the neuroscientific perspective of diffusion methods, which primarily involve top-down creation using pre-trained knowledge from extensive image datasets, but tend to lack detail-driven bottom-up perception, leading to a loss of faithful details. In this paper, we propose NeuralDiffuser, which incorporates primary visual feature guidance to provide detailed cues in the form of gradients. This extension of the bottom-up process for diffusion models achieves both semantic coherence and detail fidelity when reconstructing visual stimuli. Furthermore, we have developed a novel guidance strategy for reconstruction tasks that ensures the consistency of repeated outputs with original images rather than with various outputs. Extensive experimental results on the Natural Senses Dataset (NSD) qualitatively and quantitatively demonstrate the advancement of NeuralDiffuser by comparing it against baseline and state-of-the-art methods horizontally, as well as conducting longitudinal ablation studies.

![1](E:\Users\pc\Desktop\新建文件夹\img\1.png)

---

### Overview

This repository is a supplementary code to the paper named "[NeuralDiffuser: Neuroscience-inspired Diffusion Guidance for fMRI Visual Reconstruction](https://ieeexplore.ieee.org/document/10838320/)". It uses [stable-diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) to reconstruct natural images of human retina based on the [Natural Scene Dataset (NSD)](https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information).

It first projects fMRI voxels to the feature space of stable-diffusion inspired by [mindeyev2](https://medarc-ai.github.io/mindeye2/). Then, it trains a model to provide guided features (multiple feature layers from [CLIP-ViT-B-32](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)). Finally, they are fed to the proposed guided diffusion model to reconstruct natural images of the retina.The framework diagram is as follows:

![2](E:\Users\pc\Desktop\新建文件夹\img\2.png)

---

### Requirement

[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

[MindEye2](https://huggingface.co/datasets/pscotti/mindeyev2)

---

### Usage

1. To improve training efficiency, you need to run script `src/img2feat_sd_pipe.py` and `src/img2feat_guidance.py` to pre-save target embeddings (z: latent space of vqvae; c: text space of clip-vit-large-patch-14; g: features of layer-2,4,6,8,10,12 in CLIP-ViT-B-32).

2. training model

   ```shell
   cd scripts
   bash ./train.sh
   ```

3. inference

   ```shell
   cd scrips
   bash ./score.sh
   ```

4. reconstruction

   ```shell
   cd scrips
   bash ./recon.sh
   ```

---

### Results

![3](E:\Users\pc\Desktop\新建文件夹\img\3.png)

![4](E:\Users\pc\Desktop\新建文件夹\img\4.png)

![5](E:\Users\pc\Desktop\新建文件夹\img\5.png)

---

### Citing

If you use this repository in your research, please cite this paper.

```
@ARTICLE{li2024neuraldiffuser,
  author={Li, Haoyu and Wu, Hao and Chen, Badong},
  journal={IEEE Transactions on Image Processing}, 
  title={NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction}, 
  year={2025},
  volume={34},
  pages={552-565}}
```

---

### Acknowledgment

[Natural Scene Dataset (NSD)](https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information)

[stable-diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

[Mind-Eye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)

[MindEyeV2](https://github.com/MedARC-AI/MindEyeV2)

[MindDiffuser](https://github.com/ReedOnePeck/MindDiffuser)

I'd like to express thanks for [Reese Kneeland](https://www.reesekneeland.com/).