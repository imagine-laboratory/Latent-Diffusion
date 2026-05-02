<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Text color */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">Learning Compact Representations of Agricultural Fields</span>: A Study of Variational Autoencoders Variants for Aerial Drone Imagery
</h1>

<div align="center">
  <a href="mailto:gonzalezhernandez.manfred@gmail.com" target="_blank">Manfred&nbsp;Gonzalez-Hernandez</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:dxie@ic-itcr.ac.cr" target="_blank">Danny&nbsp;Xie-Li</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:fabian.fallasmoya@ucr.ac.cr" target="_blank">Fabian&nbsp;Fallas-Moya</a><sup>3</sup>
  <a href="mailto:sebastian.rojasgonzalez@ugent.be" target="_blank">Sebastian&nbsp;Rojas-Gonzalez</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="mailto:Ivo.Couckuyt@ugent.be" target="_blank">Ivo&nbsp;Couckuyt</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <br>
  <sup>1</sup> Faculty of Engineering and Architecture, Ghent University - imec &emsp; <br>
  <sup>2</sup> Instituto Tecnológico de Costa Rica, Cartago, Costa Rica &emsp; <br>
  <sup>3</sup> Sede del Atlántico, Imagine Lab, Universidad de Costa Rica, Cartago, Costa Rica &emsp;
</div>

---
<p align="center">
  <img src="assets/Gecco_intro.jpg?raw=true" width="99.1%" />
</p>

## 📝 Abstract

While Latent Diffusion Models (LDMs) excel at synthetic data generation, they suffer from severe training instability on the small, homogeneous UAV datasets typical of precision agriculture. In this work, we investigate these architectural bottlenecks by training a Denoising Diffusion Probabilistic Model (DDPM) from scratch on constrained pineapple plantation imagery. Our systematic comparison reveals that continuous latent spaces ($\beta$-VAE) preserve complex crop structures, whereas discrete architectures (VQ-VAE) fail under scarce data conditions. Additionally, we demonstrate that replacing destabilizing global attention mechanisms with a purely convolutional U-Net backbone ensures smooth convergence in repetitive visual domains. Ultimately, combining this robust architecture with DDIM sampling yields a stable, highly efficient framework for low-data agricultural image synthesis. Code and models will be available at [https://github.com/imagine-laboratory/Latent-Diffusion](https://github.com/imagine-laboratory/Latent-Diffusion).

## Implementation details.
## create a conda env:
```bash
conda create -n ddpm_training python=3.10

conda activate ddpm_training

pip install numpy tqdm pillow pyYaml
```
Linux:
```bash
pip3 install torch torchvision torchaudio
```
Windows:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install opencv-python
pip install wandb
pip install wandb[media]
```
## Steps to clone this repo:

1. Clone it in an empty directory with the following command:
git clone --recurse-submodules <your-repo-URL>