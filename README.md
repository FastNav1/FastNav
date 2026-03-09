# FastNav：An Efficient and Relible Visual Navigation Approach via Diffusion Distillation

**Overview:** FastNav is an efficient vision-conditioned diffusion framework for RGB-only image-goal navigation. It performs navigation by generating future motion trajectories. The framework introduces Temporal-Dilated Attention (TDA) and Temporal–Feature Decoupled Reweighting (TFDR) mechanisms to maintain a lightweight architecture while preserving strong scene understanding. Additionally, it employs bidirectional-gradient distillation to compress the denoising process into a single denoising step for real-time generation of high-quality trajectories.  

🔹 Architecture of FastNav: 
<p align="center">
  <img src="assets/architecture_of_FastNav.png" width="600">
</p> 
🔹 Distillation Framework: 
<p align="center">
  <img src="assets/distillation.png" width="600">
</p> 

## 📌 TODO List

- [ ] Release training and deployment code (comming soon!) 
- [ ] Update experimental demo videos

## 🎥 Experiment Videos

<table>
<tr>
<td width="33%">
<video src="https://github.com/FastNav1/FastNav/issues/1#issue-4045158647" controls width="100%"></video>
</td>



## ⚙️ Environment Setup

1. Clone the repository
```
git clone https://github.com/FastNav1/FastNav.git
```
2. Create environment
```
conda activate fastnav
pip install -e train/
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

## 📁 Directory 

## 📦 Data Preparation

1. Download public datasets:
- [GoStanford2](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_)
- [RECON](https://sites.google.com/view/recon-robot/dataset) 
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [SACSoN](https://sites.google.com/view/sacson-review/huron-dataset)

2. 
