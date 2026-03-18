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
🔹 Visualizing trajectory quality of single-step denoising: 
<p align="center">
  <img src="assets/trajectory_generation.png" width="600">
</p> 

## 📌 TODO List
- [x] Release training code
- [ ] Release deployment code (comming soon!) 
- [ ] Update experimental demo videos

## 🎥 Experiment Videos

https://github.com/user-attachments/assets/958a91cc-593c-4b95-90f3-1039ed96edb5

## ⚙️ Environment Setup (Training)

1. Clone the repository：  
    ```
    git clone https://github.com/FastNav1/FastNav.git
    ```
2. Set up the conda environment:
    ```
    conda env create -f train/train_environment.yml
    ```
3. Install packages:
    ```
    conda activate fastnav
    pip install -e train/
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```

## 📁 Directory 



## 📦 Data Preparation

### Download public datasets:
- [GoStanford2](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_)
- [RECON](https://sites.google.com/view/recon-robot/dataset) 
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [SACSoN](https://sites.google.com/view/sacson-review/huron-dataset) 

    After downloading the public datasets above, process these datasets using the following steps.

### Process datasets:

1. Process rosbag files with `process_bags.py` or RECON HDF5s with `process_recon.py`： 
    ```
    python train/process_bags.py  # for processing rosbags
    python train/process_recon.py # for processing RECON HDF5s
    ```   

    This will produce a dataset structured as follows:
    ``` 
    ├── <dataset_name>
    │   ├── <name_of_traj1>
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   ├── ...
    │   │   ├── T_1.jpg
    │   │   └── traj_data.pkl
    │   ...
    └── └── <name_of_trajN>
        	├── 0.jpg
        	├── 1.jpg
        	├── ...
            ├── T_N.jpg
            └── traj_data.pkl
    ```
    Each trajectory consists of forward-facing RGB observations and a corresponding `traj_data.pkl` dictionary containing the robot's `position` (and `yaw`) at each observation.

     - `position`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
     - `yaw`: An np.ndarray [T,] of the yaws of the robot at each image observation.

  2. Run `data_split.py` on your dataset folder with the relevant args. After running `data_split.py`, the processed data splits will be organized under `vint_release/train/vint_train/data/data_splits/` with the following structure:  
      ``` 
      ├── <dataset_name>
      │   ├── train
      |   |   └── traj_names.txt
      └── └── test
              └── traj_names.txt 
      ```
    
## 🧠 Model Training

The training pipeline of **FastNav** consists of two stages.  

1. We first train a teacher navigation model based on a full diffusion policy to learn robust trajectory generation from visual observations.
      ```bash
      python train_teacher.py --config config/fastnav_teacher.yaml

2. After training the teacher model, we perform distillation to compress the multi-step denoising process into a one-step denoising student model, enabling fast inference during deployment.
      ```
      python train_distill.py --config config/distill.yaml
      ```

      The config yaml files are in the `train/config` directory.

## 🚀 Deployment

1. Environment setup (Deployment)
    ```
    conda env create -f deployment/deployment_environment.yaml
    conda activate fastnav_deployment
    pip install -e train/
    pip install -e diffusion_policy/
    ```
2. Place the trained model checkpoint `*.pth` in the `deployment/model_weights` and update `deployment/config/models.yaml` before starting inference.

3. Collecting Topological Map

    For collecting the **topological navigation map**, we follow the implementation provided in the [Collecting a Topological Map](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#collecting-a-topological-map). Please refer it. 

4. You can find the ROS1 (**[ros-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)**) and ROS2 (**[ros2-humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)**) inference implementations in [`navigate.py`](deployment/src/navigate.py) and [`navigate_2.py`](deployment/src/navigate_2.py).

    Make sure to run these script inside the `/deployment/src/` directory. 
    ```
    python navigate.py --model fastnav  # or python navigate_2.py --model fastnav
    python pd_controller.py  # or python pd_controller_2.py
    ```

    The model, robot, and ROS topic configurations can be adjusted in the `deployment/config/` folder and [`topic_names.py`](deployment/src/topic_names.py).

## 🤖 Experimental Platform

### Real-World Platform
Our real-world platform is a **mobile manipulation robot**, equipped with an **NVIDIA Jetson Orin NX** for onboard computation and an **Intel RealSense D435i** camera for RGB-only inputs.

### Simulation Platform
All simulation experiments are conducted in **[NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)** using the **Jackal** mobile robot.  

We provide the complete **Universal Scene Description (USD)** file, including the **scene, robot configuration, and pre-configured Action Graph**, which can be found here: ➡️ [`simulation.usd`](assets/simulation.usd)


## Acknowledgement 

This work is built upon the **[NoMaD](https://github.com/robodhruv/visualnav-transformer)**. We sincerely thank the authors of NoMaD for their valuable contributions to the field of robotic navigation, which have greatly inspired and enabled our research.
