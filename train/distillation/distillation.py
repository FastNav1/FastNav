import os
import time
import math
import yaml
import wandb
import tqdm
import itertools

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW,Adam
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel


# from vint_train.models.vint.vint import ViNT
from vint_train.models.fastnav.fastnav import FastNav, DenseNetwork
from vint_train.models.fastnav.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# from vint_train.data.vint_dataset import ViNT_Dataset

from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.training.logger import Logger

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training

ACTION_STATS = {
    'min': np.array([-2.5, -4]),    # min_action_value [min_dx, min_dy]
    'max': np.array([5, 4])         # max_action_value [max_dx, max_dy]
}

class DistillTrainer:
    """
    Distills a pre-trained multi-step denoising teacher model
    into a single-step denoising student model.

    The goal is to compress the iterative denoising process
    into one-step trajectory generation for real-time navigation,
    while preserving trajectory quality and robustness.
    """
    def __init__(self, cfg, pth_path, device):
        super().__init__()
        self.cfg = cfg 
        self.device = device

        vision_encoder_net = NoMaD_ViNT(
                obs_encoding_size=self.cfg["encoding_size"],
                context_size=self.cfg["context_size"],
            )

        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.cfg["encoding_size"],
                down_dims=self.cfg["down_dims"],
                cond_predict_scale=self.cfg["cond_predict_scale"],
            )
        
        dist_pred_network = DenseNetwork(embedding_dim=self.cfg["encoding_size"])

        # Model for downloading original parameters only
        self.model = FastNav(
            vision_encoder = vision_encoder_net,
            noise_pred_net = noise_pred_net,
            dist_pred_net = dist_pred_network,
        )
        
        checkpoint = torch.load(pth_path, map_location=self.device)
        state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

        # vision_encoder
        self.vision_encoder = NoMaD_ViNT(
                obs_encoding_size=self.cfg["encoding_size"],
                context_size=self.cfg["context_size"],
            ).to(self.device)
        self.vision_encoder.load_state_dict(self.model.vision_encoder.state_dict())
        self.vision_encoder.eval()

        # dist_pred_network
        self.dist_predict = DenseNetwork(embedding_dim=self.cfg["encoding_size"]).to(self.device)
        self.dist_predict.load_state_dict(self.model.dist_pred_net.state_dict())
        self.dist_predict.eval()

        # teacher
        self.teacher = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.cfg["encoding_size"],
                down_dims=self.cfg["down_dims"],
                cond_predict_scale=self.cfg["cond_predict_scale"],
            ).to(self.device)
        
        self.teacher.load_state_dict(self.model.noise_pred_net.state_dict())

        # student
        self.student = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.cfg["encoding_size"],
                down_dims=self.cfg["down_dims"],
                cond_predict_scale=self.cfg["cond_predict_scale"],
            ).to(self.device)
        
        self.student.load_state_dict(self.model.noise_pred_net.state_dict())

        # aux
        self.aux = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=self.cfg["encoding_size"],
                down_dims=self.cfg["down_dims"],
                cond_predict_scale=self.cfg["cond_predict_scale"],
            ).to(self.device)
        
        self.aux.load_state_dict(self.model.noise_pred_net.state_dict())

        # optimizer
        lr = float(self.cfg["lr"])
        self.cfg["optimizer"] = self.cfg["optimizer"].lower()  
        if self.cfg["optimizer"] == "adam":
            self.optimizer_student = Adam(self.student.parameters(), lr=lr, betas=(0.9, 0.98))
            self.optimizer_aux = Adam(self.aux.parameters(), lr=1e-3, betas=(0.9, 0.98))
        elif self.cfg["optimizer"] == "adamw":
            print("[DEBUG] Using AdamW optimizer")
            self.optimizer_student = AdamW(self.student.parameters(), lr=lr)
            self.optimizer_aux = AdamW(self.aux.parameters(),lr=lr*10)
        elif self.cfg["optimizer"] == "sgd":
            self.optimizer_student = torch.optim.SGD(self.student.parameters(), lr=lr, momentum=0.9)
            self.optimizer_aux = torch.optim.SGD(self.aux.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {self.cfg['optimizer']} not supported")
        
        cosine_scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_student, T_max=self.cfg["epochs"]
        )
        cosine_scheduler_aux = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_aux, T_max=self.cfg["epochs"]
        )

        # using warmup_scheduler
        self.scheduler_student = GradualWarmupScheduler(
            self.optimizer_student,
            multiplier=1,
            total_epoch=self.cfg["warmup_epochs"],
            after_scheduler=cosine_scheduler_student
        )
        self.scheduler_aux = GradualWarmupScheduler(
            self.optimizer_aux,
            multiplier=1,
            total_epoch=self.cfg["warmup_epochs"],
            after_scheduler=cosine_scheduler_aux
        )
        
        # ddpm_scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.cfg["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.noise_scheduler_student = DDPMScheduler(
            num_train_timesteps=self.cfg["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # EMA_model
        self.ema_model_student = EMAModel(model=self.student, power=0.75)
        self.ema_model_aux = EMAModel(model=self.aux, power=0.75)

    def train_one_epoch(self, train_dataloader, epoch):
        print_log_freq: int = 100
        wandb_log_freq: int = 10
        image_log_freq: int = 1000

        action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
        action_waypts_cos_similairity = Logger("action_waypts_cos_similairity", "train", window_size=print_log_freq)
        multi_action_waypts_cos_similairity = Logger("action_waypts_cos_similairity", "train", window_size=print_log_freq)

        Loggers = {
            "action_loss": action_loss_logger,
            "action_waypts_cos_similairity": action_waypts_cos_similairity,
            "multi_action_waypts_cos_similairity": multi_action_waypts_cos_similairity
        }

        self.teacher.eval()
        self.dist_predict.eval()
        self.vision_encoder.eval()

        num_batches = len(train_dataloader)

        total_loss_student = 0

        cudnn.benchmark = True
        
        transform = ([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform = transforms.Compose(transform)

        # alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        with tqdm.tqdm(train_dataloader, desc=f"Train_batch", leave=False) as tepoch:
            for i, data in enumerate(tepoch):
                (   obs_image, 
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_idx,
                    action_mask, ) = data
                # to [(context_size+1), B, 3, H, W]
                obs_images = torch.split(obs_image, 3, dim=1) # list of (B,3,H,W)
                # current and goal for visualization
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]) # 当前观测
                batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
                
                batch_obs_images = [transform(obs) for obs in obs_images] # normalize
                # to [B, 3*(context_size+1), H, W]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(self.device)
                # to [B, 3, H, W]
                batch_goal_images = transform(goal_image).to(self.device)
                action_mask = action_mask.to(self.device)

                B = actions.shape[0]    
                goal_mask = None

                # Get distance label
                # distance = distance.float().to(device)

                # vision_condition
                with torch.no_grad():
                    obsgoal_cond = self.vision_encoder(
                                                obs_img=batch_obs_images, 
                                                goal_img=batch_goal_images, 
                                                input_goal_mask=None
                                                )
                # action
                deltas = get_delta(actions)
                ndeltas = normalize_data(deltas, ACTION_STATS)
                naction = from_numpy(ndeltas).to(self.device)
                assert naction.shape[-1] == 2, "action dim must be 2"

                # Sample noise to add to actions
                noise_1 = torch.randn(naction.shape, device=self.device)
                noise_2 = torch.randn(naction.shape, device=self.device)

                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()

                def action_reduce(unreduced_loss: torch.Tensor):
                    # Reduce over non-batch dimensions to get loss per batch element
                    while unreduced_loss.dim() > 1:
                        unreduced_loss = unreduced_loss.mean(dim=-1)
                    assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
                    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

                # =============================================================================================
                # Train Student
                # =============================================================================================

                self.aux.eval()
                self.student.train()

                gauss_noise_s = torch.randn(naction.shape, device=self.device)
                naction_s = gauss_noise_s

                self.noise_scheduler_student.set_timesteps(1)

                for k in self.noise_scheduler_student.timesteps[:]:
                    noise_pred_s = self.student(
                        sample=naction_s,
                        timestep=k.unsqueeze(0).repeat(B).to(self.device),
                        global_cond=obsgoal_cond,
                    )
                    naction_s = self.noise_scheduler_student.step(
                        model_output=noise_pred_s,
                        timestep=k,
                        sample=naction_s
                    ).prev_sample
                

                # noisy_action_shared (requires_grad=True)
                # fed into the teacher model and auxiliary model
                noisy_action_shared_t = self.noise_scheduler.add_noise(naction_s, noise_1, timesteps)
                noisy_action_shared_a = self.noise_scheduler.add_noise(naction_s, noise_1, timesteps)

                with torch.no_grad():
                    teacher_noise_pred = self.teacher(
                                            sample=noisy_action_shared_t,
                                            timestep=timesteps,
                                            global_cond=obsgoal_cond,
                                            )
                    aux_noise_pred = self.aux(
                                            sample=noisy_action_shared_a, 
                                            timestep=timesteps, 
                                            global_cond=obsgoal_cond
                                            )
                    
                # Compute the score gradient for updating the model
                # sigma_t = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1)
                # score_gradient = torch.nan_to_num(sigma_t**2 * (teacher_noise_pred - aux_noise_pred))
                score_gradient = (teacher_noise_pred - aux_noise_pred)
                target = (naction_s - score_gradient).detach()
                loss_kl = action_reduce(F.mse_loss(naction_s.float(), target.float(), reduction="none"))

                # structure loss
                # Teacher model: 10-step denoising generates final action：
                with torch.no_grad():
                    # Initial Gaussian noise（ for student's single-step denoising and also teacher‘s 10-step denoising）
                    naction_t = gauss_noise_s
                    for k in self.noise_scheduler.timesteps[:]: # [9,8,7....]
                        # predict noise
                        noise_pred_t = self.teacher(
                            sample=naction_t,
                            timestep=k.unsqueeze(-1).repeat(naction_t.shape[0]).to(self.device),
                            global_cond=obsgoal_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction_t = self.noise_scheduler.step(
                            model_output=noise_pred_t,
                            timestep=k,
                            sample=naction_t
                        ).prev_sample

                # naction_t: Teacher's result from 10-step denoising of Gaussian input [B, 2]
                # naction_s: Student's result from 1-step denoising of full Gaussian [B, 2]
                # naction: Ground truth action [B, 2]
                loss1 = Loss1(naction_s=naction_s, naction_t=naction_t, naction_g=naction, actions=actions.to(self.device), action_mask=action_mask)
                loss2 = Loss2(naction_s=naction_s, action_mask=action_mask)
                loss_NF = 0.9*loss1 + 0.1*loss2
                total_loss_student = 1.0*loss_NF + 1.0*loss_kl

                self.optimizer_student.zero_grad()
                total_loss_student.backward()
                self.optimizer_student.step()
                
                self.ema_model_student.step(self.student)

                # =============================================================================================
                # Train Aux with Student's one denoised output
                # =============================================================================================
                self.aux.train()
                self.student.eval()

                with torch.no_grad():
                    # student's 1-step denoising output
                    gauss_noise_s = torch.randn(naction.shape, device=self.device)
                    naction_s_aux = gauss_noise_s
                    self.noise_scheduler_student.set_timesteps(1)
                    
                    for k in self.noise_scheduler_student.timesteps[:]:
                        # print("[DEBUG]:Student denoise step: ", k)
                        # predict noise
                        noise_pred_s = self.student(
                            sample=naction_s_aux,
                            timestep=k.unsqueeze(-1).repeat(naction_s.shape[0]).to(self.device),
                            global_cond=obsgoal_cond
                        )
                        # inverse diffusion step (remove noise) -> naction_s
                        naction_s_aux = self.noise_scheduler_student.step(
                            model_output=noise_pred_s,
                            timestep=k,
                            sample=naction_s_aux
                        ).prev_sample

                    # Apply renoising to the student's 1-step denoising output as auxiliary model's input
                    noisy_action_aux = self.noise_scheduler.add_noise(naction_s_aux, noise_2, timesteps)
                    
                # Auxiliary model's noise prediction
                aux_noise_pred_2 = self.aux(
                                    sample=noisy_action_aux, 
                                    timestep=timesteps, 
                                    global_cond=obsgoal_cond
                                    )

                aux_diffusion_loss = action_reduce(F.mse_loss(aux_noise_pred_2, noise_2, reduction="none"))
 
                self.optimizer_aux.zero_grad()
                aux_diffusion_loss.backward()
                self.optimizer_aux.step()

                # self.ema_model_aux.step(self.aux)

                # log
                loss_cpu = total_loss_student.item()
                tepoch.set_postfix(loss=loss_cpu)

                wandb.log({"student_total_loss": loss_cpu})
                wandb.log({"KL_loss": loss_kl.item()})
                wandb.log({"Navigational_fidelity_loss": loss_NF.item()})
                wandb.log({"Adaptive_collaborative_loss": loss1.item()})
                wandb.log({"Path_efficiency_loss": loss2.item()})
                wandb.log({"aux_diffusion_loss": aux_diffusion_loss.item()})

                # evaluate while training
                if i % print_log_freq == 0:
                    eval_metrics = calculate_metrics(
                        ema_model=self.ema_model_student.averaged_model,
                        vision_net=self.vision_encoder,
                        dist_net=self.dist_predict,
                        noise_scheduler=self.noise_scheduler_student,
                        batch_obs_images=batch_obs_images,
                        batch_goal_images=batch_goal_images,
                        batch_action_label=actions.to(self.device),
                        device=self.device,
                        action_mask=action_mask.to(self.device)
                    )
                    for key, value in eval_metrics.items():
                        if key in Loggers:
                            logger = Loggers[key]
                            logger.log_data(value.item())
                    
                    data_log = {}
                    for key, logger in Loggers.items():
                        data_log[logger.full_name()] = logger.latest()
                        if i % print_log_freq == 0 and print_log_freq != 0:
                            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                    
                    if i % wandb_log_freq == 0 and wandb_log_freq != 0:
                        wandb.log(data_log, commit=True)
                    
                    if i % image_log_freq == 0 and image_log_freq != 0:
                        print("[DEBUG] Logging images in Train")
                        visualize_diffusion_action_in_img(
                            ema_model=self.ema_model_student.averaged_model,
                            vision_net=self.vision_encoder,
                            dist_net=self.dist_predict,
                            noise_scheduler=self.noise_scheduler_student,
                            batch_obs_images=batch_obs_images,
                            batch_goal_images=batch_goal_images,
                            batch_viz_obs_images=batch_viz_obs_images,
                            batch_viz_goal_images=batch_viz_goal_images,
                            batch_action_label=actions,
                            batch_distance_labels=distance,
                            batch_goal_pos=goal_pos,
                            device=self.device,
                            eval_type='train',
                            epoch=epoch,
                            num_images_log=8,
                        )


    def train_evaluate_loop(self, train_dataloader, test_dataloaders, epochs, train_mode:bool):
        for epoch in range(0, epochs):
            if train_mode:
                print(f"Start training epoch: {epoch}/{epochs}")
                self.train_one_epoch(train_dataloader=train_dataloader, epoch=epoch)
                self.scheduler_aux.step()
                self.scheduler_student.step()
            
            project_folder = self.cfg["project_folder"]
            latest_path = os.path.join(project_folder, f"latest.pth")

            # save model weights
            numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            torch.save(self.ema_model_student.averaged_model.state_dict(), numbered_path)
            numbered_path = os.path.join(project_folder, f"ema_latest.pth")
            print(f"Saved EMA model to {numbered_path}")

            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(self.student.state_dict(), numbered_path)
            torch.save(self.student.state_dict(), latest_path)
            print(f"Saved model to {numbered_path}")

            # Evaluate on test set at the end of each epoch
            if (epoch + 1) % 1 == 0:
                for dataset_type in test_dataloaders:
                    print(
                    f"Start {dataset_type} FastNav DP Testing Epoch {epoch}/{epochs - 1}"
                    )
                    loader = test_dataloaders[dataset_type]
                    self.evaluate_one_epoch(
                        eval_type=dataset_type,
                        test_dataloader=loader,
                        ema_model=self.ema_model_student.averaged_model,
                        epoch=epoch
                    )
            wandb.log({"lr_student": self.optimizer_student.param_groups[0]["lr"],}, commit=False)
            wandb.log({"lr_aux": self.optimizer_student.param_groups[0]["lr"],}, commit=False)


    def evaluate_one_epoch(self, eval_type:str, test_dataloader, ema_model, epoch):
        print_log_freq: int = 100
        wandb_log_freq: int = 10
        image_log_freq: int = 1000

        action_loss_logger = Logger("action_loss", eval_type, window_size=print_log_freq)
        action_waypts_cos_similairity = Logger("action_waypts_cos_similairity", eval_type, window_size=print_log_freq)
        multi_action_waypts_cos_similairity = Logger("multi_action_waypts_cos_similairity", eval_type, window_size=print_log_freq)

        Loggers = {
            "action_loss": action_loss_logger,
            "action_waypts_cos_similairity": action_waypts_cos_similairity,
            "multi_action_waypts_cos_similairity": multi_action_waypts_cos_similairity
        }

        num_batches = len(test_dataloader)
        num_batches = max(int(num_batches * 0.25), 1)

        cudnn.benchmark = True  # good if input sizes don't 
        
        transform = ([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform = transforms.Compose(transform)

        with tqdm.tqdm(
            itertools.islice(test_dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating {eval_type} for epoch {epoch}", 
            leave=False) as tepoch:
            for i, data in enumerate(tepoch):
                (
                    obs_image,
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_idx,
                    action_mask,
                ) = data
                print(f"[DEBUG-tqdm] Batch {i}: obs_image {obs_image.shape}, goal_image {goal_image.shape}, actions {actions.shape}")
                obs_images = torch.split(obs_image, 3, dim=1)
                batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=1).to(self.device)
                batch_goal_images = transform(goal_image).to(self.device)
                action_mask = action_mask.to(self.device)

                B = actions.shape[0]

                obsgoal_cond = self.vision_encoder(obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=None)
                obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

                distance = distance.to(self.device)

                deltas = get_delta(actions)
                ndeltas = normalize_data(deltas, ACTION_STATS)
                naction = from_numpy(ndeltas).to(self.device)
                assert naction.shape[-1] == 2, "action dim must be 2"

                print("Test-time evaluation")
                eval_metrics = calculate_metrics(
                    ema_model=ema_model,
                    vision_net=self.vision_encoder,
                    dist_net=self.dist_predict,
                    noise_scheduler=self.noise_scheduler_student,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_action_label=actions.to(self.device),
                    device=self.device,
                    action_mask=action_mask.to(self.device)
                )
                for key, value in eval_metrics.items():
                    if key in Loggers:
                        logger = Loggers[key]
                        logger.log_data(value.item())
                
                data_log = {}
                for key, logger in Loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
                
                if i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)
                
                if i % image_log_freq == 0 and image_log_freq != 0:
                    print("[DEBUG] Logging images in Test")
                    visualize_diffusion_action_in_img(
                        ema_model=ema_model,
                        vision_net=self.vision_encoder,
                        dist_net=self.dist_predict,
                        noise_scheduler=self.noise_scheduler_student,
                        batch_obs_images=batch_obs_images,
                        batch_goal_images=batch_goal_images,
                        batch_viz_obs_images=batch_viz_obs_images,
                        batch_viz_goal_images=batch_viz_goal_images,
                        batch_action_label=actions,
                        batch_distance_labels=distance,
                        batch_goal_pos=goal_pos,
                        device=self.device,
                        eval_type=eval_type,
                        epoch=epoch,
                        num_images_log=8,
                    )
               

def Loss1(naction_s, naction_t, naction_g, actions, action_mask):
    """
    Adaptive_collaborative_loss.
        naction_s: student's prediction
        naction_t: teacher's prediction
        naction_g: get_deltas(actions)
        actions: ground truth waypoints
    """
    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)  
    # Action loss between student and teacher
    loss_s_t = F.mse_loss(naction_s, naction_t, reduction='none') # [B,8,2]
    loss_s_t = action_reduce(loss_s_t)  # ->[]
    # Action loss between student and ground-truth
    loss_s_g = F.mse_loss(naction_s, naction_g, reduction='none') # [B,8,2]
    loss_s_g = action_reduce(loss_s_g)  # ->[]

    # waypoints_s = get_action(naction_s)
    watpoints_t = get_action(naction_t, ACTION_STATS)
    # Cosine similarity between teacher's and ground truth waypoints (direction consistency)
    action_waypts_cos_sim_t_g = F.cosine_similarity(watpoints_t[:, :, :2], actions[:, :, :2], dim=-1)    # ->[-1,1]
    action_waypts_cos_sim_t_g = action_reduce(action_waypts_cos_sim_t_g)    # ->[]
    action_waypts_cos_sim_t_g = torch.clamp(action_waypts_cos_sim_t_g, min=0, max=1)

    multi_action_waypts_cos_sim_t_g = F.cosine_similarity(
        torch.flatten(watpoints_t[:, :, :2], start_dim=1),
        torch.flatten(actions[:, :, :2], start_dim=1),
        dim=-1,) 
    multi_action_waypts_cos_sim_t_g = action_reduce(multi_action_waypts_cos_sim_t_g)    # ->[]
    action_waypts_cos_sim_t_g = torch.clamp(multi_action_waypts_cos_sim_t_g, min=0, max=1)

    cos_sim = 0.5*action_waypts_cos_sim_t_g + 0.5*multi_action_waypts_cos_sim_t_g  # -1~1

    # Loss1 = cos_sim/1.5 * loss_s_t + (1 - cos_sim/1.5) * loss_s_g
    Loss1 = cos_sim/2 * loss_s_t + (1 - cos_sim/2) * loss_s_g
    return Loss1

def Loss2(naction_s, action_mask):
    """Path_efficiency_loss"""
    def action_reduce(unreduced_loss: torch.Tensor):
        # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)  

    watpoints = get_action(naction_s, ACTION_STATS)
    end_point = watpoints[:,7,:]  # [B,2]
    straight_line_dist = torch.norm(end_point, dim=-1)      # [B]
    real_dist = torch.norm(naction_s, dim=-1).sum(dim=1)    # [B]
    path_efficiency = straight_line_dist / (real_dist + 1e-2)
    Loss2 = action_reduce(F.relu(1 - path_efficiency))
    return Loss2

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    diffusion_net: nn.Module,
    vision_net:nn.Module,
    dist_net: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    # num_samples: int,
    device: torch.device,
):
    obsgoal_cond = vision_net(obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=None)
   
    noisy_diffusion_output = torch.randn((len(obsgoal_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output
    noise_scheduler.set_timesteps(1)

    for k in noise_scheduler.timesteps[:]:
        # print("[DEBUG]evaluate denoise timestep: ", k)
        noise_pred = diffusion_net(
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    out_actions = get_action(diffusion_output,ACTION_STATS)

    out_distances = dist_net(obsgoal_cond)

    print(">>> model_output done")
    print("[DEBUG]out_actions shape: ", out_actions.shape)
    print("[DEBUG]out_distances shape: ", out_distances.shape)

    return {
        "out_actions": out_actions,
        "out_distances": out_distances
    }      
    
def calculate_metrics(    
    ema_model,
    vision_net,
    dist_net,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor
):
    """ Compute evaluation metrics for the test model: action_loss, action_waypts_cos_similairity, multi_action_waypts_cos_sim"""
    def action_reduce(unreduced_loss: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)  
    
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]
    output = model_output(
        diffusion_net = ema_model,
        vision_net=vision_net,
        dist_net=dist_net,
        noise_scheduler=noise_scheduler,
        batch_obs_images=batch_obs_images,
        batch_goal_images=batch_goal_images,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        device=device
    )
    out_actions = output["out_actions"]
    action_loss = action_reduce(F.mse_loss(out_actions, batch_action_label, reduction='none'))
    action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        out_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    multi_action_waypts_cos_similarity = action_reduce(F.cosine_similarity(
        torch.flatten(out_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))
    results = {
        "action_loss": action_loss,
        "action_waypts_cos_similairity": action_waypts_cos_similairity,
        "multi_action_waypts_cos_similairity": multi_action_waypts_cos_similarity,
    }
    return results

def visualize_diffusion_action_in_img(
        ema_model,
        vision_net,
        dist_net,
        noise_scheduler: DDPMScheduler,
        batch_obs_images: torch.Tensor,
        batch_goal_images: torch.Tensor,
        batch_viz_obs_images: torch.Tensor,
        batch_viz_goal_images: torch.Tensor,
        batch_action_label: torch.Tensor,
        batch_distance_labels: torch.Tensor,
        batch_goal_pos: torch.Tensor,
        device: torch.device,
        eval_type: str,
        epoch: int,
        num_images_log: int,
):
    """Plot action trajectories generated by diffusion in the image"""
    visualize_path = os.path.join(
        'path/to/visualization/folder',
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]
    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    batch_distance_labels = batch_distance_labels[:num_images_log]

    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    actions_list = []
    distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        output = model_output(
            diffusion_net=ema_model,
            vision_net=vision_net,
            dist_net=dist_net,
            noise_scheduler=noise_scheduler,
            batch_obs_images=obs,
            batch_goal_images=goal,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            device=device
        )
        
        output_actions = output['out_actions']
        output_distances = output['out_distances']
        # Save to CPU memory in numpy format
        actions_list.append(to_numpy(output_actions))
        distances_list.append(to_numpy(output_distances))

    # concatenate
    actions_list = np.concatenate(actions_list, axis=0)
    distances_list = np.concatenate(distances_list, axis=0)

    # split into actions per observation
    actions_list = np.split(actions_list, num_images_log, axis=0)
    # 8[1,1]
    distances_list = np.split(distances_list, num_images_log, axis=0)

    distance_avg = [np.mean(dist) for dist in distances_list]
    distance_std = [np.std(dist) for dist in distances_list]

    assert len(actions_list) == len(distances_list) == num_images_log, "Number of actions and distances should match number of images"

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):

        fig, ax = plt.subplots(1, 3)
        actions = actions_list[i]
        action_label = to_numpy(batch_action_label[i])
        traj_list = np.concatenate(
            [actions, action_label[None]],
            axis=0,
        )
        
        traj_colors = ["green"] * len(actions) + ["magenta"]
        traj_alphas = [0.1] * (len(actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]
        
        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )

        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={distance_avg[i]:.2f}±{distance_avg[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    
    wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)



if __name__ == "__main__":
    vision_encoder_net = NoMaD_ViNT(
                obs_encoding_size=256,
                context_size=5,
            )

    noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=256,
            down_dims=[64, 128, 256],
            cond_predict_scale=False,
        )
        
    dist_pred_network = DenseNetwork(256)

    model = FastNav(
        vision_encoder = vision_encoder_net,
        noise_pred_net = noise_pred_net,
        dist_pred_net = dist_pred_network,
    )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {count_parameters(model):,}")