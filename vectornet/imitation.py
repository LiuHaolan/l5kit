

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDatasetVectorized
#from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory

from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os



# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/scratch/v_liuhaolan/l5kit_data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./config.yaml")

perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# rasterisation and perturbation
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])
perturbation = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()


#vectorizer = build_vectorizer(cfg, dm)
#train_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer, perturbation)

preprocessed_dir = "/mnt/home/v_liuhaolan/preprocessed_vectornet/"


weights_scaling = [1.0, 1.0, 1.0]

_num_predicted_frames = cfg["model_params"]["future_num_frames"]
_num_predicted_params = len(weights_scaling)

"""
model = VectorizedUnrollModel(
    history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
    history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
    num_targets=_num_predicted_params * _num_predicted_frames,
    weights_scaling=weights_scaling,
    criterion=nn.L1Loss(reduction="none"),
    global_head_dropout=cfg["model_params"]["global_head_dropout"],
    disable_other_agents=cfg["model_params"]["disable_other_agents"],
    disable_map=cfg["model_params"]["disable_map"],
    disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"],
    detach_unroll=cfg["model_params"]["detach_unroll"],
    warmup_num_frames=cfg["model_params"]["warmup_num_frames"],
    discount_factor=cfg["model_params"]["discount_factor"],

)
"""
model =  torch.load("./planning_model_0.pt")

# ===== INIT DATASET
from l5kit.dataset import CachedEgoDatasetVectorized

vectorizer = build_vectorizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = CachedEgoDatasetVectorized(cfg, train_zarr, vectorizer, perturbation=perturbation, if_preprocess = False, preprocessed_path=preprocessed_dir)


val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()

vectorizer_val = build_vectorizer(cfg, dm)
val_dataset = EgoDatasetVectorized(cfg, train_zarr, vectorizer_val)


train_cfg = cfg["train_data_loader"]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])

val_cfg = cfg["val_data_loader"]
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                             num_workers=val_cfg["num_workers"])


#from torch.nn import DataParallel

#if torch.cuda.device_count() > 1:
#    model = DataParallel(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

epoch_num = 8
losses_train1 = []


max_val_loss = 100000

for epochs in range(epoch_num):
    
    iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
    
    model.train()
    for step,batch in enumerate(iter_bar):
        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        loss = result["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
#        if step%10000 == 0:
#            print("model save")
#            torch.save(model.state_dict(),"./planning_model.pt")

        losses_train1.append(loss.item())
        loss_v = np.round(loss.item(),6)
        loss_average = np.round(np.mean(losses_train1) ,6)
        iter_bar.set_description(f"loss: {(loss_v)} loss(avg): {loss_average}")

    # write a loop in evaluation dataset
    
    model.eval()  
    loss_val = 0
    cnt = 100
    for step,batch in enumerate(val_dataloader):
        # inference pass
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        loss = result["loss"]
        
#        if step%10000 == 0:
#            print("model save")
#            torch.save(model.state_dict(),"./planning_model.pt")
        
        loss_val += loss.item()

        if step >= cnt:
            break

    loss_average = loss_val/cnt
    print("val loss: {}".format(loss_average))
#    if loss_average < max_val_loss:
#        max_val_loss = loss_average
#        torch.save(model, "./planning_model_{}.pt".format(epochs))
    
    torch.save(model, "./planning_model_{}.pt".format(epochs+1))



