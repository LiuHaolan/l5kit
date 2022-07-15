from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os

#!echo $PWD
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/scratch/v_liuhaolan/l5kit_data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./nmp-config.yaml")


perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)


# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer)

print(train_dataset)

from l5kit.planning.rasterized.nmp_model import NMPPlanningModel


model = NMPPlanningModel(
        model_arch="simple_cnn",
        num_input_channels=rasterizer.num_channels(),
#        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
        num_targets = cfg["raster_params"]["raster_size"][0]*cfg["raster_params"]["raster_size"][1],
        weights_scaling= [1., 1., 1.],
        criterion=nn.MSELoss(reduction="none")
        )

train_cfg = cfg["train_data_loader"]

# get quarter of the dataset
# train_dataset = train_dataset[:len(train_dataset)/2]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)

losses_train = []
model.train()
torch.set_grad_enabled(True)

epoch_num = cfg["train_params"]["epochs"]



optimizer = optim.Adam(model.parameters(), lr=1e-4)
epoch_num = 16

for epochs in range(epoch_num):

    iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
    for step,batch in enumerate(iter_bar):
        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        loss = result["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50000 == 0:
            print("model save")
            torch.save(model.state_dict(),"./nmp_planning_model.pt")

        losses_train.append(loss.item())
        iter_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")


