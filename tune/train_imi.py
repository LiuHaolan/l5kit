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

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/scratch/v_liuhaolan/l5kit_data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("../nmp/nmp-config.yaml")

perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)
mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])


perturbation = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)

"""
# ===== INIT DATASET
train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = EgoDataset(cfg, train_zarr, rasterizer, perturbation)

# plot same example with and without perturbation
for perturbation_value in [1, 0]:
    perturbation.perturb_prob = perturbation_value

    data_ego = train_dataset[1]
    im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
    target_positions = transform_points(data_ego["target_positions"], data_ego["raster_from_agent"])
    draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
    plt.imshow(im_ego)
    plt.axis('off')
    plt.show()
"""
# before leaving, ensure perturb_prob is correct
perturbation.perturb_prob = perturb_prob

preprocessed_dir = "/mnt/scratch/v_liuhaolan/preprocessed"

from l5kit.dataset import CachedEgoDataset

model = torch.load("./planning_model_v200.pt")
print(model)

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = CachedEgoDataset(cfg, train_zarr, rasterizer, preprocessed_path=preprocessed_dir,perturbation=perturbation)

"""
model = RasterizedPlanningModel(
        model_arch="resnet50",
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
        weights_scaling= [1., 1., 1.],
        criterion=nn.L1Loss(reduction="none")
        )
"""


#print(model)

train_cfg = cfg["train_data_loader"]
print(train_cfg["batch_size"])
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                             num_workers=train_cfg["num_workers"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(train_dataset)

from l5kit.dataset import EgoDataset
#from l5kit.evaluation import create_chopped_dataset

val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
val_dataset = EgoDataset(cfg, val_zarr, rasterizer, perturbation=perturbation)

val_cfg = cfg["val_data_loader"]
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                             num_workers=val_cfg["num_workers"])

optimizer = optim.Adam(model.parameters(), lr=1e-5)
epoch_num = 4
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

    cnt = 10
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
    if loss_average < max_val_loss:
        max_val_loss = loss_average
        torch.save(model, "./planning_model_v200_4epoch.pt")
