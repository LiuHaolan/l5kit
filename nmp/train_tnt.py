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
cfg = load_config_data("./goal-config.yaml")

# rasterisation and perturbation
rasterizer = build_rasterizer(cfg, dm)

preprocessed_dir = "/mnt/scratch/v_liuhaolan/preprocessed_goal"

from l5kit.dataset import CachedEgoDataset

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = CachedEgoDataset(cfg, train_zarr, rasterizer, preprocessed_path=preprocessed_dir)

from model.models import RasterizedTNTWithHistory
model = RasterizedTNTWithHistory(
        model_arch="resnet50",
        num_input_channels=rasterizer.num_channels(),
        num_targets=3 * cfg["model_params"]["future_num_frames"],  # X, Y, Yaw * number of future states,
        weights_scaling= [1., 1., 1.],
#        criterion=nn.L1Loss(reduction="none")
         criterion=torch.nn.HuberLoss(reduction="none"),
        num_mlp_hidden = 64
        )

train_cfg = cfg["train_data_loader"]
print(train_cfg["batch_size"])
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                             num_workers=train_cfg["num_workers"])
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:0"
#device = "cpu"

model = model.to(device)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(train_dataset)


from l5kit.dataset import EgoDataset
#from l5kit.evaluation import create_chopped_dataset

val_zarr = ChunkedDataset(dm.require(cfg["val_data_loader"]["key"])).open()
val_dataset = EgoDataset(cfg, val_zarr, rasterizer)

val_cfg = cfg["val_data_loader"]
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"],
                             num_workers=val_cfg["num_workers"])


optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 16
losses_train1 = []

max_val_loss = 100000

for epochs in range(epoch_num):

    iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')

    model.train()
    loss_avg = 0
    for step,batch in enumerate(iter_bar):
        # Forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        loss = result["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        losses_train1.append(loss.item())
        loss_v = np.round(loss.item(),3)
        loss_average = np.round(np.mean(losses_train1) ,3)

        motion_loss = result["motion_loss"]
        motion_loss = np.round(motion_loss.item(),3)
        target_loss = result["target_loss"]
        target_loss = np.round(target_loss.item(),3)
 
        cl_loss = result["classification"]
        cl_loss = np.round(cl_loss.item(),3)
        rg_loss = result["regression"]
        rg_loss = np.round(rg_loss.item(),3)

        loss_avg = loss_avg + loss.item()
#        iter_bar.set_description(f"loss: {(loss_v)} loss(avg): {loss_average}")
        iter_bar.set_description(f"loss: {(loss_v)} motion_loss: {motion_loss} target_loss: {target_loss} classification: {cl_loss} regression loss: {rg_loss}")
    #loss_avg = loss_avg / step
    #print("loss: {}".format(loss_avg))

    # write a loop in evaluation dataset
    model.eval()

    loss_val = 0
    loss1 = 0
    loss2 = 0

    cnt = 1000
    for step,batch in enumerate(val_dataloader):
        # inference pass
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        loss = result["loss"]
        loss1 = loss1 + result["motion_loss"].item()
        loss2 = loss2 + result["target_loss"].item()
    
        loss_val += loss.item()

        if step >= cnt:
            break
    loss_average = loss_val/cnt
    print("val loss: {}, motion_loss: {}, target_loss: {}".format(loss_average, loss1/cnt, loss2/cnt))
    
    torch.save(model.state_dict(),"./ckpt/nooffset/planning_tnt_{}_new.pt".format(epochs))


