from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.vectorization.vectorizer_builder import build_vectorizer


from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory
#from l5kit.planning.rasterized.model import RasterizedPlanningModel
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

import os

#!echo $PWD
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/mnt/scratch/v_liuhaolan/l5kit_data"
dm = LocalDataManager(None)
# get config
#cfg = load_config_data("./data_v2.yaml")
cfg = load_config_data("./config.yaml")

perturb_prob = cfg["train_data_loader"]["perturb_probability"]

# perturbation

mean = np.array([0.0, 0.0, 0.0])  # lateral, longitudinal and angular
std = np.array([0.5, 1.5, np.pi / 6])


perturbation = AckermanPerturbation(
        random_offset_generator=GaussianRandomGenerator(mean=mean, std=std), perturb_prob=perturb_prob)


perturbation.perturb_prob = perturb_prob



preprocessed_dir = "/mnt/home/v_liuhaolan/preprocessed_vectornet/"


# ===== INIT DATASET
from l5kit.dataset import CachedEgoDatasetVectorized

vectorizer = build_vectorizer(cfg, dm)


train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()
train_dataset = CachedEgoDatasetVectorized(cfg, train_zarr, vectorizer, perturbation=perturbation, if_preprocess = True, preprocessed_path=preprocessed_dir)
