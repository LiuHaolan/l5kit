import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .mlp import MLP, Identity

from torchvision.ops import sigmoid_focal_loss

import math

def gaussian(xL, yL, H, W, sigma=2):

    channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel 

def batch_gaussian_tensor(xL, yL, H, W, device, sigma=5 ):

 #channel = torch.zeros((H*W), dtype=torch.float)#.to("cuda:0")
    batch_size = xL.shape[0]
    channel = torch.zeros(batch_size, H, W).to(device)

    x = torch.arange(H).view(H,1).repeat(1,W).to(device)
    y = torch.arange(W).repeat(H,1).to(device)
    
    for i in range(batch_size):
        channel[i] = torch.exp((-((x - xL[i]) ** 2 + (y - yL[i]) ** 2) / (2 * sigma ** 2)))

    channel = channel.reshape(batch_size, H, W)

    return channel



def batch_gaussian_tensor_obj(xL, yL, H, W, obj_list, obj_num, road_mask,  device, sigma=5 ):

 #channel = torch.zeros((H*W), dtype=torch.float)#.to("cuda:0")
    batch_size = xL.shape[0]
    channel = torch.full((batch_size, H, W), 0.5).to(device)

    x = torch.arange(H).view(H,1).repeat(1,W).to(device)
    y = torch.arange(W).repeat(H,1).to(device)

    for i in range(batch_size):
        channel[i] += 0.5*torch.exp((-((x - xL[i]) ** 2 + (y - yL[i]) ** 2) / (2 * sigma ** 2)))
        
        obnum = obj_num[i]
        for j in range(obnum):
            xy = obj_list[i][j]
            xo = xy[1]
            yo = xy[0]
            channel[i] += -0.5*torch.exp((-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma ** 2)))
        
        channel[i][(road_mask[i]==0)] = 0
                
    return channel


import segmentation_models_pytorch as smp
class RasterizedGaussianGoalMapWithUNet(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        """
        self.end_shape = 0
        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            # "only resnet 50 is allowed at current time"
            self.end_shape = 512
            self.model.fc = Identity() #nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
            self.end_shape = 2048
            self.model.fc = Identity()
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        """

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
#        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
        self.goal_network = model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        self.motion_network = MLP(in_channels=(2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        # interpolate original tensor to 128x128
        embedding = torch.nn.functional.interpolate(image_batch, size=(128,128))
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        pre_goal_map = self.goal_network(embedding)
        
        goal_map = torch.nn.functional.interpolate(pre_goal_map, size=(112,112)).squeeze()
        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            #stage2_tensor = torch.cat([embedding, targets], dim=1)
            # no embedding!
            trained_traj = self.motion_network(targets)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            #gt_map_tensor = torch.zeros(batch_size, 112*112)
            #index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            #gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            #gt_map_tensor = gt_map_tensor.to(embedding.device)
            gt_goal_positions_pixels = data_batch["goal_pixel"]
            gt_map_tensor = batch_gaussian_tensor(gt_goal_positions_pixels[:,1], gt_goal_positions_pixels[:,0], 112, 112,sigma=2,device="cuda:0")#data_batch["gt_heatmap_full"]#

            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="mean")
            aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=4,reduction="mean")
            
            #aux_loss = nn.MSELoss(reduction="mean")(goal_map, gt_map_tensor)

            alpha = 10.0
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map.view(batch_size,-1), dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%112
                y = torch.div(stage1_target[i], 112, rounding_mode='floor')
                stage1_target_2d[i][0] = x
                stage1_target_2d[i][1] = y
                
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            #stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage1_target_2d).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = goal_map
            train_dict["predicted_goal"] = stage1_target_2d
            
        return train_dict

import math

class RasterizedGaussianGoalMap(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        self.end_shape = 0
        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            # "only resnet 50 is allowed at current time"
            self.end_shape = 512
            self.model.fc = Identity() #nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
            self.end_shape = 2048
            self.model.fc = Identity()
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        self.map_size = 308
        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=self.map_size, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
    
        self.motion_network = MLP(in_channels=(self.end_shape+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    #import math
    """
    def gaussian(xL, yL, H, W, sigma=2):

        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel 
    """
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        goal_map = self.goal_network(embedding)
        goal_map = torch.sigmoid(goal_map)

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            stage2_tensor = torch.cat([embedding, targets], dim=1)
            trained_traj = self.motion_network(stage2_tensor)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            #gt_map_tensor = torch.zeros(batch_size, 28,11)
            #index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            #gt_map_tensor[torch.arange(batch_size), data_batch["goal_gt"]] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            gt_map_tensor = data_batch["gt_heatmap"]#gt_map_tensor.to(embedding.device)
            
            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="sum")
            aux_loss = nn.MSELoss(reduction="sum")(goal_map, gt_map_tensor.view(batch_size,-1))

            alpha = 1.0
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map, dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%28
                y = stage1_target[i]//28
                stage1_target_2d[i][0] = 28 + x*2  
                stage1_target_2d[i][1] = 46 + y*2
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage2_input).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = goal_map
            train_dict["predicted_goal"] = stage1_target_2d
            
        return train_dict

import segmentation_models_pytorch as smp
class RasterizedGaussianGoalMapWithUNetGrid50(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64,
        weighted: bool = False,
        downweight: int = 1,
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.weighted = weighted
        self.downweight = downweight

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
#        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
        self.goal_network = model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        #self.embedding_mlp = MLP(in_channels=(112*112), out_channels=1024
        self.motion_network = MLP(in_channels=(2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        # interpolate original tensor to 128x128
        embedding = torch.nn.functional.interpolate(image_batch, size=(128,128))
        
#       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])
    
        # input: batch_size x 128 x 128
        pre_goal_map = self.goal_network(embedding)
        
        full_goal_map = torch.nn.functional.interpolate(pre_goal_map, size=(112,112)).squeeze().view(batch_size, 112, 112)
        
        goal_map = full_goal_map[:,28:84,28:84]
        goal_map = torch.sigmoid(goal_map) # activation

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            #stage2_tensor = torch.cat([embedding, targets], dim=1)
            # no embedding!
            trained_traj = self.motion_network(targets)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            #gt_map_tensor = torch.zeros(batch_size, 112*112)
            #index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            #gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            #gt_map_tensor = gt_map_tensor.to(embedding.device)
            gt_goal_positions_pixels = data_batch["goal_pixel"]
            #gt_map_tensor = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,0], gt_goal_positions_pixels[:,1], 56, 56, data_batch["ocg"], data_batch["obj_num"], data_batch["road_mask"],sigma=2,device="cuda:0")#data_batch["gt_heatmap_full"]#

            batch_road_mask = data_batch["road_mask"] 
            gt_map_tensor_full = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,1], gt_goal_positions_pixels[:,0], 112, 112, data_batch["ocg"], data_batch["obj_num"], batch_road_mask, sigma=5, device="cuda:0")

            gt_map_tensor = gt_map_tensor_full[:,28:84,28:84]


            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="mean")
            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="sum")
            
            if self.weighted:
                mse_loss = nn.MSELoss(reduction="none")(goal_map, gt_map_tensor)
                # add a weight that (score-0.5)^gamma, downweighting the middle prediction
                #gamma = 2
                batch_weight = torch.where(gt_map_tensor != 0.5, 1.0, self.downweight)

                alpha = 1.0
                aux_loss = torch.sum(mse_loss*batch_weight)
            
            else:
                mse_loss = nn.MSELoss(reduction="none")(goal_map, gt_map_tensor)
                aux_loss = torch.sum(mse_loss)
            
            alpha = 1.0
            loss = motion_loss + alpha*aux_loss
            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map.reshape(batch_size,-1), dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%56
                y = torch.div(stage1_target[i], 56, rounding_mode='floor')
                stage1_target_2d[i][0] = x + 28
                stage1_target_2d[i][1] = y + 28
                
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            #stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage1_target_2d).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = full_goal_map
            train_dict["predicted_goal"] = stage1_target_2d
            train_dict["gt_heatmap"] = gt_map_tensor_full

        return train_dict

import segmentation_models_pytorch as smp
class RasterizedGaussianGoalMapWithUNetGrid(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64,
        weighted: bool = False,
        downweight: int = 1,
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.weighted = weighted
        self.downweight = downweight

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
#        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
        self.goal_network = model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        #self.embedding_mlp = MLP(in_channels=(112*112), out_channels=1024
        self.motion_network = MLP(in_channels=(2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        # interpolate original tensor to 128x128
        embedding = torch.nn.functional.interpolate(image_batch, size=(128,128))
        
#       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])
    
        # input: batch_size x 128 x 128
        pre_goal_map = self.goal_network(embedding)
        
        full_goal_map = torch.nn.functional.interpolate(pre_goal_map, size=(112,112)).squeeze().view(batch_size, 112, 112)
        
        goal_map = full_goal_map[:,28:84,28:84]
        goal_map = torch.sigmoid(goal_map) # activation

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            #stage2_tensor = torch.cat([embedding, targets], dim=1)
            # no embedding!
            trained_traj = self.motion_network(targets)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            #gt_map_tensor = torch.zeros(batch_size, 112*112)
            #index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            #gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            #gt_map_tensor = gt_map_tensor.to(embedding.device)
            gt_goal_positions_pixels = data_batch["goal_pixel"]
            #gt_map_tensor = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,0], gt_goal_positions_pixels[:,1], 56, 56, data_batch["ocg"], data_batch["obj_num"], data_batch["road_mask"],sigma=2,device="cuda:0")#data_batch["gt_heatmap_full"]#

            batch_road_mask = data_batch["road_mask"] 
            gt_map_tensor_full = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,1], gt_goal_positions_pixels[:,0], 112, 112, data_batch["ocg"], data_batch["obj_num"], batch_road_mask, sigma=5, device="cuda:0")

            gt_map_tensor = gt_map_tensor_full[:,28:84,28:84]


            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="mean")
            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="sum")
            
            if self.weighted:
                mse_loss = nn.MSELoss(reduction="none")(goal_map, gt_map_tensor)
                # add a weight that (score-0.5)^gamma, downweighting the middle prediction
                #gamma = 2
                batch_weight = torch.where(gt_map_tensor != 0.5, 1.0, self.downweight)

                alpha = 1.0
                aux_loss = torch.sum(mse_loss*batch_weight)
            
            else:
                mse_loss = nn.MSELoss(reduction="none")(goal_map, gt_map_tensor)
                aux_loss = torch.sum(mse_loss)
            
            alpha = 1.0
            loss = motion_loss + alpha*aux_loss
            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map.reshape(batch_size,-1), dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%56
                y = torch.div(stage1_target[i], 56, rounding_mode='floor')
                stage1_target_2d[i][0] = x + 28
                stage1_target_2d[i][1] = y + 28
                
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            #stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage1_target_2d).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = full_goal_map
            train_dict["predicted_goal"] = stage1_target_2d
            train_dict["gt_heatmap"] = gt_map_tensor_full

        return train_dict

class RasterizedGaussianGoalMapWithUNetGridBCE(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64,
        weighted: bool = False,
        downweight: int = 1,
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.weighted = weighted
        self.downweight = downweight

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
#        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
        self.goal_network = model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        #self.embedding_mlp = MLP(in_channels=(112*112), out_channels=1024
        self.motion_network = MLP(in_channels=(2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        # interpolate original tensor to 128x128
        embedding = torch.nn.functional.interpolate(image_batch, size=(128,128))
        
#       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])
    
        # input: batch_size x 128 x 128
        pre_goal_map = self.goal_network(embedding)
        
        full_goal_map = torch.nn.functional.interpolate(pre_goal_map, size=(112,112)).squeeze()
        
        goal_map = full_goal_map[:,28:84,28:84]
        #goal_map = torch.sigmoid(goal_map) # activation

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            #stage2_tensor = torch.cat([embedding, targets], dim=1)
            # no embedding!
            trained_traj = self.motion_network(targets)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            #gt_map_tensor = torch.zeros(batch_size, 112*112)
            #index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            #gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            #gt_map_tensor = gt_map_tensor.to(embedding.device)
            gt_goal_positions_pixels = data_batch["goal_pixel"]
            #gt_map_tensor = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,0], gt_goal_positions_pixels[:,1], 56, 56, data_batch["ocg"], data_batch["obj_num"], data_batch["road_mask"],sigma=2,device="cuda:0")#data_batch["gt_heatmap_full"]#

            batch_road_mask = data_batch["road_mask"] 
            gt_map_tensor = batch_gaussian_tensor_obj(gt_goal_positions_pixels[:,1], gt_goal_positions_pixels[:,0], 112, 112, data_batch["ocg"], data_batch["obj_num"], batch_road_mask, sigma=2, device="cuda:0")

            gt_map_tensor = gt_map_tensor[:,28:84,28:84]


            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="mean")
            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="sum")
            
            if self.weighted:
                mse_loss = nn.MSELoss(reduction="none")(goal_map, gt_map_tensor)
                # add a weight that (score-0.5)^gamma, downweighting the middle prediction
                #gamma = 2
                batch_weight = torch.where(gt_map_tensor != 0.5, 1.0, self.downweight)

                alpha = 1.0
                aux_loss = torch.sum(mse_loss*batch_weight)
            
            else:
                mse_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(goal_map, gt_map_tensor)
                aux_loss = torch.sum(mse_loss)
            
            alpha = 1.0
            loss = motion_loss + alpha*aux_loss
            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map.reshape(batch_size,-1), dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%56
                y = torch.div(stage1_target[i], 56, rounding_mode='floor')
                stage1_target_2d[i][0] = x + 28
                stage1_target_2d[i][1] = y + 28
                
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            #stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage1_target_2d).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = goal_map
            train_dict["predicted_goal"] = stage1_target_2d
            
        return train_dict



from .unet import UNet

class RasterizedGoalMapAdaptive(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            assert False
            # "only resnet 50 is allowed at current time"
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            #self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
            #self.output_size = (112, 112)
            #self.avgpool = Identity() #torch.nn.AdaptiveAvgPool2d(self.output_size)
            self.model.fc = Identity()
        elif model_arch == "unet":
            """ 
            self.model = nn.Sequential(nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=56,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ), UNet())
            """
            self.model = UNet()
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        if model_arch in {"resnet18", "resnet50"} and self.num_input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        # print(self.model)

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
        self.goal_network = Identity()#MLP(in_channels=(2048), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
      
        self.output_size = (10,10)
        self.avgpool = nn.AdaptiveAvgPool2d(self.output_size)
        self.motion_network = MLP(in_channels=(100+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]

        #print(image_batch.shape)

        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        #print(embedding.shape)

        batch_size = len(data_batch["image"])
        embedding = embedding.view(batch_size,-1)

        goal_map = self.goal_network(embedding).view(batch_size, -1)
        
      
        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            stage2_embedding = self.avgpool(embedding.view(batch_size,1,112,112)).view(batch_size,-1)
            stage2_tensor = torch.cat([stage2_embedding, targets], dim=1)
            trained_traj = self.motion_network(stage2_tensor)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            # [batch_size,  2]
            gt_map_tensor = torch.zeros(batch_size, 112*112)
            
            index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            
            #gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            gt_map_tensor = gt_map_tensor.to(embedding.device)
            
            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            aux_loss = nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)


            loss = motion_loss + aux_loss

            train_dict = {"loss": loss, "loss1": motion_loss, "loss2": aux_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map, dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                stage1_target_2d[i][0] =  int(stage1_target[i]/112)
                stage1_target_2d[i][0] =  (stage1_target[i]%112)
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            results = torch.cat([stage2_embedding, stage1_target_2d], dim=1)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["heatmap"] = goal_map
 

        return train_dict




import segmentation_models_pytorch as smp
class RasterizedGaussianGoalMapWithUNetStage1(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        #num_modes: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 64
    ) -> None:
        """Initializes the planning model.
        :param model_arch: model architecture to use
        :param num_input_channels: number of input channels in raster
        :param num_targets: number of output targets
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = num_input_channels
        self.num_targets = num_targets,
#        self.num_modes
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
#        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
        self.goal_network = model = smp.Unet(
            encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=13,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        )

        #self.motion_network = MLP(in_channels=(2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        # interpolate original tensor to 128x128
        embedding = torch.nn.functional.interpolate(image_batch, size=(128,128))
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        pre_goal_map = self.goal_network(embedding)
        
        goal_map = torch.nn.functional.interpolate(pre_goal_map, size=(112,112)).squeeze()
        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            # gt_goal_batch = data_batch["target_positions"][:,-1,:].squeeze(dim=1) # take the last waypoint as target
            #stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            targets = (data_batch["goal_pixel"]).view(
                batch_size, -1
            ).float()
            
            #stage2_tensor = torch.cat([embedding, targets], dim=1)
            # no embedding!
            #trained_traj = self.motion_network(targets)
            
            target_traj = torch.cat([data_batch["target_positions"],data_batch["target_yaws"]], dim=2).view(batch_size, -1)

            loss = 0
            gt_goal_positions_pixels = data_batch["goal_pixel"]
            gt_map_tensor = batch_gaussian_tensor(gt_goal_positions_pixels[:,1], gt_goal_positions_pixels[:,0], 112, 112,sigma=2,device="cuda:0")#data_batch["gt_heatmap_full"]#

            #motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
            motion_loss = 0
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="mean")
            aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=4,reduction="mean")
            
            #aux_loss = nn.MSELoss(reduction="mean")(goal_map, gt_map_tensor)

            alpha = 1.0
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            """
            stage1_target = torch.argmax(goal_map.view(batch_size,-1), dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                x = stage1_target[i]%112
                y = torch.div(stage1_target[i], 112, rounding_mode='floor')
                stage1_target_2d[i][0] = x
                stage1_target_2d[i][1] = y
                
            stage1_target_2d = stage1_target_2d.to(stage1_target.device)
               
            stage1_target_2d = stage1_target_2d.float()
            #stage2_input = torch.cat([embedding, stage1_target_2d], dim=1)
            results = self.motion_network(stage1_target_2d).view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = results[:,:, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = results[:,:, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            """

            train_dict["heatmap"] = goal_map
            # train_dict["predicted_goal"] = stage1_target_2d
            
        return train_dict


