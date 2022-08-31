import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .mlp import MLP, Identity

from torchvision.ops import sigmoid_focal_loss

class RasterizedGoalMap(nn.Module):
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

        # driving waypoint final output
#        self.fc = nn.Linear(in_features=2048, out_features=num_targets)
        self.goal_network = MLP(in_channels=(self.end_shape), out_channels=112*112, hidden_unit=num_mlp_hidden)
        # try an adaptive pooling layer?
    
        self.motion_network = MLP(in_channels=(self.end_shape+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        goal_map = self.goal_network(embedding)
        
      
        
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
            gt_map_tensor = torch.zeros(batch_size, 112*112)
            index_tensor = (data_batch["goal_pixel"][:,0]*112 + data_batch["goal_pixel"][:,1]).type(torch.LongTensor)
            gt_map_tensor[torch.arange(batch_size), index_tensor] = 1           
            # gt_map_tensor = gt_map_tensor.type(torch.LongTensor)
            gt_map_tensor = gt_map_tensor.to(embedding.device)
            
            # [batch_size, num_steps]
            #print(target_traj.shape)
            #print(trained_traj.shape)
            motion_loss = torch.mean(self.criterion(trained_traj.view(batch_size,-1), target_traj))
          
             
            #yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            #aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())
            #aux_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(goal_map, gt_map_tensor)

            #aux_loss = sigmoid_focal_loss(goal_map, gt_map_tensor,gamma=2,reduction="sum")
            aux_loss = nn.MSELoss(reduction="mean")(goal_map, gt_map_tensor)

            alpha = 1.0
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:
            
            stage1_target = torch.argmax(goal_map, dim=1)
            stage1_target_2d = torch.zeros(batch_size,2)
            for i in range(batch_size):
                stage1_target_2d[i][0] =  int(stage1_target[i]/112)
                stage1_target_2d[i][0] =  (stage1_target[i]%112)
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
                stage1_target_2d[i][0] =  int(stage1_target[i]/112)
                stage1_target_2d[i][0] =  (stage1_target[i]%112)
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

