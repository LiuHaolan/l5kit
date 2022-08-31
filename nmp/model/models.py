import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

import sys
sys.path.append( "/mnt/home/v_liuhaolan/haolan/l5kit/nmp/model")

from mlp import MLP, Identity

# built for centerline-based model
class RasterizedImitationModel(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
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
        self.num_targets = num_targets
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
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        #self.target_mlp = MLP(in_channels=(2048), out_channels=1, hidden_unit=128)
        
        self.motion_network = MLP(in_channels=(2048), out_channels=num_targets, hidden_unit=64)
        self.yaw_network = MLP(in_channels=(2048), out_channels=1, hidden_unit=64)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
        
        outputs = self.motion_network(embedding)
        outputs_yaw = self.yaw_network(embedding)

        batch_size = len(data_batch["image"])

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
            
            loss = 0
            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            motion_loss = torch.mean(self.criterion(outputs.view(batch_size,-1), targets) * target_weights)
        
            yaw_batch = data_batch["yaw"] + data_batch["random_shift"]
            # gt_goals
      #      aux_loss_tensor = (nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))

            aux_loss = nn.L1Loss(reduction="mean")(yaw_batch, outputs_yaw.squeeze())

            loss = motion_loss + aux_loss

            train_dict = {"loss": loss, "aux_loss": aux_loss}
#            return train_dict
        if not self.training:
#            predicted = outputs.view(batch_size, -1, 3)
            predicted = outputs.view(batch_size,-1,3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
        return train_dict

# grid-sampling Target-driven Model
class RasterizedTNT(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 128
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
        self.num_targets = num_targets
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
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        self.target_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        # inference for the stage 1
        goal_batch = data_batch["goal_list"]
        goal_num = data_batch["goal_num"][0]
        feature_repeat_batch = embedding.unsqueeze(1).repeat(1,goal_num,1)
        input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        x = self.target_mlp(input_batch).squeeze(dim=2)
        target_prediction = self.target_mlp_softmax(x)
        target_index = torch.argmax(target_prediction, dim=1)
 #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]
 
        # print(target_selection.shape) batch_size X 2
        # target_selection = target_selection.unsqueeze(1).repeat(1,goal_num,1)
        # print(target_selection.shape)

        x_offset = self.offset_x_mlp(input_batch)
        y_offset = self.offset_y_mlp(input_batch)
        xy_offset = torch.cat([x_offset, y_offset], dim=2)
        xy_offset = xy_offset[torch.arange(batch_size), data_batch["goal_gt"]]       
        # print(xy_offset.shape)

        # Shape: batch_size X goal_num X 2
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
#        batch_gt = batch_gt.unsqueeze(1).repeat(1, goal_num, 1)

        # print(target_prediction.shape)
        # x this time should be batch_num*500


        # inference for the stage 2
       # print(stage2_motion_traj.shape)
        # can be used for regression

        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            gt_goal_batch = data_batch["target_positions"][:,-1,:2] # take the last waypoint as target
            stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            stage2_motion_traj = self.motion_network(stage2_input_batch)
  

            loss = 0
            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            motion_loss = torch.mean(self.criterion(stage2_motion_traj.view(batch_size,-1), targets) * target_weights)
           
            # gt_goals
      #      aux_loss_tensor = (nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))
        #    aux_loss = 0
            
        #    for i in range(batch_size):
             #   goal_num = data_batch["goal_num"][i]
            aux_loss = torch.mean(self.criterion(torch.sum(target_selection+xy_offset,dim=1).float(), torch.sum(batch_gt,dim=1).float())) + torch.mean(nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))
            

            #aux_loss = aux_loss/batch_size

            alpha = 1.0
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "motion_loss": motion_loss, "target_loss": aux_loss}
#            return train_dict
        if not self.training:

            # using target selection
            stage2_input_batch_eval = torch.cat([embedding,target_selection],dim=1)
            stage2_motion_traj_eval = self.motion_network(stage2_input_batch_eval)
  
            # predicted = outputs.view(batch_size, -1, 3)
            predicted = stage2_motion_traj_eval.view(batch_size,-1,3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["target_softmax"] = target_selection
        return train_dict

# grid-sampling Target-driven Model
# with history
# but only one rasterized image
# only goals, no offset network like RasterizedTNT

class RasterizedTNTWithHistory(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        pretrained: bool = True,
        num_mlp_hidden: int = 128
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
        self.num_targets = num_targets
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
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        # goal number 
        self.target_mlp = MLP(in_channels=(2048), out_channels=10, hidden_unit=num_mlp_hidden)
        self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        # inference for the stage 1
        goal_batch = data_batch["goal_list"]

        #print(goal_batch.shape)

        goal_num = data_batch["goal_num"][0]
        #feature_repeat_batch = embedding.unsqueeze(1).repeat(1,goal_num,1)
        #input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        #print(goal_num)
        #print(data_batch["goal_gt"])

        # x: batch_size X goal_score
        x = self.target_mlp(embedding)
        target_prediction = self.target_mlp_softmax(x)
        target_index = torch.argmax(target_prediction, dim=1)
 #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]

        input_batch = torch.cat([embedding, target_selection], dim=1)

        # print(target_selection.shape) batch_size X 2
        # target_selection = target_selection.unsqueeze(1).repeat(1,goal_num,1)
        # print(target_selection.shape)

        x_offset = self.offset_x_mlp(input_batch)
        y_offset = self.offset_y_mlp(input_batch)
        xy_offset = torch.cat([x_offset, y_offset], dim=1)
        
        # clamp the xy offset
        BOUND = 5 
        xy_offset = torch.clamp(xy_offset, min=-BOUND, max=BOUND)

        #xy_offset = xy_offset[torch.arange(batch_size), data_batch["goal_gt"]]       
        # print(xy_offset.shape)

        # Shape: batch_size X goal_num X 2
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
#        batch_gt = batch_gt.unsqueeze(1).repeat(1, goal_num, 1)

        # print(target_prediction.shape)
        # x this time should be batch_num*500


        # inference for the stage 2
       # print(stage2_motion_traj.shape)
        # can be used for regression

        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 
            gt_goal_batch = data_batch["target_positions"][:,-1,:2] # take the last waypoint as target
            stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
            stage2_motion_traj = self.motion_network(stage2_input_batch)
  

            targets_stage1 = (data_batch["target_positions"][:,-1,:2]).view(
                batch_size, -1
            )
            

            loss = 0
            # [batch_size, num_steps * 2]
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )
            motion_loss = torch.mean(self.criterion(stage2_motion_traj.view(batch_size,-1), targets) * target_weights)
           
            # gt_goals
            # aux_loss = torch.mean(self.criterion(torch.sum(target_selection,dim=1).float(), torch.sum(batch_gt,dim=1).float())) 
             
            aux_loss1 =  torch.mean(self.criterion(target_selection+xy_offset, targets_stage1))
            aux_loss2 = (nn.NLLLoss(reduction="mean")(target_prediction, data_batch["goal_gt"]))
        
            aux_loss = aux_loss1 + aux_loss2
            #aux_loss = aux_loss/batch_size

            alpha = 0.3
            loss = motion_loss + alpha*aux_loss

            train_dict = {"loss": loss, "motion_loss": motion_loss, "target_loss": aux_loss, "classification": aux_loss1, "regression": aux_loss2}
#            return train_dict
        if not self.training:

            # using target selection
            stage2_input_batch_eval = torch.cat([embedding,target_selection],dim=1)
            stage2_motion_traj_eval = self.motion_network(stage2_input_batch_eval)
  
            # predicted = outputs.view(batch_size, -1, 3)
            predicted = stage2_motion_traj_eval.view(batch_size,-1,3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            train_dict["target_softmax"] = target_selection
        return train_dict




class RasterizedTNTWithHistoryStage1(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
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
        self.num_targets = num_targets
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
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        # goal number 
        self.target_mlp = MLP(in_channels=(2048), out_channels=10, hidden_unit=num_mlp_hidden)
        self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        # inference for the stage 1
        goal_batch = data_batch["goal_list"]

        #print(goal_batch.shape)

        goal_num = data_batch["goal_num"][0]
        #feature_repeat_batch = embedding.unsqueeze(1).repeat(1,goal_num,1)
        #input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        #print(goal_num)
        #print(data_batch["goal_gt"])

        # x: batch_size X goal_score
        x = self.target_mlp(embedding)
        target_prediction = self.target_mlp_softmax(x)
        target_index = torch.argmax(target_prediction, dim=1)
 #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]

        input_batch = torch.cat([embedding, target_selection], dim=1)

        # print(target_selection.shape) batch_size X 2
        # target_selection = target_selection.unsqueeze(1).repeat(1,goal_num,1)
        # print(target_selection.shape)

        x_offset = self.offset_x_mlp(input_batch)
        y_offset = self.offset_y_mlp(input_batch)
        xy_offset = torch.cat([x_offset, y_offset], dim=1)
        
        # clamp the xy offset
        BOUND = 5 
        xy_offset = torch.clamp(xy_offset, min=-BOUND, max=BOUND)

        #xy_offset = xy_offset[torch.arange(batch_size), data_batch["goal_gt"]]       
        # print(xy_offset.shape)

        # Shape: batch_size X goal_num X 2
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
#        batch_gt = batch_gt.unsqueeze(1).repeat(1, goal_num, 1)

        # print(target_prediction.shape)
        # x this time should be batch_num*500


        # inference for the stage 2
       # print(stage2_motion_traj.shape)
        # can be used for regression

        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 

            loss = 0
            # [batch_size, num_steps * 2]
            # targets = data_batch["target_positions"][:,-1,:2].view(
            #    batch_size, -1
            # )
            # [batch_size, num_steps]
            #motion_loss = torch.mean(self.criterion(stage1_goal.view(batch_size,-1), targets))
 
            #print(target_selection.shape)
            #print(xy_offset.shape)
            #print(targets.shape)
            
            # still regressing the target in rasterized image
            aux_loss1 =  torch.mean(self.criterion(target_selection+xy_offset, (batch_gt).float()))#.float()) 
            aux_loss2 = (nn.NLLLoss(reduction="mean")(target_prediction, data_batch["goal_gt"]))
        
            aux_loss = aux_loss1 + aux_loss2
            #aux_loss = aux_loss/batch_size

            alpha = 1.0
            loss = alpha*aux_loss

            train_dict = {"loss": loss, "classification": aux_loss2, "regression": aux_loss1}
#            return train_dict
        if not self.training:

            # using target selection
            pass 
        return train_dict



class RasterizedTNTCenterline(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
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
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            # "only resnet 50 is allowed at current time"
            self.model.fc = Identity()#nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        # goal number 
        # take in (observation, x, y)
        # output (score, delta_x, delta_y)
        self.last_layer = 512

        self.target_mlp = MLP(in_channels=(self.last_layer+2), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        self.motion_network = MLP(in_channels=(self.last_layer+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        #self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 
        batch_size = len(data_batch["image"])
        # inference for the stage 1

        # goal batch: batch_size X goal_num X 2
        goal_batch = data_batch["goal_list"]
        goal_num = data_batch["goal_num"]
        
        MAX_NUM = 1500
        feature_repeat_batch = embedding.unsqueeze(1).repeat(1,MAX_NUM,1)
        # input_batch: batch_size X 1500 X 2048 (x,y) in rasterization coordinates
        input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        # x: batch_size X goal_score
        x = self.target_mlp(input_batch)

        #target_prediction = self.target_mlp_softmax(x)
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]].squeeze()

        input_stage2_batch = torch.cat([embedding, batch_gt],dim=1)
        stage2_traj = self.motion_network(input_stage2_batch)

        #target_index = torch.argmax(target_prediction, dim=1)
 #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        
        # batch_size X 2
        #target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]
        #xy_offset = offset_batch[torch.arange(goal_batch.size(0)), target_index]

        #xy_offset = torch.clamp(xy_offset, min=-BOUND, max=BOUND)

        #xy_offset = xy_offset[torch.arange(batch_size), data_batch["goal_gt"]]       
        # print(xy_offset.shape)

        # Shape: batch_size X goal_num X 2
        # batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
#        batch_gt = batch_gt.unsqueeze(1).repeat(1, goal_num, 1)

        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 

            loss = 0
            # [batch_size, num_steps * 2]
            targets = torch.concat([data_batch["target_positions"][:,:,:], data_batch["target_yaws"][:,:,:]], dim=2).view(
                batch_size, -1
            )
            # [batch_size, num_steps]
            motion_loss = torch.mean(self.criterion(stage2_traj.view(batch_size,-1), targets))
 
            #print(target_selection.shape)
            #print(xy_offset.shape)
            #print(targets.shape)
            
            # still regressing the target in rasterized image
            #aux_loss1 =  torch.mean(self.criterion(target_selection+xy_offset, (batch_gt).float()))#.float()) 
            #aux_loss2 = (nn.NLLLoss(reduction="mean")(target_prediction, data_batch["goal_gt"]))
        
            
            #aux_loss = aux_loss1 + aux_loss2
            #aux_loss = aux_loss/batch_size

            goal_gt_batch = torch.zeros(batch_size, 1500).to(x.device)
            goal_gt_batch[torch.arange(batch_size), data_batch["goal_gt"].squeeze()] = 1
            #aux_loss = nn.BCEWithLogitsLoss(reduction='mean')(x[:,:goal_num], goal_gt_batch[:, :goal_num])
            aux_loss = 0
            for i in range(batch_size):
                aux_loss = nn.BCEWithLogitsLoss(reduction='mean')(x[i,:goal_num[i]].squeeze(), goal_gt_batch[i, :goal_num[i].squeeze()])
            aux_loss = aux_loss/batch_size

            alpha = 5.0
            loss = alpha*aux_loss + motion_loss

            train_dict = {"loss": loss, "loss1": aux_loss, "loss2": motion_loss}
#            return train_dict
        if not self.training:

            # using target selection
            #for i in range(batch_size)
            #stage1_target = torch.argmax(x[:,:goal_num] 
            pass 
        return train_dict





class RasterizedTNTWithHistoryStage1Version2(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        num_history : int,
        pretrained: bool = True,
        num_mlp_hidden: int = 64, 
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
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.num_history = num_history

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            assert False
            # "only resnet 50 is allowed at current time"
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        # goal number 
        # take in (observation, x, y)
        # output (score, delta_x, delta_y) 

        self.target_mlp = MLP(in_channels=(2048+self.num_history+2), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        #self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        #self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        image_embedding = self.model(image_batch)
        batch_size = len(data_batch["image"])
     
        history_batch = data_batch["history_positions"].view(batch_size,self.num_history)
        embedding = torch.cat([image_embedding, history_batch],dim=1)

        # inference for the stage 1

        # goal batch: batch_size X goal_num X 2
        goal_batch = data_batch["goal_list"]
        goal_num = data_batch["goal_num"][0]
        
        feature_repeat_batch = embedding.unsqueeze(1).repeat(1,goal_num,1)
        # input_batch: batch_size X goal_num X 2048 (x,y) in rasterization coordinates
        input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        # x: batch_size X goal_num
        x = self.target_mlp(input_batch)

        #target_prediction = self.target_mlp_softmax(x)
        
        #target_index = torch.argmax(target_prediction, dim=1)
 #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        
        # batch_size X 2
        #target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]
        #xy_offset = offset_batch[torch.arange(goal_batch.size(0)), target_index]

        #xy_offset = torch.clamp(xy_offset, min=-BOUND, max=BOUND)

        #xy_offset = xy_offset[torch.arange(batch_size), data_batch["goal_gt"]]       
        # print(xy_offset.shape)

        # Shape: batch_size X goal_num X 2
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
#        batch_gt = batch_gt.unsqueeze(1).repeat(1, goal_num, 1)

        # print(target_prediction.shape)
        # x this time should be batch_num*500


        # inference for the stage 2
       # print(stage2_motion_traj.shape)
        # can be used for regression

        
        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 

            loss = 0
            # [batch_size, num_steps * 2]
            # targets = data_batch["target_positions"][:,-1,:2].view(
            #    batch_size, -1
            # )
            # [batch_size, num_steps]
            #motion_loss = torch.mean(self.criterion(stage1_goal.view(batch_size,-1), targets))
 
            #print(target_selection.shape)
            #print(xy_offset.shape)
            #print(targets.shape)
            
            # still regressing the target in rasterized image
            #aux_loss1 =  torch.mean(self.criterion(target_selection+xy_offset, (batch_gt).float()))#.float()) 
            #aux_loss2 = (nn.NLLLoss(reduction="mean")(target_prediction, data_batch["goal_gt"]))
        
            
            #aux_loss = aux_loss1 + aux_loss2
            #aux_loss = aux_loss/batch_size
            aux_loss = nn.CrossEntropyLoss(reduction='mean')(x, data_batch["goal_gt"].unsqueeze(1))

            alpha = 1.0
            loss = alpha*aux_loss

            train_dict = {"loss": loss}#, "classification": aux_loss2, "regression": aux_loss1}
#            return train_dict
        if not self.training:

            # using target selection
            train_dict = {"goal_score": x}
        return train_dict


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP2(nn.Module):
    def __init__(self, hidden_size, out_features=None):
        super(MLP2, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states

class RasterizedGoalPlanning(nn.Module):
    """Raster-based planning model."""

    def __init__(
        self,
        model_arch: str,
        num_input_channels: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,
        num_history : int,
        pretrained: bool = True,
        num_mlp_hidden: int = 64, 
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
        self.num_targets = num_targets
        self.register_buffer("weights_scaling", torch.tensor(weights_scaling))
        self.pretrained = pretrained
        self.criterion = criterion
        self.num_history = num_history

        if pretrained and self.num_input_channels != 3:
            warnings.warn("There is no pre-trained model with num_in_channels != 3, first layer will be reset")

        if model_arch == "resnet18":
            self.model = resnet18(pretrained=pretrained)
            assert False
            # "only resnet 50 is allowed at current time"
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
#            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
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
        # goal number 
        # take in (observation, x, y)
        # output (score, delta_x, delta_y) 

        hidden_size = 64
        self.hidden_size = hidden_size
        self.target_mlp = MLP(in_channels=(2048+hidden_size+3), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_x_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)
        #self.offset_y_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=num_mlp_hidden)

        #self.goals_2D_mlps = MLP(in_channels=2, out_channels=hidden_size,hidden_unit=hidden_size)
        
        self.goals_2D_mlps = nn.Sequential(
                MLP2(2, hidden_size // 2),
                MLP2(hidden_size//2, hidden_size // 2),
                MLP2(hidden_size//2, hidden_size),
        ) 

        self.motion_network = MLP(in_channels=(2048+2+3), out_channels=num_targets, hidden_unit=num_mlp_hidden)
        self.target_mlp_softmax = nn.LogSoftmax(dim=1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        image_embedding = self.model(image_batch)
        batch_size = len(data_batch["image"])
     
        #history_batch = data_batch["history_positions"].view(batch_size,self.num_history)
        velocity_batch = torch.concat([data_batch["history_velocities"][:,-1,:], data_batch["history_yaws"][:,-1,:]],dim=1)
        embedding = torch.cat([image_embedding, velocity_batch],dim=1)

        # inference for the stage 1

        # goal batch: batch_size X goal_num X 2
        goal_batch = data_batch["goal_list"]
        goal_embedding = self.goals_2D_mlps(goal_batch.float())
        goal_num = data_batch["goal_num"][0]
        
        feature_repeat_batch = embedding.unsqueeze(1).repeat(1,goal_num,1)
        # input_batch: batch_size X goal_num X 2048 (x,y) in rasterization coordinates
        input_batch = torch.cat([feature_repeat_batch,goal_embedding],dim=2)

        # x: batch_size X goal_num
        x = self.target_mlp(input_batch)

        #target_prediction = self.target_mlp_softmax(x)
        
        #       target_selection = torch.index_select(goal_batch, dim=1, index=target_index)
        
        # batch_size X 2
        #target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]
        # Shape: batch_size X goal_num X 2
        batch_gt = data_batch["goal_list"][torch.arange(batch_size), data_batch["goal_gt"]]
       
        gt_goal_batch = data_batch["target_positions"][:,-1,:2] # take the last waypoint as target
        
        
        stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
        stage2_motion_traj = self.motion_network(stage2_input_batch)

        
       

        train_dict = {}
#        if self.training:
        if True:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")
 

            loss = 0
            # [batch_size, num_steps * 2]
            
            #targets = data_batch["target_positions"][:,-1,:2].view(
            #    batch_size, -1
            #)
            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
                batch_size, -1
            )
            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
                batch_size, -1
            )

            # [batch_size, num_steps]
            #distance = (self.criterion(stage1_goal.view(batch_size,-1), targets))
 
           
            # still regressing the target in rasterized image
            #aux_loss1 =  torch.mean(self.criterion(target_selection+xy_offset, (batch_gt).float()))#.float()) 
            #aux_loss2 = (nn.NLLLoss(reduction="mean")(target_prediction, data_batch["goal_gt"]))
            
            motion_loss = torch.mean(self.criterion(stage2_motion_traj.view(batch_size,-1), targets) * target_weights)
            
            #dist_tensor = torch.sum((target_selection - gt_goal_batch).pow(2), dim=1).sqrt().squeeze().detach()
            
            #dist_tensor = torch.mean(self.criterion(pseudo_motion_traj.view(batch_size,-1), stage2_motion_traj.view(batch_size,-1)) * target_weights, dim=1).squeeze().detach()

            #aux_loss = aux_loss1 + aux_loss2
            #aux_loss = aux_loss/batch_size
            aux_loss = nn.CrossEntropyLoss(reduction='mean')(x, data_batch["goal_gt"].unsqueeze(1)).squeeze()

            #loss_weight = torch.mean(dist_tensor * aux_loss)*0.01
            #print(dist_tensor.shape)
            #print(aux_loss.shape)
            #print(loss_weight.shape)

            alpha = 1.0
            loss = alpha*aux_loss + motion_loss

            train_dict = {"loss": loss, "loss2": motion_loss, "loss1": aux_loss}
#            return train_dict
        if not self.training:

            target_prediction = self.target_mlp_softmax(x).detach()
            target_index = torch.argmax(target_prediction, dim=1).squeeze()
            target_selection = goal_batch[torch.arange(goal_batch.size(0)), target_index]

            pseudo_input_batch = torch.cat([embedding,target_selection],dim=1)
            pseudo_motion_traj = self.motion_network(pseudo_input_batch)
        

            # predicted = outputs.view(batch_size, -1, 3)
            predicted = pseudo_motion_traj.view(batch_size,-1,3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]

            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
            

            # using target selection
            train_dict.update({"goal_score": x})
        return train_dict






