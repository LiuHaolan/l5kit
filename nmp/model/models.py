import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .mlp import MLP, Identity

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
        num_goals: int = 500
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
        self.target_mlp = MLP(in_channels=(2048+2), out_channels=1, hidden_unit=128)
        
        self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=128)
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
        feature_repeat_batch = embedding.unsqueeze(1).repeat(1,500,1)
        input_batch = torch.cat([feature_repeat_batch,goal_batch],dim=2)

        x = self.target_mlp(input_batch).squeeze(dim=2)
        for batch_num in range(x.shape[0]):
            x[batch_num][data_batch["goal_num"][batch_num]:] = -float("inf")

        target_prediction = self.target_mlp_softmax(x)
        # print(target_prediction.shape)        
        # x this time should be batch_num*500

        # inference for the stage 2
        gt_goal_batch = data_batch["target_positions"][:,-1,:2] # take the last waypoint as target
        stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
        stage2_motion_traj = self.motion_network(stage2_input_batch)
        # print(stage2_motion_traj.shape)
        # can be used for regression

        # also need to find the ground truth goal position by closest neighborhood
        # TODO
        
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
            motion_loss = torch.mean(self.criterion(stage2_motion_traj.view(batch_size,-1), targets) * target_weights)
           
            # gt_goals
      #      aux_loss_tensor = (nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))
            aux_loss = 0
            
            for i in range(batch_size):
                goal_num = data_batch["goal_num"][i]
                aux_loss = aux_loss + (nn.NLLLoss(reduction="none")(target_prediction[i][:goal_num], data_batch["goal_gt"][i]))


            aux_loss = aux_loss/batch_size

            loss = motion_loss + 0.1*aux_loss

            train_dict = {"loss": loss, "motion_loss": motion_loss, "target_loss": aux_loss}
#            return train_dict
        if not self.training:
#            predicted = outputs.view(batch_size, -1, 3)
            predicted = stage2_motion_traj.view(batch_size,-1,3)
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
        gt_goal_batch = data_batch["target_positions"][:,-1,:2] # take the last waypoint as target
        stage2_input_batch = torch.cat([embedding,gt_goal_batch],dim=1)
        stage2_motion_traj = self.motion_network(stage2_input_batch)
        # print(stage2_motion_traj.shape)
        # can be used for regression

        
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
            motion_loss = torch.mean(self.criterion(stage2_motion_traj.view(batch_size,-1), targets) * target_weights)
           
            # gt_goals
      #      aux_loss_tensor = (nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))
        #    aux_loss = 0
            
        #    for i in range(batch_size):
             #   goal_num = data_batch["goal_num"][i]
            aux_loss = torch.mean(self.criterion(torch.sum(target_selection+xy_offset,dim=1).float(), torch.sum(batch_gt,dim=1).float())) + torch.mean(nn.NLLLoss(reduction="none")(target_prediction, data_batch["goal_gt"]))
            

            #aux_loss = aux_loss/batch_size

            loss = motion_loss + 0.1*aux_loss

            train_dict = {"loss": loss, "motion_loss": motion_loss, "target_loss": aux_loss}
#            return train_dict
        if not self.training:
#            predicted = outputs.view(batch_size, -1, 3)
            predicted = stage2_motion_traj.view(batch_size,-1,3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
#            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            train_dict["positions"] = pred_positions
            train_dict["yaws"] = pred_yaws
        return train_dict

