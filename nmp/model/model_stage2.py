import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .mlp import MLP, Identity

class RasterizedTNTStageTwo(nn.Module):
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

        self.motion_network = MLP(in_channels=(2048+2), out_channels=num_targets, hidden_unit=num_mlp_hidden)

    def forward(self, data_batch: Dict[str, torch.Tensor], goal_batch) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        embedding = self.model(image_batch)
 #       outputs = self.fc(embedding)
        batch_size = len(data_batch["image"])

        
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
           
            loss = motion_loss

            train_dict = {"loss": loss}
#            return train_dict
        if not self.training:
            
            
            # using target selection
            stage2_input_batch_eval = torch.cat([embedding,goal_batch],dim=1)
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
        return train_dict