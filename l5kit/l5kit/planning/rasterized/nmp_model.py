import warnings
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from l5kit.environment import models


class NMPPlanningModel(nn.Module):
    """Neurla Motion Planner"""

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
            self.model.fc = nn.Linear(in_features=512, out_features=num_targets)
        elif model_arch == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)
        elif model_arch == "simple_cnn":
            self.model = models.SimpleCNN_GN(self.num_input_channels, num_targets)
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


    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # [batch_size, channels, height, width]
        image_batch = data_batch["image"]
        # [batch_size, num_steps * 2]
        outputs = self.model(image_batch)
        batch_size = len(data_batch["image"])

        if self.training:
            if self.criterion is None:
                raise NotImplementedError("Loss function is undefined.")

            # [batch_size, num_steps * 2]
#            targets = (torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)).view(
#                batch_size, -1
#            )
            # [batch_size, num_steps]
#            target_weights = (data_batch["target_availabilities"].unsqueeze(-1) * self.weights_scaling).view(
#                batch_size, -1
#            )
 #           loss = torch.mean(self.criterion(outputs, targets) * target_weights)
            cost_map = outputs.reshape(batch_size, 112, 112)
            loss = max_margin_loss(data_batch["negative_positions_pixels"], data_batch["target_positions_pixels"], cost_map)

            train_dict = {"loss": loss, "cost_map": cost_map }
            return train_dict
        else:
            predicted = outputs.view(batch_size, -1, 3)
            # [batch_size, num_steps, 2->(XY)]
            pred_positions = predicted[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = predicted[:, :, 2:3]
            eval_dict = {"positions": pred_positions, "yaws": pred_yaws}
            return eval_dict


def max_margin_loss(batch_negative_index, batch_gt_index, cost_map):
    # calculate the costs aggregated with regard to time.
    batch_size = batch_negative_index.shape[0] 
    traj_size = 12
    pixel_size = 0.5
    loss = 0
    for i in range(batch_size):
        each_loss = 0
        diff = batch_gt_index[i] - batch_negative_index[i]
        diff = torch.sum(torch.pow(torch.sum(torch.pow(diff, 2),dim=1),0.5), dim=0).item()

        
        for j in range(traj_size):
            time_step = cost_map[i][batch_gt_index[i][j][0]][batch_gt_index[i][j][1]] - cost_map[i][batch_negative_index[i][j][0]][batch_negative_index[i][j][1]] + diff*pixel_size

            if time_step > 0:
                each_loss = each_loss + (time_step)
        loss = loss + each_loss
    return loss / batch_size
