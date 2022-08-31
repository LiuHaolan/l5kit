

from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel

class VectorTNTModel(VectorizedUnrollModel):

    def __init__(  self,
        history_num_frames_ego: int,
        history_num_frames_agents: int,
        num_targets: int,
        weights_scaling: List[float],
        criterion: nn.Module,  # criterion is only needed for training and not for evaluation
        global_head_dropout: float,
        disable_other_agents: bool,
        disable_map: bool,
        disable_lane_boundaries: bool,
        detach_unroll: bool,
        warmup_num_frames: int,
        discount_factor: float,
        limit_predicted_yaw: bool = True,
    ) -> None:
        """ Initializes the model.
        :param history_num_frames_ego: number of history ego frames to include
        :param history_num_frames_agents: number of history agent frames to include
        :param num_targets: number of values to predict
        :param weights_scaling: target weights for loss calculation
        :param criterion: loss function to use
        :param gobal_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it
        :param disable_other_agents: ignore agents
        :param disable_map: ignore map
        :param disable_lane_boundaries: ignore lane boundaries
        :param detach_unroll: detach gradient between steps (disable BPTT)
        :param warmup_num_frames: "sample" warmup_num_frames by following the model's policy
        :param discount_factor: discount future_timesteps via discount_factor**t
        :param limit_predicted_yaw: limit predicted yaw to 0.3 * tanh(x) if enabled - recommended for more stable
            training
        """

        super().__init__( self,
            history_num_frames_ego,
            history_num_frames_agents,
            num_targets,
            weights_scaling,
            criterion: nn.Module,  # criterion is only needed for training and not for evaluation
            global_head_dropout,
            disable_other_agents,
            disable_map,
            disable_lane_boundaries,
            detach_unroll,
            warmup_num_frames,
            discount_factor,
            limit_predicted_yaw)


