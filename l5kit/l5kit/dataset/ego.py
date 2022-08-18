import bisect
from functools import partial
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling.agent_sampling import generate_agent_sample
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from l5kit.vectorization.vectorizer import Vectorizer

from l5kit.geometry import transform_point, transform_points


class BaseEgoDataset(Dataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
        """
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]

        # build a partial so we don't have to access cfg each time
        self.sample_function = self._get_sample_function()

    def _get_sample_function(self) -> Callable[..., dict]:
        raise NotImplementedError()

    def __len__(self) -> int:
        """
        Get the number of available AV frames

        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames)

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
            (position and yaw) along with their availability, the 2D matrix to center that agent,
            the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        # TODO (@lberg): this should be done in the sample function
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces

        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        # add information only, so that all data keys are always preserved
        data["scene_index"] = scene_index
        data["host_id"] = np.uint8(convert_str_to_fixed_length_tensor(self.dataset.scenes[scene_index]["host"]).cpu())
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        # we can sample a negative trajectory and return the pixel index
        # haolan: adding a negative_position entry
        # 
        """
        data["negative_positions"] = data["target_positions"].copy()

        import random
        shooting_delta = random.randint(-2, 2)/10.0
        delta = 0.04 if shooting_delta > 0 else -0.04
        for i in range(len(data["negative_positions"])):
            delta = delta + shooting_delta
            data["negative_positions"][i][1] -= delta
        
        # also build image coordinate
        data['target_positions_pixels'] = np.round(transform_points(data["target_positions"], data["raster_from_agent"]),0).astype(int)
        data['negative_positions_pixels'] = np.round(transform_points(data["negative_positions"], data["raster_from_agent"]),0).astype(int)
        """
        data = self.get_additional_info_grid_goal(data)
        return data

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index)

    def get_scene_dataset(self, scene_index: int) -> "BaseEgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        return BaseEgoDataset(self.cfg, dataset)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. EgoDataset iterates over frames, so this is just a matter
        of finding the scene boundaries.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        frames = self.dataset.frames
        assert frame_idx < len(frames), f"frame_idx {frame_idx} is over len {len(frames)}"
        return np.asarray((frame_idx,), dtype=np.int64)

    def __str__(self) -> str:
        return self.dataset.__str__()


class EgoDataset(BaseEgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
            None if not desired
        """
        self.perturbation = perturbation
        self.rasterizer = rasterizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        render_context = RenderContext(
            raster_size_px=np.array(self.cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(self.cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(self.cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=self.cfg["raster_params"]["set_origin_to_bottom"],
        )

        return partial(
            generate_agent_sample,
            render_context=render_context,
            history_num_frames=self.cfg["model_params"]["history_num_frames"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=self.rasterizer,
            perturbation=self.perturbation,
        )

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        data = super().get_frame(scene_index, state_index, track_id=track_id)
        # TODO (@lberg): this should not be here but in the rasterizer
        data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        return data

    def get_scene_dataset(self, scene_index: int) -> "EgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation)

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element

        Args:
        index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)

        if scene_index == 0:
            state_index = index
        else:
            state_index = index - self.cumulative_sizes[scene_index - 1]
        data = self.get_frame(scene_index, state_index)

        # getting our customized data in get_additional
        data = self.get_additional_info_grid_goal(data)
        # data = self.get_additional_info_centerline_goal(data)
        return data

    # centerline_goal version
    def get_additional_info_centerline_goal(self,data):
        raster_from_world = data["raster_from_world"]
        world_from_raster = np.linalg.inv(raster_from_world)

        from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
    
        rast = self.rasterizer
        raster_radius = float(np.linalg.norm(rast.raster_size * rast.pixel_size)) / 2
        center_in_raster_px = np.asarray(rast.raster_size) * (0.5, 0.5)
        center_in_world = transform_point(center_in_raster_px, world_from_raster)
        # TODO, raster_radius/4 to adjust target numbers
        lane_indices = indices_in_bounds(center_in_world, rast.sem_rast.mapAPI.bounds_info["lanes"]["bounds"], raster_radius/4)

        from l5kit.data.map_api import InterpolationMethod, MapAPI, TLFacesColors

        from collections import defaultdict
        from enum import IntEnum
        from typing import Dict, List, Optional
        
        # TB Tested
        INTERPOLATION_POINTS = 20
        lanes_mask: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(lane_indices) * 2, dtype=np.bool))
        lanes_area = np.zeros((len(lane_indices) * 2, INTERPOLATION_POINTS, 2))

        
        centerline_area = []
        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = rast.sem_rast.mapAPI.bounds_info["lanes"]["ids"][lane_idx]
            lane_dict = rast.sem_rast.mapAPI.get_lane_coords(lane_idx)
    
            step = max((lane_dict["xyz_left"]).shape[0],(lane_dict["xyz_right"]).shape[0])

            xyz_left = rast.sem_rast.mapAPI.interpolate(lane_dict["xyz_left"], step,InterpolationMethod.INTER_ENSURE_LEN)
            xyz_right = rast.sem_rast.mapAPI.interpolate(lane_dict["xyz_left"], step,InterpolationMethod.INTER_ENSURE_LEN)
            xyz_center = (xyz_left+xyz_right)/2
        
            mid_steps = 5
            xyz_center = rast.sem_rast.mapAPI.interpolate(xyz_center, mid_steps, InterpolationMethod.INTER_METER)
            xyz_center = xyz_center[:,:2]
            
            # get_lane_as_interpolation is stateful function
            # use stateless interpolate instead
    
            for p in xyz_center:
                xy_point = transform_point(p, raster_from_world)
                centerline_area.append(xy_point) 
        
        # needs to pad equally sized 300
        MAX_GOAL_NUM = 500
        assert len(centerline_area) < MAX_GOAL_NUM

#        goal_matrix=np.zeros((len(centerline_area),2),dtype=int)
        goal_matrix = np.zeros((MAX_GOAL_NUM,2),dtype=int)
        for k in range(len(centerline_area)):
            goal_matrix[k,:] = np.array([int(centerline_area[k][0]),int(centerline_area[k][1])])
        data["goal_list"] = goal_matrix
        data["goal_num"] = len(centerline_area)

        # finding the closest target
        gt_goal_positions_pixels = transform_point((data["target_positions"][-1,:2]), data["raster_from_agent"])
        # needs to slice the goal list to avoid using the zero-padded entry
        xy = (data["goal_list"][:len(centerline_area)]-gt_goal_positions_pixels)
        
        if len(xy) != 0:
            data["goal_gt"] = np.argmin(np.linalg.norm(xy, axis=-1))
        else:
            data["goal_gt"] = None

        return data
  
    
    def get_additional_info_grid_goal(self,data):
        raster_from_world = data["raster_from_world"]
        #assert raster_from_world != None

        world_from_raster = np.linalg.inv(raster_from_world)

        target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
        original_pixel = target_positions_pixels[0]

        centerline_area = []
        centerline_area.append((original_pixel[0],original_pixel[1]))
        for i in range(0,40,2):
            for j in range(-10,10,2):
                centerline_area.append((original_pixel[0]+i,original_pixel[1]+j))

        GOAL_NUM = len(centerline_area)
        assert GOAL_NUM == 201
        goal_matrix = np.zeros((GOAL_NUM,2),dtype=int)
        for k in range(len(centerline_area)):
            goal_matrix[k,:] = np.array([int(centerline_area[k][0]),int(centerline_area[k][1])])
        data["goal_list"] = goal_matrix
        data["goal_num"] = GOAL_NUM

        # finding the closest target
        gt_goal_positions_pixels = transform_point((data["target_positions"][-1,:2]), data["raster_from_agent"])
        # needs to slice the goal list to avoid using the zero-padded entry
        xy = (data["goal_list"]-gt_goal_positions_pixels)

        if len(xy) != 0:
            data["goal_gt"] = np.argmin(np.linalg.norm(xy, axis=-1))
        else:
            data["goal_gt"] = None
#            print("None goal gt!")

        return data


           

         

class EgoDatasetVectorized(BaseEgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired
        """
        self.perturbation = perturbation
        self.vectorizer = vectorizer
        super().__init__(cfg, zarr_dataset)

    def _get_sample_function(self) -> Callable[..., dict]:
        return partial(
            generate_agent_sample_vectorized,
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"]["history_num_frames_agents"],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"]["filter_agents_threshold"],
            perturbation=self.perturbation,
            vectorizer=self.vectorizer
        )

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetVectorized":
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDatasetVectorized(self.cfg, dataset, self.vectorizer, self.perturbation)
