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

from l5kit.geometry import transform_points


import os
import pickle
from tqdm import tqdm
import zlib
import pickle

def rasterize_proc(self, scene_index, state_index):
    res = self.get_frame(scene_index, state_index)
                
    index = scene_index*self.k + frame_index
    filename = os.path.join(self.cached_dir, str(index))
    self.filename_list[index] = filename
            
                # compress the result
    res = zlib.compress(pickle.dumps(res))
    # no need to cache them in memory right now
#    self.cached_item_list[index] = res
            
    file_handle = open(filename, "wb" )
    pickle.dump(res, file_handle)
    file_handle.close()


class CachedBaseEgoDataset(Dataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            preprocessed_path: str,
            k: int = 20
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
       
        # number of frames to pick up  
        self.k = k

        # the in memory list that keeps the cached data.
        self.cached_item_list = [None]*self.__len__()
        self.filename_list = [None]*self.__len__()
        self.cached_dir = preprocessed_path
        assert self.cached_dir is not None
#        self.cached_dir = "/home/haolan/Downloads/prediction_dataset/preprocessed"
       
        # if there is a pre-processed directory, read it into the memory.
        if len(self) == len(os.listdir(self.cached_dir)):
            print("Read preprocessed dataset into memory...")
            for name in tqdm(os.listdir(self.cached_dir)):
                idx = int(name)
                self.filename_list[idx] = os.path.join(self.cached_dir, name)
                
                filename = self.filename_list[idx]
                file_handle = open(filename, "rb" )
                res = pickle.load(file_handle)
                file_handle.close()
#                res = pickle.loads(zlib.decompress(res))
                self.cached_item_list[idx] = res
            return
    
        print("not reading the dataset?")
        return

        # the sampled 
        # skip that part
        for scene_index in tqdm(range(len(self.dataset.scenes))):
            if scene_index == 0:
                frame_num = self.cumulative_sizes[0]
            else:
                frame_num = self.cumulative_sizes[scene_index] - self.cumulative_sizes[scene_index-1]

            frame_num = frame_num - 40

#            pool = multiprocessing.Pool(20)
#            processes = []

            # starting from 20, till -20
            for frame_index in range(self.k):
                # maximum: 180
                state_index = 20+int(frame_num/self.k) * frame_index
#                processes.append(pool.apply_async(rasterize_proc, args=(self, scene_index, state_index)))
#            _ = [p.get() for p in processes]
                res = self.get_frame(scene_index, state_index)
                
                index = scene_index*self.k + frame_index
                filename = os.path.join(self.cached_dir, str(index))
                self.filename_list[index] = filename
            
                # compress the result
                res = zlib.compress(pickle.dumps(res))
                # no need to cache them in memory right now
                #    self.cached_item_list[index] = res
            
                file_handle = open(filename, "wb" )
                pickle.dump(res, file_handle)
                file_handle.close()


    def _get_sample_function(self) -> Callable[..., dict]:
        raise NotImplementedError()

    
    def __len__(self) -> int:
        """
        Get the number of available AV frames

        Returns:
            int: the number of elements in the dataset
        """
#        return len(self.dataset.frames)
        return len(self.dataset.scenes)*self.k

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
        
        # maintain a list of cached_data in the memory.
        
        
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
        data["negative_positions"] = data["target_positions"].copy()

        import random
        shooting_delta = random.randint(-6, 6)/10.0
        delta = 0.05 if shooting_delta > 0 else -0.05
        for i in range(len(data["negative_positions"])):
            delta = delta + shooting_delta
            data["negative_positions"][i][1] -= delta
        
        # also build image coordinate
        data['target_positions_pixels'] = np.round(transform_points(data["target_positions"], data["raster_from_agent"]),0).astype(int)
        data['negative_positions_pixels'] = np.round(transform_points(data["negative_positions"], data["raster_from_agent"]),0).astype(int)

        return data

    def __getitem__(self, index: int) -> dict:
        """
        Function called by Torch to get an element

        Args:
            index (int): index of the element to retrieve

        Returns: please look get_frame signature and docstring

        """
        
        # if index already in memory, fetch it from the cached_item_list

        if self.cached_item_list[index] is not None:
            return pickle.loads(zlib.decompress(self.cached_item_list[index]))
        
        else:
            # read from the predefined files
            filename = self.filename_list[index]
            file_handle = open(filename, "rb" )
            res = pickle.load(file_handle)
            file_handle.close()
            res = pickle.loads(zlib.decompress(res))
            return res

        
        # the following is not run
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
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        
        scene_index = index
        if scene_index == 0:
            state_index = int(self.cumulative_sizes[index]/2)
        else:
            state_index = int((self.cumulative_sizes[index]-self.cumulative_sizes[index-1])/2)
            
        res = self.get_frame(scene_index, state_index)
        
        self.cached_item_list[index] = res
        return res

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


class CachedEgoDataset(CachedBaseEgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            preprocessed_path: str = None,
            k = 20
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
        super().__init__(cfg, zarr_dataset, preprocessed_path, k)

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


class EgoDatasetVectorized(CachedBaseEgoDataset):
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