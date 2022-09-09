import torch
import torch.utils.data
import torchvision

import tqdm
import pickle


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None, adjusting_ratio=5):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        self.yaw_label = [-1]*len(self.indices)
        self.yaw_list = [-1]*len(self.indices)

        self.cached_file="/mnt/scratch/v_liuhaolan/cached_yaw"
        file_handle = open(self.cached_file, "rb")          
        (self.yaw_list) = pickle.load(file_handle)  
        file_handle.close()
        
        label_to_count = {}
        for idx in tqdm.tqdm(self.indices):
            label = self._get_label_cached(dataset, idx)
            self.yaw_label[idx] = label
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        
        """
        label_to_count = {}
        for idx in tqdm.tqdm(self.indices):
            label,yaw_value = self._get_label(dataset, idx)
            self.yaw_label[idx] = label
            self.yaw_list[idx] = yaw_value
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        
        
        
        # add logic, if exists, read it into the memory
       
        file_handle = open(self.cached_file, "wb")          
        pickle.dump((self.yaw_list), file_handle)  
        file_handle.close()
        """

        
        label_to_count[2] = label_to_count[2]*adjusting_ratio
        
        print(label_to_count)
        
        #self.additional_file="/mnt/scratch/v_liuhaolan/cached_yaw"
        
        # weight for each sample
#        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
#                   for idx in tqdm.tqdm(self.indices)]
        weights = [1.0 / label_to_count[self.yaw_label[idx]]
            for idx in tqdm.tqdm(self.indices)]
        
        
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        yaw = dataset[idx]["target_yaws"][-1]
#        yaw = dataset.get_yaw(idx)[-1]
        if yaw > 0.4:
            return 0, yaw
        elif yaw < -0.4:
            return 1, yaw
        else:
            return 2, yaw
        
    def _get_label_cached(self, dataset, idx):
        yaw = self.yaw_list[idx]
#        yaw = dataset.get_yaw(idx)[-1]
        if yaw > 0.4:
            return 0
        elif yaw < -0.4:
            return 1
        else:
            return 2

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


# usage
# see viz_notebook/sampler.ipynb
