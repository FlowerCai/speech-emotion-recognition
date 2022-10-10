import h5py
import torch
from torch.utils import data


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_IDs, labels, path, max_length):
        """
        Initialization
        list_IDs : list of sample names, used for getting one sample
        labels : dict, mapping sample name to label
        path : str, path to hdf5
        max_length : int ,max lengths of time frame
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.file_path = path
        self.max_length = max_length
        self.dataset = None

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        X = self.dataset[ID][()]
        X = torch.from_numpy(X)
        # [frames, bins]
        if self.max_length:
            X = X[:, :self.max_length]
        y = self.labels[ID]
        z = X.shape[1]

        return X, y, z, ID


from torch.utils.data.sampler import Sampler
import random


class EvenlyLengthSampler(Sampler):
    """
    split samples to four buckets, according to its length, and fetch samplers with evenly lengths
    """

    def __init__(self, data_source):
        """
        :param data_source: comes from Dataset, tuple of (x, y, z, ID)s,
        will be called as "
        training_generator = data.DataLoader(training_set, collate_fn=pad_and_sort_batch,
                                         sampler=SegmentCountSampler(training_set), **configs)"
        """
        # sort dataset by lengths to get a fixed permutation.Not necessary I think, since the permutation is fixed even
        # not sorted, since the seed is fixed. Still sort, not fixed is seen sometime.
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        quantiles = [136, 213, 344, 469]  # todo:may differ in folds.
        lengths = [x[2] for x in self.data_source]
        buckets = [[], [], [], []]
        for j, length in enumerate(lengths):
            for i, quantile in enumerate(quantiles):
                if length <= quantile:
                    buckets[i].append(j)
                    break
        for bucket in buckets:
            random.shuffle(bucket)
        total_list = []
        p = 0
        points = [0, 0, 0, 0]
        while p < len(lengths):
            for i in range(len(buckets)):
                if points[i] < len(buckets[i]):
                    total_list.append(buckets[i][points[i]])
                    points[i] += 1
                    p += 1
        return iter(total_list)

    def __len__(self):
        return len(self.data_source)
