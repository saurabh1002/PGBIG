from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import os, glob
import torch
from tqdm import tqdm
from utils.haggling_utils import normalize

import sys
sys.path.append("/home/user/social_motion")
from social_motion.datasets.haggling import get_haggling_train_sequences, get_haggling_test_sequences

class HagglingDataset(Dataset):

    def __init__(self, opt, split):
        super(HagglingDataset, self).__init__()
        
        self.split = split
        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.all_seqs = self.load_all()

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def load_all(self, frame_rate: int = 30):
        if self.split == "train":
            sequences = get_haggling_train_sequences()
        elif self.split == "test":
            sequences = get_haggling_test_sequences()

        all_motion_poses = []
        seq_len = self.input_n + self.output_n

        for sequence in sequences:
            poses_3person = sequence.Seq.reshape(-1, 3, 19, 3)
            N = len(poses_3person)
            if N < seq_len:
                continue

            if (N> seq_len):
                ids = np.arange(0, N - seq_len, seq_len)
                for i in ids:
                    all_motion_poses.append(poses_3person[i:i+seq_len, 0])
                    all_motion_poses.append(poses_3person[i:i+seq_len, 1])
                    all_motion_poses.append(poses_3person[i:i+seq_len, 2])

        return all_motion_poses
    
    def __getitem__(self, item):
        motion_poses = self.all_seqs[item]
        motion = torch.from_numpy(normalize(motion_poses, self.input_n - 1)).reshape(motion_poses.shape[0], -1)
        return motion