from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import os, glob
import torch
from tqdm import tqdm
from utils.bam_utils import unknown_pose_shape_to_known_shape, normalize


class BAM_Motion3D(Dataset):

    def __init__(self, opt, split):
        super(BAM_Motion3D, self).__init__()
        self.path_to_data = opt.data_dir
        self.input_n = opt.input_n
        self.output_n = opt.output_n

        self.split = split

        self.bam_file_names = self.get_bam_names()

        self.all_seqs = self.load_all()

    def get_bam_names(self):
        seq_names = []
        if self.split == 0:
            seq_names += np.loadtxt(
                os.path.join(self.path_to_data, "bam_train.txt"), dtype=str, ndmin=1
                ).tolist()
            self.is_test = False
        else:
            seq_names += np.loadtxt(
                os.path.join(self.path_to_data, "bam_test.txt"), dtype=str, ndmin=1
                ).tolist()
            self.is_test = True

        file_list = []
        for dataset in seq_names:
            files = glob.glob(self.path_to_data + '/' + dataset + '/poses_pid*.npy')
            file_list.extend(files)

        return file_list

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def load_all(self, frame_rate: int = 25):
        sampled_seq = []
        seq_len = self.input_n + self.output_n
        for bam_motion_name in tqdm(self.bam_file_names):
            bam_motion_poses = unknown_pose_shape_to_known_shape(np.load(bam_motion_name)) # (N x 18 x 3)
            bam_motion_poses = bam_motion_poses[:, :17]
            N = len(bam_motion_poses)
            if N < seq_len:
                continue
            
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            bam_motion_poses = bam_motion_poses[sampled_index]

            num_frames = len(bam_motion_poses)
            fs = np.arange(0, num_frames - seq_len + 1, 20)
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))
            fs_sel = fs_sel.T
            seq_sel = bam_motion_poses[fs_sel, :]
            if len(sampled_seq) == 0:
                sampled_seq = seq_sel
            else:
                sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
        print(sampled_seq.shape)
        return sampled_seq
    
    def __getitem__(self, item):
        bam_motion_poses = self.all_seqs[item]
        bam_motion = normalize(bam_motion_poses, self.input_n - 1).reshape(bam_motion_poses.shape[0], -1)
        return bam_motion