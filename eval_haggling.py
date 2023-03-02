from model import stage_4
from utils.opt import Options
from tqdm import tqdm

from utils.haggling_eval import HagglingEvalDataset
from utils.haggling_utils import normalize, undo_normalization_to_seq

import torch
import numpy as np

option = Options.init_from_json("./checkpoint-haggling/main_haggling_3d_all_in10_out25_dctn35_dropout_0.3_lr_0.005_d_model_16/option.json")

model = stage_4.MultiStageModel(opt=option.opt)
model.to(option.opt.cuda_idx)
model.eval()

ckpt = torch.load(option.opt.ckpt + "/ckpt_best.pth.tar")
model.load_state_dict(ckpt["state_dict"])

sequences_in = HagglingEvalDataset(option.opt)

sequences_out = []
n_seqs = len(sequences_in) // 3

for i in tqdm(range(n_seqs)):
    sequence_out_3_persons = []
    for p in range(3):
        sequence = sequences_in[(3 * i) + p]   # 178 x 19 x 3
        normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
        for n_iter in range(1870 // option.opt.output_n + 1):
            seq_in_torch = torch.from_numpy(normalized_seq[-option.opt.input_n:].reshape(option.opt.input_n, -1)).unsqueeze(0).to(option.opt.cuda_idx)
            seq_out_torch = model(seq_in_torch)[0][0].cpu().detach()[option.opt.input_n:]
            seq_out = undo_normalization_to_seq(seq_out_torch.numpy().reshape(-1, 19, 3), normalization_params[0], normalization_params[1])
            sequence = np.concatenate((sequence, seq_out), 0)
            normalized_seq, normalization_params = normalize(sequence, -1, return_transform=True)
        sequence_out_3_persons.append(sequence)
    sequences_out.append(np.stack(sequence_out_3_persons, 1))

sequences_out = np.array(sequences_out)[:, :2048]
print(sequences_out.shape)
np.save("haggling_eval_pgbig.npy", sequences_out)