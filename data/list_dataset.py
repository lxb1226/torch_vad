import json

import numpy as np
import torch
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data_list, label_list):
        # usually we need args rather than single datalist to init the dataset
        super(ListDataset, self).__init__()

        infos = [line.strip() for line in open(data_list).readlines()]
        self.audio_paths = infos
        with open(label_list, 'r') as f:
            self.labels = json.load(f)

    @staticmethod
    def get_audio_id(audio_path) -> str:
        return audio_path.strip().split('\\')[-1].split('.')[0]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_data = np.load(self.audio_paths[idx])
        audio_path = self.audio_paths[idx]
        label = self.labels[self.get_audio_id(audio_path)]
        audio_data = torch.from_numpy(audio_data).float()
        label = torch.tensor(label).float()
        return audio_data, label
