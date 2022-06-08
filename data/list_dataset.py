from torch.utils.data import Dataset
from utils.process_audio import convert_times_to_labels, extract_feature
import numpy as np
import librosa


class ListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super(self, ListDataset).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list

        self.mel_args = {
            'n_mels': args.n_mels,
            'n_fft': args.n_fft,
            'hop_length': int(args.sr * args.hop_len / 1000),
            'win_length': int(args.sr * args.win_len / 1000)
        }
        infos = [line.split() for line in open(data_list).readlines()]
        self.audio_paths = [info[0] for info in infos]
        self.audio_labels = [info[1:] for info in infos]

    def preprocess(self, audio, label):
        # you can add other process method or augment here
        features = extract_feature(audio, mel_args = self.mel_args)
        # labels = convert_times_to_labels(label)
        labels = [int(num) for num in label]
        return features, labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        data, _ = librosa.load(self.audio_paths[idx], sr=8000)
        label = self.audio_labels[idx]
        data, labels = self.preprocess(data, label)
        return data, label
