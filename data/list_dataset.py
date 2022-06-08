import librosa
import spafe.features.mfcc as mfcc
import spafe.utils.preprocessing as preprocess
from torch.utils.data import Dataset

from utils.process_audio import parse_vad_label


class ListDataset(Dataset):
    def __init__(self, args, is_train):
        # usually we need args rather than single datalist to init the dataset
        super(self, ListDataset).__init__()
        if is_train:
            data_list = args.train_list
        else:
            data_list = args.val_list
        self.sr = args.sample_rate
        self.win_len = args.win_len
        self.n_mels = args.n_mels
        self.win_hop = args.win_hop

        # self.mel_args = {
        #     'n_mels': args.n_mels,
        #     'n_fft': args.n_fft,
        #     'hop_length': int(args.sr * args.hop_len / 1000),
        #     'win_length': int(args.sr * args.win_len / 1000)
        # }
        infos = [line.split() for line in open(data_list).readlines()]
        self.audio_paths = [info[0] for info in infos]
        self.audio_labels = [info[1:] for info in infos]

    def preprocess(self, audio, label):
        # you can add other process method or augment here
        # 分帧
        audio_framed, frame_len = preprocess.frameing(audio, fs=self.sr, win_len=self.win_len, win_hop=self.win_hop)
        # 提取mfcc特征 前[0:n_mels]个特征点
        features = mfcc.mfcc(audio_framed, fs=self.sr, win_len=self.win_len, win_hop=self.win_hop)[: self.n_mels]
        # 提取label
        labels = parse_vad_label(label)
        return features, labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        data, _ = librosa.load(self.audio_paths[idx], sr=8000)
        label = self.audio_labels[idx]
        data, labels = self.preprocess(data, label)
        return data, label
