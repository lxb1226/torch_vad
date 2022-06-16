import os

from multiprocessing import Pool

import librosa
import numpy as np
from spafe.features import mfcc
import spafe.utils.preprocessing as preprocess


def extract_feature(audio_path, sr=8000, win_len=0.032, win_hop=0.008):
    audio_data, _ = librosa.load(audio_path, sr=sr)
    audio_framed, frame_len = preprocess.framing(audio_data, fs=sr, win_len=win_len,
                                                 win_hop=win_hop)
    frame_energy = (audio_framed ** 2).sum(1)[:, np.newaxis]
    frame_mfcc = mfcc.mfcc(audio_data, fs=sr, win_len=win_len, win_hop=win_hop)
    # 联结帧能量以及mfcc特征
    frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)

    return frame_feats


def generate_feat(audio_path, out_path):
    feats = extract_feature(audio_path)
    filename = audio_path.split('\\')[-1]
    save_file = os.path.join(out_path, filename.replace(".wav", "") + ".npy")
    np.save(save_file, feats)


if __name__ == '__main__':
    test_path = r"F:\workspace\GHT\projects\vad\data\dataset\test"
    test_feat_path = r"F:\workspace\GHT\projects\vad\data\feat\test"
    origin_test_path = os.path.join(test_path, "origin_dataset")
    origin_test_feat_path = os.path.join(test_feat_path, "origin_dataset")

    filenames = [os.path.join(origin_test_path, name) for name in os.listdir(origin_test_path)]
    pool = Pool()
    pool.map(generate_feat, iter(filenames), origin_test_feat_path)
    pool.close()
    pool.join()
