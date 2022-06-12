import argparse
import os
import json

import librosa
import spafe.utils.preprocessing as preprocess
import spafe.features.mfcc as mfcc
import numpy as np

from utils.process_audio import parse_vad_label


def extract_feature(audio_path, label, sr=8000, win_len=0.032, win_hop=0.008):
    audio_data, _ = librosa.load(audio_path, sr=sr)
    audio_framed, frame_len = preprocess.framing(audio_data, fs=sr, win_len=win_len,
                                                 win_hop=win_hop)
    frame_num = audio_framed.shape[0]
    # assert frame_num >= len(label), "frame_num : {}, len of labels : {}".format(frame_num, len(label))
    if frame_num > len(label):
        label += [0] * (frame_num - len(label))
    else:
        label = label[: frame_num]
    frame_energy = (audio_framed ** 2).sum(1)[:, np.newaxis]
    frame_mfcc = mfcc.mfcc(audio_data, fs=sr, win_len=win_len, win_hop=win_hop)
    # 联结帧能量以及mfcc特征
    frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
    # 将特征 + 能量 保存到文件中
    return label, frame_feats


def pre_process_data(data_list, feat_path, json_path):
    lbl_dict = {}
    with open(data_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            audio_path = data[0].strip()
            label = ' '.join(data[1:])
            # logger.debug('label : {}'.format(label))
            label = parse_vad_label(label)
            label, frame_feats = extract_feature(audio_path, label)
            audio_id = audio_path.strip().split('\\')[-1].split('.')[0]
            # logger.debug("audio_id : {}, label : {}".format(audio_id, label))
            np.save(os.path.join(feat_path, audio_id + '.npy'), frame_feats)
            lbl_dict[audio_path] = label
    # 保存到json文件中
    json_str = json.dumps(lbl_dict)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run LSTM for VAD')
    parser.add_argument('--sample_rate', default=8000, type=int)
    parser.add_argument('--win_len', default=0.032, type=float)
    parser.add_argument('--win_hop', default=0.008, type=float)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--report_interval', default=50, type=int)
    parser.add_argument('--stage', default=1, type=int)
    parser.add_argument('--L', default=5, type=int)  # adjust length in VACC calculation
    parser.add_argument('--model', default='VADNet', type=str)
    parser.add_argument('--data_path', default=r'F:\workspace\GHT\projects\vad\data', type=str, help='data path')
    parser.add_argument('--data_list', default=r'F:\workspace\GHT\projects\vad\data\labels\train_labels.txt')
    parser.add_argument('--val_list', default=r'F:\workspace\GHT\projects\vad\data\labels\val_labels.txt')

    args = parser.parse_args()
    data_path = args.data_path
    # 数据集存放路径
    train_path = os.path.join(data_path, "dataset", "train")
    val_path = os.path.join(data_path, "dataset", "val")

    # 提取的特征存放路径
    feat_path = os.path.join(data_path, "feat")
    train_feat_path = os.path.join(feat_path, "train")
    val_feat_path = os.path.join(feat_path, "val")

    # 标签存放路径
    labels_path = os.path.join(data_path, 'labels')
    train_labels_path = os.path.join(labels_path, r'train_lbl_dict.json')
    val_labels_path = os.path.join(labels_path, r'val_lbl_dict.json')

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
    for path in [train_feat_path, val_feat_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    # 预处理数据集
    pre_process_data(args.data_list, train_feat_path, train_labels_path)
    pre_process_data(args.val_list, val_feat_path, val_labels_path)
