import argparse
import os
import json
import random

import librosa
import spafe.utils.preprocessing as preprocess
import spafe.features.mfcc as mfcc
import numpy as np
import soundfile as sf
from loguru import logger

from utils.process_audio import parse_vad_label

# 生成数据集
"""
如何生成数据集。
目前训练集有3000条数据，验证集有1100条数据。
拆分训练集为2500条数据，验证集500条数据，测试集1100条数据。

对训练集进行增强。往数据集中随机添加不同分贝的噪声。见过的噪声集来源于noise-92wav
1. 对于每一段音频，往其中几段中随机添加不同分贝的噪声。
2. 生成了新的音频文件后，需要同步修改标签文件。新生成的音频文件命名为originfile_noise.wav
3. 新生成的数据的文件夹划分为train、val、test。其中测试集又包含了seen_noise、unseen_noise。
4. 对于训练集和验证集，噪声集为noise-92wav。对于测试集，噪声来源为未见过的噪声集N101-N115 noises(raw, 16k, 16bit, mono)和见过的噪声集noise-92wav

"""


def delete_files(data_path):
    for filename in os.listdir(data_path):
        if filename.find('_') != -1:
            # delete
            os.remove(os.path.join(data_path, filename))


# 遍历文件夹的所有文件，并将路径写到文件中
def generate_list(data_path, out_path):
    with open(out_path, 'w') as f:
        for filename in os.listdir(data_path):
            path = os.path.join(data_path, filename)
            f.write(path)
            f.write("\n")


# 添加噪声
def add_noise(data, noise_data):
    data_len = len(data)
    noise_len = len(noise_data)
    ratio = data_len / noise_len
    # 如果超过了1倍，则随机选一段添加
    if ratio <= 1:
        start = random.randint(0, data_len)
        aug_data = np.concatenate((data[: start], data[start:] + noise_data[: data_len - start]))
    else:
        # 否则全部添加
        aug_data = data[: noise_len] + noise_data[:]
    return aug_data


# 生成数据集
def generate_data(data_path, noise_path, out_path, wav_sr=8000, noise_sr=8000):
    # 获取数据集
    # data_list = [line.strip() for line in open(data_path, 'r').readlines()]
    # 获取噪声集
    # 见过的噪声
    noises = {}
    for name in os.listdir(noise_path):
        path = os.path.join(noise_path, name)
        name = name.replace(".wav", "")
        data, sr = librosa.load(path, 8000)
        noises[name] = data

    noises_len = len(noises)
    seen_noises_keys = list(noises.keys())
    for name in os.listdir(data_path):
        filename = os.path.join(data_path, name)
        data, sr = librosa.load(filename, 8000)
        # 随机选取3种噪声加入到数据集中
        indices = list(range(0, noises_len))
        random.shuffle(indices)

        for idx in indices[:3]:
            save_file = os.path.join(out_path, name.replace(".wav", "") + "_" + seen_noises_keys[idx] + ".wav")
            # save_file = filename.replace(".wav", "") + "_" + seen_noises_keys[idx] + ".wav"
            noise_data = noises[seen_noises_keys[idx]]
            # 添加噪声
            aug_data = add_noise(data, noise_data)
            # 保存数据
            sf.write(save_file, aug_data, 8000)


# 增强数据后增加标签
def generate_labels(lbl_json_path, data_path, out_path):
    with open(lbl_json_path, 'r') as f:
        lbl_dict = json.load(f)

    out_lbl_dict = lbl_dict.copy()
    for filename in os.listdir(data_path):
        name = filename.replace(".wav", "")
        if name.find("_") != -1:
            # 说明是加了噪声的
            new_name = name.split("_")[0]
            out_lbl_dict[name] = lbl_dict[new_name]
    with open(out_path, 'w') as f:
        json.dump(out_lbl_dict, f)


def generate_feats(data_path, out_path):
    for filename in os.listdir(data_path):
        audio_path = os.path.join(data_path, filename)
        feats = extract_feature(audio_path)
        save_file = os.path.join(out_path, filename.replace(".wav", "") + ".npy")
        np.save(save_file, feats)


def extract_feature(audio_path, sr=8000, win_len=0.032, win_hop=0.008):
    audio_data, _ = librosa.load(audio_path, sr=sr)
    audio_framed, frame_len = preprocess.framing(audio_data, fs=sr, win_len=win_len,
                                                 win_hop=win_hop)
    frame_energy = (audio_framed ** 2).sum(1)[:, np.newaxis]
    frame_mfcc = mfcc.mfcc(audio_data, fs=sr, win_len=win_len, win_hop=win_hop)
    # 联结帧能量以及mfcc特征
    frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)

    return frame_feats


# def extract_feature(audio_path, label, sr=8000, win_len=0.032, win_hop=0.008):
#     audio_data, _ = librosa.load(audio_path, sr=sr)
#     audio_framed, frame_len = preprocess.framing(audio_data, fs=sr, win_len=win_len,
#                                                  win_hop=win_hop)
#     frame_num = audio_framed.shape[0]
#     # assert frame_num >= len(label), "frame_num : {}, len of labels : {}".format(frame_num, len(label))
#     if frame_num > len(label):
#         label += [0] * (frame_num - len(label))
#     else:
#         label = label[: frame_num]
#     frame_energy = (audio_framed ** 2).sum(1)[:, np.newaxis]
#     frame_mfcc = mfcc.mfcc(audio_data, fs=sr, win_len=win_len, win_hop=win_hop)
#     # 联结帧能量以及mfcc特征
#     frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
#     # 将特征 + 能量 保存到文件中
#     return label, frame_feats


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


def delete_generate_wav(path):
    for filename in os.listdir(path):
        if filename.find('_') != -1:
            # delete
            filepath = os.path.join(path, filename)
            os.remove(filepath)


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
    parser.add_argument('--data_path', default=r'F:\workspace\GHT\projects\vad\small_data', type=str, help='data path')
    parser.add_argument('--data_list', default=r'F:\workspace\GHT\projects\vad\small_data\labels\train_labels.txt')
    parser.add_argument('--val_list', default=r'F:\workspace\GHT\projects\vad\small_data\labels\val_labels.txt')
    parser.add_argument('--noise_path', default=r"F:\workspace\GHT\projects\vad\small_data\noise")

    args = parser.parse_args()
    data_path = args.data_path
    # 数据集存放路径
    train_path = os.path.join(data_path, "dataset", "train")
    val_path = os.path.join(data_path, "dataset", "val")
    test_path = os.path.join(data_path, "dataset", "test")

    # 提取的特征存放路径
    feat_path = os.path.join(data_path, "feat")
    train_feat_path = os.path.join(feat_path, "train")
    val_feat_path = os.path.join(feat_path, "val")
    test_feat_path = os.path.join(feat_path, "test")

    # 标签存放路径
    labels_path = os.path.join(data_path, 'labels')
    train_labels_path = os.path.join(labels_path, r'train_lbl_dict.json')
    val_labels_path = os.path.join(labels_path, r'val_lbl_dict.json')

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
    for path in [train_feat_path, val_feat_path, test_feat_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    # 删除生成的噪声文件
    # delete_generate_wav(train_path)
    # delete_generate_wav(val_path)

    # 生成数据集
    logger.info("start generate data")

    noise_path = args.noise_path
    seen_noise_path = os.path.join(noise_path, "noisex-92wav")
    unseen_noise_path = r""
    logger.info("start generate train dataset")
    generate_data(train_path, seen_noise_path, train_path)
    logger.info("start generate val dataset")
    generate_data(val_path, seen_noise_path, val_path)
    origin_test_path = os.path.join(test_path, "origin_dataset")
    seen_path = os.path.join(test_path, "seen_noise_dataset")
    unseen_path = os.path.join(test_path, "unseen_noise_dataset")
    generate_data(origin_test_path, seen_noise_path, seen_path)
    logger.info("generate done!")
    # TODO:暂时不考虑未见过的噪声
    # generate_data(origin_test_path, unseen_noise_path, unseen_path, noise_sr=16000)

    # 生成标签
    logger.info("start generate labels")
    new_train_labels_path = os.path.join(labels_path, "small_train_lbl_dict.json")
    new_val_labels_path = os.path.join(labels_path, "small_val_lbl_dict.json")
    generate_labels(train_labels_path, train_path, new_train_labels_path)
    generate_labels(val_labels_path, val_path, new_val_labels_path)
    logger.info("generate labels done!")
    #
    # # 生成特征
    logger.info("start generate feats")
    generate_feats(train_path, train_feat_path)
    generate_feats(val_path, val_feat_path)
    origin_test_path = os.path.join(test_path, "origin_dataset")
    origin_test_feat_path = os.path.join(test_feat_path, "origin_dataset")
    seen_test_path = os.path.join(test_path, "seen_noise_dataset")
    seen_test_feat_path = os.path.join(test_feat_path, "seen_noise_dataset")
    generate_feats(origin_test_path, origin_test_feat_path)
    generate_feats(seen_test_path, seen_test_feat_path)
    logger.info("generate feats done!")

    # 预处理数据集
    # pre_process_data(args.data_list, train_feat_path, train_labels_path)
    # pre_process_data(args.val_list, val_feat_path, val_labels_path)

    # 生成文件
    paths = [train_feat_path, val_feat_path]
    suffixs = ["train_feats.txt", "val_feats.txt"]
    for i in range(2):
        path = os.path.join(feat_path, suffixs[i])
        generate_list(paths[i], path)

