import argparse

import librosa
import torch

from model.model_entry import select_model
from utils.preprocess import extract_feature
from utils.torch_utils import load_match_dict
from utils.viz import show_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str, default=r'F:\workspace\GHT\projects\vad\code\torch_vad\wavs'
                                                        r'\vad_test_src.wav')
    parser.add_argument('--model_type', type=str, default='dnn_vad')
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--model_type', type=str, default='dnn_vad')
    parser.add_argument('--input_dim', type=int, default=14)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--load_model_path', type=str,
                        default=r'F:\workspace\GHT\projects\vad\code\torch_vad\checkpoints\dnn_vad_pref\19_000000.pth')

    args = parser.parse_args()
    wav_path = args.wav_path
    sr = args.sample_rate
    data, _ = librosa.load(wav_path, sr=sr)
    feats = extract_feature(data, sr=sr)

    # 加载模型
    model = select_model(args)
    if args.load_model_path != '':
        print("=> using pre-trained weights for VAD")
        load_match_dict(model, args.load_model_path)
        # if args.load_not_strict:
        #     load_match_dict(model, args.load_model_path)
        # else:
        #     model.load_state_dict(torch.load(args.load_model_path).state_dict())
    # 预测
    inp = torch.from_numpy(feats).float()
    preds = model(inp)
    label = torch.argmax(preds, dim=1)
    print(preds.size())
    print(label.size())
    print(label)

    # 画图
    show_result(data, label, sr)
