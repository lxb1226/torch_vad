import argparse

from data.list_dataset import ListDataset
# from data.mem_list_dataset import MemListDataset
from torch.utils.data import DataLoader

from loguru import logger


def get_dataset_by_type(args, is_train=False):
    type2data = {
        'list': ListDataset(args, is_train),
        # 'mem_list': MemListDataset(args, is_train)
    }
    dataset = type2data[args.data_type]
    return dataset


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, True)
    # print('{} samples found in train'.format(len(train_dataset)))
    logger.info('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    return train_loader


def select_eval_loader(args):
    eval_dataset = get_dataset_by_type(args)
    # print('{} samples found in val'.format(len(eval_dataset)))
    logger.info('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset, 1, shuffle=False)
    return val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='dnn_vad', help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='list', help='used in data_entry.py')
    parser.add_argument('--train_list', type=str, default='../../../data/labels/train_labels.txt')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--win_len', default=10, type=int, help='FFT duration in ms')
    parser.add_argument('--win_hop', default=5, type=int, help='hop duration in ms')
    parser.add_argument('--n_fft', default=2048, type=int)
    parser.add_argument('--n_mels', default=12, type=int)

    args = parser.parse_args()
    train_loader = select_train_loader(args)
    for i, data in enumerate(train_loader):
        print(i, data)


