import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--model_type', type=str, default='rnn_vad', help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='list', help='used in data_entry.py')
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='model path for pretrain or test')
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--val_list', type=str, default=r'F:\workspace\GHT\projects\vad\data\feat\val_feats.txt',
                        help='val list in train, test list path in test')
    parser.add_argument('--val_label_list', type=str, default=r'F:\workspace\GHT\projects\vad\data\labels'
                                                                r'\val_lbl_dict.json')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--win_len', default=10, type=int, help='FFT duration in ms')
    parser.add_argument('--win_hop', default=5, type=int, help='hop duration in ms')
    parser.add_argument('--n_fft', default=2048, type=int)
    parser.add_argument('--n_mels', default=12, type=int)
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--input_dim', type=int, default=14)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--loss', type=str, default='ce', help='loss function')
    parser.add_argument('--train_list', type=str, default=r'F:\workspace\GHT\projects\vad\data\feat\train_feats.txt')
    parser.add_argument('--train_label_list', type=str, default=r'F:\workspace\GHT\projects\vad\data\labels'
                                                                r'\train_lbl_dict.json')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)

    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        # os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        # os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
