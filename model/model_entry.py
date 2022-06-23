from model.base.fcn import DnnVAD, LstmVAD, RnnVAD
import torch.nn as nn


def select_model(args):
    type2model = {
        'dnn_vad': DnnVAD(),
        'rnn_vad': RnnVAD(input_dim=args.input_dim, hidden_size=args.hidden_size, use_gpu=args.use_gpu),
        'lstm_vad': LstmVAD(input_dim=args.input_dim, hidden_size=args.hidden_size, num_layers=args.num_layers,
                            use_gpu=args.use_gpu)
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
