import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from loguru import logger

from data.data_entry import select_train_loader, select_eval_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict
from metrics import compute_auc, compute_accuracy, compute_recall, compute_f1, compute_precision, compute_eer, \
    compute_fpr_and_tpr


def gen_imgs_to_write(img, pred, label, is_train):
    # override this method according to your visualization
    prefix = 'train/' if is_train else 'val/'
    return {
        prefix + 'img': img[0],
        prefix + 'pred': pred[0],
        prefix + 'label': label[0]
    }


# TODO(heyjude): 补充需要计算的评估指标，补充输入输出的维度
def compute_metrics(pred, target, is_train):
    """
    :param pred: [seq_len, output_dim]
    :param gt:  [seq_len]
    :param is_train:
    :return:
    """
    loss = torch.nn.functional.cross_entropy(pred, target.long())

    prefix = 'train/' if is_train else 'val/'
    pred = torch.argmax(pred, dim=1).tolist()
    target = target.tolist()
    metrics = {
        prefix + 'ce': loss,
        prefix + 'accu': compute_accuracy(target, pred),
        prefix + 'recall': compute_recall(target, pred),
        prefix + 'prec': compute_precision(target, pred),
        prefix + 'auc': compute_auc(target, pred),
        prefix + 'f1': compute_f1(target, pred),
    }
    return metrics


class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = select_model(args)

        if args.load_model_path != '':
            print("=> using pre-trained weights for VAD")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay)

    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()
        logger.debug('len of train data : {}'.format(len(self.train_loader)))
        for i, data in enumerate(self.train_loader):

            audio, pred, label = self.step(data)

            # compute loss
            pred = pred.squeeze(dim=0)
            label = label.squeeze(dim=0)
            metrics = compute_metrics(pred, label, is_train=True)

            # get the item for backward
            loss = metrics['train/ce']

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # only save img at first step
            # if i == len(self.train_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)

            # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {} batch {} Loss {}'.format(epoch, i, loss))

    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            audio, pred, label = self.step(data)

            pred = pred.squeeze(dim=0)
            label = label.squeeze(dim=0)
            metrics = compute_metrics(pred, label, is_train=False)

            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # if i == len(self.val_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)

    def step(self, data):
        """
        :param data:
        :return:
            audio: [batch_size, seq_len, input_dim]
            label: [batch_size, seq_len]
            pred: [batch_size, seq_len, output_dim]
        """
        audio, label = data
        audio = audio.to(self.device)
        label = label.to(self.device)
        # compute output
        pred = self.model(audio)
        pred = pred.to(self.device)
        return audio, pred, label

    def compute_loss(self, pred, gt):
        if self.args.loss == 'l1':
            loss = (pred - gt).abs().mean()
        elif self.args.loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, gt)
        else:
            loss = torch.nn.functional.mse_loss(pred, gt)
        return loss


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
