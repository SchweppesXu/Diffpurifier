import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from utils.dataloaders import (full_path_loader, full_test_loader, CDDloader)
from utils.metrics import jaccard_loss, dice_loss, FocalLoss
# from utils.losses import hybrid_loss
from models.Models import Siam_NestedUNet_Conc,SESNUNet, SESNUNet_ablation,SNUNet_ECAM
# from models.siamunet_dif import SiamUnet_diff
from utils.utiltool import FeatureConverter, rgb_to_xylab

logging.basicConfig(level=logging.INFO)

ETA_POS = 2
GAMMA_CLR = 0.1
OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)

NUM_ITERS = 5
NUM_FILTERS = 32
NUM_FEATS_IN = 5
NUM_FEATS_OUT = 20

H = 256
W = 256



def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_auc_scores': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }
    return metrics

def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values

    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics

    Returns
    -------
    dict
        dict of floats that reflect mean metric value

    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, auc_scores, precision,recall,f1,lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_auc_scores'].append(auc_scores.item())
    metric_dict['cd_precisions'].append(precision.item())
    metric_dict['cd_recalls'].append(recall.item())
    metric_dict['cd_f1scores'].append(f1.item())
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, auc_scores, precision,recall,f1):

    metric_dict['cd_auc_scores'].append(auc_scores.item())
    metric_dict['cd_precisions'].append(precision.item())
    metric_dict['cd_recalls'].append(recall.item())
    metric_dict['cd_f1scores'].append(f1.item())

    return metric_dict


def get_loaders(opt):


    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=int(opt.batch_size/opt.num_gpus),
                                               sampler=train_sampler,
                                               num_workers=opt.num_workers,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=int(opt.valbatch_size/opt.num_gpus),
                                             sampler=val_sampler,
                                             num_workers=opt.num_workers,drop_last=False)
    return train_loader, val_loader,train_dataset,val_dataset

def get_test_loaders(opt):

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = CDDloader(test_full_load, aug=False)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=int(opt.valbatch_size/opt.num_gpus),
                                             shuffle=False,
                                             num_workers=opt.num_workers,drop_last=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                          batch_size=int(opt.valbatch_size / opt.num_gpus),
    #                                          sampler=test_sampler,
    #                                          num_workers=opt.num_workers, drop_last=False)

    return test_loader

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)


    # for prediction in predictions:
    #     bce = focal(prediction, target)
    #     dice = dice_loss(prediction, target)
    #     loss += bce + dice
    bce = focal(predictions, target)
    dice = dice_loss(predictions, target)
    loss += bce + dice

    return loss

def get_criterion(opt,dev):
    """get the user selected loss function

    Parameters
    ----------
    opt : dict
        Dictionary of options/flags

    Returns
    -------
    method
        loss function

    """
    if opt.loss_function == 'hybrid':
        criterion = hybrid_loss
    if opt.loss_function == 'focal':
        criterion = FocalLoss(gamma=2, alpha=0.25)
    if opt.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1]).to(dev),ignore_index=-1)
    if opt.loss_function == 'dice':
        criterion = dice_loss
    if opt.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_model(opt, args_optical):
    """Load the model

    Parameters
    ----------
    opt : dict
        User specified flags/options
    device : string
        device on which to train model

    """
    diffusion_model_opt = args_optical

    model = SESNUNet(
        NUM_ITERS,
        n_spixels=opt.num_spixels,
        out_channels=opt.out_channels,
        inner_channel=diffusion_model_opt.num_channels,
        channel_multiplier=(1, 1, 2, 2, 4, 4)
    )
    return model
