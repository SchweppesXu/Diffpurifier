import sys
import os
local_rank = int(os.environ["LOCAL_RANK"])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import datetime
import math
import torch, gc
from torchsummary import summary
import matplotlib.pyplot as plt
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import random
import logging
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix,roc_auc_score
from tensorboardX import SummaryWriter
import numpy as np
import json
from tqdm import tqdm
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,get_feats
)

from common import (read_model_and_diffusion,read_model_and_diffusion_ema)
from utils.helpers import (get_loaders, get_criterion,
                              load_model, initialize_metrics, get_mean_metrics,
                              set_metrics)
from utils.parser import get_parser_with_args, get_parser_with_args1

def create_argparser_cd():
    defaults = dict(
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.set_defaults(
        config='/data/yiquan.xu/ddib-main/metadata.json',
    )
    return parser
def create_argparser_optical():
    defaults = dict(
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.set_defaults(
        num_gpus=4,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        diffusion_steps=1000,
        noise_schedule='linear',
        lr=2e-5,
        batch_size=1,
        microbatchsize=-1,
        in_channels=3,
        learn_sigma=True,
        num_heads=1,
        num_head_channels=64,
        resblock_updown="True",
        use_scale_shift_norm="True",
        attention_resolutions="32,16,8",
        rescale_timesteps=True,
        timestep_respacing="ddim250",
        use_ddim=True
    )
    return parser
def create_argparser_sar():
    defaults = dict(
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.set_defaults(
        num_gpus=4,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        diffusion_steps=1000,
        noise_schedule='linear',
        lr=2e-5,
        batch_size=1,
        microbatchsize=-1,
        in_channels=3,
        learn_sigma=True,
        num_heads=1,
        num_head_channels=64,
        resblock_updown="True",
        use_scale_shift_norm="True",
        attention_resolutions="32,16,8",
        rescale_timesteps=True,
        timestep_respacing="ddim250",
        use_ddim=True
    )
    return parser

args_cd = create_argparser_cd().parse_args()
args_optical = create_argparser_optical().parse_args()
args_sar = create_argparser_sar().parse_args()
parser, metadata = get_parser_with_args1()
opt = parser.parse_args()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)
dev = torch.device("cuda", local_rank)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(seed=777)

criterion = get_criterion(opt, dev)
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
total_step = -1

if __name__ == "__main__":
    """
    Load Dataset and model
    """
    train_loader, val_loader,train_dataset,val_dataset = get_loaders(opt)
    # print(len(train_dataset))
    # print(len(val_dataset))
    logger.info('Initial Dataset Finished')

    model = load_model(opt, args_optical)

    if opt.resume:
        multi_checkpoint = torch.load(opt.checkpoint_path,map_location='cuda:{}'.format(local_rank))
        optimizer = torch.optim.AdamW(model.parameters(), lr=multi_checkpoint['optimizer']['param_groups'][0]['lr'], weight_decay=1)
        model.load_state_dict({k.replace('module.', ''): v for k, v in multi_checkpoint['model'].items()}, strict=True)
        optimizer.load_state_dict(multi_checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        # optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001, last_epoch=-1, verbose=False)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.5,last_epoch=-1)
        initepoch = multi_checkpoint['epoch'] + 1
        logger.info("====>loaded checkpoint (epoch{})".format(multi_checkpoint['epoch']))
        del multi_checkpoint
    else:
        logger.info("====>no checkpoint found.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=0.9)
        # optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,momentum=0.9, weight_decay=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)
        initepoch = 0


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(dev)
    # model = model.to(dev)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    logging.info('LOADING Model Finished')

    source_model, diffusion = read_model_and_diffusion(args_optical, opt.source_dir,dev, synthetic=False)
    for name, parameter in source_model.named_parameters():parameter.requires_grad = False
    # source_model = DDP(source_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)
    target_model, diffusion_target = read_model_and_diffusion(args_sar, opt.target_dir,dev, synthetic=False)
    for name, parameter in target_model.named_parameters():parameter.requires_grad = False
    # target_model = DDP(target_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)
    logger.info('Initial Diffusion Models Finished')
    train_writer = SummaryWriter(os.path.join(opt.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(opt.log_dir, 'val'))

    for epoch in range(initepoch, opt.epochs):
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()
        gc.collect()
        torch.cuda.empty_cache()
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        """
        Begin Training
        """
        model.train()
        logging.info('SET model mode to train!')
        batch_iter = 0
        tbar = tqdm(train_loader)
        loss=A_TP = A_TN = A_FP = A_FN = 0
        for opt_img, sar_img ,labels in tbar:
            tbar.set_description(
                "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + opt.batch_size))
            batch_iter = batch_iter + opt.batch_size
            total_step += 1
            opt_img = opt_img.to(dev)
            sar_img = sar_img.to(dev)
            source_model = source_model.to(dev)
            target_model = target_model.to(dev)

            f_A = [0] * len(opt.feat_scales)
            f_B = [0] * len(opt.feat_scales)
            for t in opt.t:
                layer = 0
                fd_A_t = get_feats(diffusion=diffusion,diffusion_target=diffusion_target,t=t, source_model=source_model,target_model=target_model,
                                                       img=opt_img,noise_type='optical')  # np.random.randint(low=2, high=8)
                fd_B_t = get_feats(diffusion=diffusion_target,diffusion_target=diffusion_target,t=t, source_model=target_model,target_model=target_model,
                                                       img=sar_img,noise_type='sar')  # np.random.randint(low=2, high=8)
                # print(fd_A_t)
                for scale in opt.feat_scales:
                    # print_feats(path=opt.output_dir, feats_A=fd_A_t, feats_B=fd_B_t, scale=scale, t=t)

                    f_A[layer] += fd_A_t[scale]
                    f_B[layer] += fd_B_t[scale]
                    layer = layer + 1
                del fd_A_t, fd_B_t

            source_model = source_model.to('cpu')
            target_model = target_model.to('cpu')

            optimizer.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()

            # print(f_A.shape)

            cd_preds, prob_ds, (Q1, Q2), (f1, f2) = model(f1=f_A, f2=f_B, merge=False)

            del f_A, f_B
            labels = labels.long().to(dev)
            loss = criterion(cd_preds, labels)
            loss.backward()
            optimizer.step()
            print(loss)
            # cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)  # 返回索引

            # Calculate and log other batch metrics
            pred = cd_preds.data.cpu().numpy().flatten()
            true = labels.data.cpu().numpy().flatten()
            loss+=loss
            A_TP += ((pred == 1)  & (true == 1) ).sum()  + 0.0001
            # TN    predict 和 label 同时为0
            A_TN += ((pred == 0)  & (true == 0) ).sum()  + 0.0001
            # FN    predict 0 label 1
            A_FN += ((pred == 0)  & (true == 1) ).sum()  + 0.0001
            # FP    predict 1 label 0
            A_FP += ((pred == 1)  & (true == 0) ).sum()  + 0.0001

        precision_train = A_TP / (A_TP + A_FP)
        recall_train = A_TP / (A_TP + A_FN)
        f1_train = 2 * recall_train * precision_train / (recall_train + precision_train)
        accuracy_score_train = (A_TP + A_TN) / (A_TP + A_TN + A_FP + A_FN)

        train_metrics = set_metrics(train_metrics,
                                    loss/len(train_dataset),
                                    accuracy_score_train,
                                    precision_train,
                                    recall_train,
                                    f1_train, optimizer.state_dict()['param_groups'][0]['lr'])

        mean_train_metrics = get_mean_metrics(train_metrics)
        for k, v in mean_train_metrics.items():
            train_writer.add_scalar(k, v, epoch)
        scheduler.step()
        if dist.get_rank() == 0:
            logging.info("epoch {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """
        Begin Validation
        """
        model.eval()
        loss = A_TP = A_TN = A_FP = A_FN = 0
        with torch.no_grad():
            for opt_img, sar_img,labels in val_loader:
                opt_img = opt_img.to(dev)
                sar_img = sar_img.to(dev)
                source_model = source_model.to(dev)
                target_model = target_model.to(dev)
                f_A = [0] * len(opt.feat_scales)
                f_B = [0] * len(opt.feat_scales)
                for t in opt.t:
                    layer = 0
                    fd_A_t = get_feats(diffusion=diffusion, diffusion_target=diffusion_target, t=t,
                                       source_model=source_model, target_model=target_model,
                                       img=opt_img, noise_type='optical')
                    fd_B_t = get_feats(diffusion=diffusion_target, diffusion_target=diffusion_target, t=t,
                                       source_model=target_model, target_model=target_model,
                                       img=sar_img, noise_type='sar')
                    # print(fd_A_t)
                    for scale in opt.feat_scales:
                        f_A[layer] += fd_A_t[scale]
                        f_B[layer] += fd_B_t[scale]
                        layer = layer + 1
                    del fd_A_t, fd_B_t
                source_model = source_model.to('cpu')
                target_model = target_model.to('cpu')
                cd_preds, prob_ds, (Q1, Q2), (f1, f2) = model(f_A, f_B, merge=False)
                del f_A, f_B

                labels = labels.long().to(dev)
                loss = criterion(cd_preds, labels)
                loss += loss
                _, cd_preds = torch.max(cd_preds, 1)  # 返回索引
                pred = cd_preds.data.cpu().numpy().flatten()
                true = labels.data.cpu().numpy().flatten()
                A_TP += ((pred == 1) & (true == 1) ).sum()  + 0.0001
                A_TN += ((pred == 0)  & (true == 0) ).sum()  + 0.0001
                A_FN += ((pred == 0)  & (true == 1) ).sum()  + 0.0001
                A_FP += ((pred == 1)  & (true == 0) ).sum()  + 0.0001

            precision_val = A_TP / (A_TP + A_FP)
            recall_val = A_TP / (A_TP + A_FN)
            f1_val = 2 * recall_val * precision_val / (recall_val + precision_val)
            accuracy_score_val = (A_TP + A_TN) / (A_TP + A_TN + A_FP + A_FN)

            del pred,true,labels

            val_metrics = set_metrics(val_metrics,
                                      loss/len(val_dataset),
                                      accuracy_score_val,
                                      precision_val,
                                      recall_val,
                                      f1_val, optimizer.state_dict()['param_groups'][0]['lr'])
            mean_val_metrics = get_mean_metrics(val_metrics)
            del opt_img, sar_img
            for k, v in mean_val_metrics.items():
                val_writer.add_scalar(k, v, epoch)
            if dist.get_rank() == 0:
                logging.info("epoch {} VALIDATION METRICS".format(epoch) + str(mean_val_metrics))
            if mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']:
                metadata['validation_metrics'] = mean_val_metrics
                best_metrics = mean_val_metrics
                state = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, opt.weight_dir + '/re_k2_checkpoint_epoch_' + str(epoch) + '.pt')
    logger.info('Done!')
    train_writer.close()
    val_writer.close()
    destroy_process_group()