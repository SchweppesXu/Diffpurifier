import sys
import os
import torch, gc
import torch
from thop import profile
from torchsummary import summary
import time
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import argparse
import random
import logging
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.utiltool import print_feats
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
from utils.helpers import get_test_loaders,load_model
from utils.parser import get_parser_with_args

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
        batch_size=2,
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
        batch_size=2,
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
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
"""
Initialize and define arguments
"""
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
ETA_POS = 2
GAMMA_CLR = 0.1
OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)

NUM_ITERS = 5
NUM_FILTERS = 32
NUM_FEATS_IN = 5
NUM_FEATS_OUT = 20

H = 256
W = 256
"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
dev = torch.device("cuda")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=7)
"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
total_step = -1
if __name__ == "__main__":

    """
    Load Dataset 
    """
    test_loader = get_test_loaders(opt)
    logger.info('Initial Dataset Finished')
    """
    Load Model 
    """
    model = load_model(opt, args_optical)
    checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0')
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()}, strict=True)
    model = model.to(dev)
    logging.info('LOADING Model')

    source_model, diffusion = read_model_and_diffusion(args_optical, opt.source_dir, dev, synthetic=False)
    for name, parameter in source_model.named_parameters(): parameter.requires_grad = False
    # source_model = DDP(source_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)

    target_model, diffusion_target = read_model_and_diffusion(args_sar, opt.target_dir, dev, synthetic=False)
    for name, parameter in target_model.named_parameters(): parameter.requires_grad = False
    # target_model = DDP(target_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)
    logger.info('Initial Diffusion Models Finished')
    # timed = []
    # costd = []
    model.eval()
    with torch.no_grad():
        tbar = tqdm(test_loader)
        for opt_img, sar_img,labels in tbar:
            opt_img = opt_img.to(dev)
            sar_img = sar_img.to(dev)
            source_model = source_model.to(dev)
            target_model = target_model.to(dev)

            pre_params = sum(p.numel() for p in source_model.parameters())
            print(f"预训练模型总参数量: {pre_params/1000000}M")
            # pre_flops, pre_params = profile(source_model, inputs=(opt_img,))
            # print('pre_flops: ', pre_flops, 'pre_params: ', pre_params)
            # flops, params = get_model_complexity_info(source_model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
            #
            # print('flops: ', flops, 'params: ', params)
            f_A = [0] * len(opt.feat_scales)
            f_B = [0] * len(opt.feat_scales)

            torch.cuda.synchronize()
            start = time.time()
            for t in opt.t:

                layer = 0
                fd_A_t = get_feats(diffusion=diffusion, diffusion_target=diffusion_target, t=t,
                                   source_model=source_model, target_model=target_model,
                                   img=opt_img, noise_type='optical')  # np.random.randint(low=2, high=8)
                fd_B_t = get_feats(diffusion=diffusion_target, diffusion_target=diffusion_target, t=t,
                                   source_model=target_model, target_model=target_model,
                                   img=sar_img, noise_type='sar')  # np.random.randint(low=2, high=8)
                # print(fd_A_t)
                for scale in opt.feat_scales:
                    # print_feats(path=opt.output_dir, feats_A=fd_A_t, feats_B=fd_B_t, scale=scale, t=t)

                    f_A[layer] += fd_A_t[scale]
                    f_B[layer] += fd_B_t[scale]
                    layer = layer + 1
                del fd_A_t, fd_B_t
            # print(img.size)
            # Feeding features from the diffusion model to the CD model
            cd_preds, prob_ds, (Q1, Q2), (f1, f2) = model(f_A, f_B, merge=False)

            torch.cuda.synchronize()
            end = time.time()
            print('infer_time:', end - start)

            params = sum(p.numel() for p in model.parameters())
            print(f"检测模型总参数量: {params / 1000000}M")
            # flops, params = profile(model, inputs=(f_A, f_B))
            # print('flops: ', flops, 'params: ', params)

            labels = labels.long().to(dev)

            _, cd_preds = torch.max(cd_preds, 1)  # 返回索引

            # Calculate and log other batch metrics

            pred = cd_preds.data.cpu().numpy().flatten()
            true = labels.data.cpu().numpy().flatten()

            pre_cut = np.delete(pred, np.where(true == -1)[0])
            true_cut = np.delete(true, np.where(true == -1)[0])

            tn, fp, fn, tp = confusion_matrix(true_cut,pre_cut, labels=[0,1]).ravel()
            del cd_preds,pred,pre_cut,true,true_cut,labels,prob_ds,Q1, Q2, f1, f2

            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (R + P)
    ACC=(tp + tn) / (tp + tn + fp + fn)
    print('ACC: {}\nRecall: {}\nF1-Score: {}\nPrecision: {}'.format(ACC, R, F1,P))
    print('Done!')
    # plt.plot(timed, costd)
    # plt.show()