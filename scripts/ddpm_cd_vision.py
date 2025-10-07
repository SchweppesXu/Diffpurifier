import sys
import os
import torch, gc
import matplotlib.pyplot as plt
import argparse
import random
import logging
from skimage.segmentation._slic import _enforce_label_connectivity_cython as enforce_connectivity
from skimage.segmentation import mark_boundaries
from skimage import color
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
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

def mark_boundaries_on_image(Q, ops, im,opt):
    idx_map = ops['map_idx'](torch.argmax(Q, 1, True).int())
    idx_map = idx_map[0,0].cpu().numpy()
    segment_size = H*W / opt.num_spixels
    min_size = int(0.06 * segment_size)
    max_size = int(3 * segment_size)
    idx_map = enforce_connectivity(idx_map[...,None].astype(np.intp), min_size, max_size)
    idx_map=idx_map.astype('int32')
    print(im.shape)
    bdy = mark_boundaries(im, idx_map[...,0], color=(1,1,1))
    return bdy,idx_map[...,0]

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)
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
    source_model, diffusion = read_model_and_diffusion(args_optical, opt.source_dir,dev, synthetic=False)
    for name, parameter in source_model.named_parameters():parameter.requires_grad = False
    # source_model = DDP(source_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)

    target_model, diffusion_target = read_model_and_diffusion(args_sar, opt.target_dir,dev, synthetic=False)
    for name, parameter in target_model.named_parameters():parameter.requires_grad = False
    # target_model = DDP(target_model, device_ids=[local_rank],output_device=local_rank, find_unused_parameters=True)
    logger.info('Initial Diffusion Models Finished')
    model.eval()
    index_img = 1
    with torch.no_grad():
        tbar = tqdm(test_loader)
        # for opt_img, sar_img, opt_origimg,labels in tbar:
        for opt_img, sar_img, labels in tbar:
            opt_img = opt_img.to(dev)
            sar_img = sar_img.to(dev)

            f_A = [0] * len(opt.feat_scales)
            f_B = [0] * len(opt.feat_scales)
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
            # Feeding features from the diffusion model to the CD model
            cd_preds, prob_ds, (Q1, Q2),(ops1, ops2), (f1, f2) = model(f_A, f_B, merge=False)

            labels = labels.long().to(dev)

            _, cd_preds = torch.max(cd_preds, 1)

            cd_preds = cd_preds.data.cpu().numpy().astype(np.float64)
            truth = labels.data.cpu().numpy()

            # truth = np.where(truth == -1, 0.5, truth)
            # cd_preds[np.where(truth == 0.5)] = 0.5
            result = cd_preds
            result = np.where((truth == 1) & (cd_preds == 1), 1, result)
            result = np.where((truth == 0) & (cd_preds == 0), 0, result)
            result = np.where((truth == 0) & (cd_preds == 1), 0, result)
            result = np.where((truth == 1) & (cd_preds == 0), 0, result)
            conf_map = np.concatenate(
                [
                    truth * 255,
                    cd_preds * 255,
                    result * 255
                ],
                axis=0,
            )

            # cd_preds = cd_preds.squeeze() * 255
            conf_map = conf_map.transpose(1, 2, 0)
            sar_img = (sar_img + 1) * 127.5
            # opt_origimg = (opt_origimg + 1) * 127.5
            # bdy1, mask1 = mark_boundaries_on_image(Q1, ops1, opt_origimg.permute(2, 3, 1, 0).squeeze().cpu().numpy(),opt)
            # bdy2, mask2 = mark_boundaries_on_image(Q2, ops2, sar_img.permute(2, 3, 1, 0).squeeze().cpu().numpy())
            # bdy1 = bdy1.squeeze()
            # bdy2 = bdy2.squeeze()
            del cd_preds,labels,prob_ds,Q1, Q2,ops1, ops2, f1, f2
            # plt.figure(1)
            # plt.imshow(bdy1.astype('uint8'))
            # plt.show()
            # plt.figure(2)
            # plt.imshow(bdy2[:, :, 0].astype('uint8'), cmap='gray')
            plt.show()
            plt.figure(3)
            plt.imshow(conf_map.astype('uint8'), cmap='gray')
            plt.show()
            # plt.imshow(mask1.astype('uint8'))
            # plt.show()
            if not os.path.exists(opt.output_dir):
                os.mkdir(opt.output_dir)
            file_path1 = opt.output_dir + '/'+ str(index_img).zfill(5)
            # file_path2 = '/data/yiquan.xu/ddib-main/ddib-snunet/tmp/lr0.001zheng/img1_mask/' + str(index_img).zfill(5)
            # file_path3 = opt.mask_dir + str(index_img).zfill(5)
            cv2.imwrite(file_path1 + '.png', conf_map[..., ::-1])
            # print(bdy1[..., ::-1].shape)
            # cv2.imwrite(file_path1 + '_slic.png', bdy1[..., ::-1].astype('uint8'))
            # cv2.imwrite(file_path3 + '.png', bdy2[..., ::-1])
            index_img += 1
