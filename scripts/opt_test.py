import argparse
import sys
import os
import shutil
from pathlib import Path

import numpy as np
import torch as th
device = 1
th.cuda.set_device(device)

import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_source_data_for_domain_translation,
    get_image_filenames_for_label
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from common import read_model_and_diffusion
from guided_diffusion.image_datasets import load_data


def main():

    if not os.path.exists('/data/yiquan.xu/DATASETS/toulouse/translate'):
        os.mkdir('/data/yiquan.xu/DATASETS/toulouse/translate')

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log(f"reading models")
    source_dir = "/data/yiquan.xu/DATASETS/toulouse/ddpm_opt_model"
    source_model, diffusion = read_model_and_diffusion(args, source_dir,synthetic=False)

    target_dir = "/data/yiquan.xu/DATASETS/toulouse/ddpm_opt_model"
    target_model, _ = read_model_and_diffusion(args, target_dir,synthetic=False)

    logger.log("running image translation...")
    data = load_source_data_for_domain_translation(
        batch_size=args.batch_size,
        image_size=args.image_size,
        in_channels=3,
        data_dir=args.data_dir,
        # use_lab_space=True
        use_yuv_space=True
    )

    for i, (batch, extra) in enumerate(data):
        batch = batch.to(dist_util.dev())
        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise = diffusion.ddim_reverse_sample_loop(
            source_model,batch,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        # Next, decode the latents to the target class.
        sample = diffusion.ddim_sample_loop(
            target_model,(args.batch_size,3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=False,
            device=dist_util.dev(),
            eta=args.eta
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #sample = (sample+127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        images = []
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(images) * args.batch_size} samples")

        logger.log("saving translated images.")
        images = np.concatenate(images, axis=0)

        for index in range(images.shape[0]):
            filepath = os.path.join("/data/yiquan.xu/DATASETS/toulouse/translate", f"{index+1}.png")
            image = Image.fromarray(images[index].squeeze())
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

    dist.barrier()
    logger.log(f"domain translation complete")

def create_argparser():
    defaults = dict(
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.set_defaults(
        # image_size=256,
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
        # rescale_learned_sigmas="False",
        # rescale_timesteps="False",
        data_dir="/data/yiquan.xu/DATASETS/toulouse/all/pre",
       # attention_resolutions="32, 16, 8",
        #class_cond=False,
        #resblock_updown=True,
        #use_fp16=True,
        #use_scale_shift_norm=True

    )
    return parser


if __name__ == "__main__":
    main()


