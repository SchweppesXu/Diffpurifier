import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch as th
from PIL import Image
import argparse
import torch.distributed as dist
import numpy as np
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
from common import (read_model_and_diffusion,read_model_and_diffusion_ema)
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
        use_ddim=True,
        data_dir="/data/yiquan.xu/DATASETS/shuguang/optical"
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
parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = th.device("cuda")

def main():

    if not os.path.exists('/data/yiquan.xu/DATASETS/shuguang/YUV2SAR'):
        os.mkdir('/data/yiquan.xu/DATASETS/shuguang/YUV2SAR')
    dist_util.setup_dist()
    logger.configure()

    logger.log(f"reading models")
    source_dir = "/data/yiquan.xu/DATASETS/shuguang/ddpm_opt_model"
    source_model, diffusion = read_model_and_diffusion(args_optical, source_dir, dev, synthetic=False)

    target_dir = "/data/yiquan.xu/DATASETS/shuguang/ddpm_sar_model1000"
    target_model, diffusion_target = read_model_and_diffusion(args_sar, target_dir, dev, synthetic=False)

    logger.log("running image translation...")
    data = load_source_data_for_domain_translation(
        batch_size=args_optical.batch_size,
        image_size=args_optical.image_size,
        in_channels=args_optical.in_channels,
        data_dir=args_optical.data_dir,
        class_cond=False,
        use_yuv_space=True
    )
    j=0

    for i, (batch, extra) in enumerate(data):
        batch = batch.to(dist_util.dev())
        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise = diffusion.ddim_reverse_sample_loop(
            source_model,batch,
            clip_denoised=True,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")


        # Next, decode the latents to the target class.
        sample = diffusion_target.ddim_sample_loop(
            target_model,(args_sar.batch_size,3, args_sar.image_size, args_sar.image_size),
            noise=noise,
            clip_denoised=True,
            device=dist_util.dev(),
            eta=args_sar.eta
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        noise = ((noise + 1) * 127.5).clamp(0, 255).to(th.uint8)
        noise = noise.permute(0, 2, 3, 1)
        noise = noise.contiguous()

        images = []
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(images) * args_sar.batch_size} samples")

        imagesnoise = []
        gathered_samples3 = [th.zeros_like(noise) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples3, noise)  # gather not supported with NCCL
        imagesnoise.extend([noise.cpu().numpy() for noise in gathered_samples3])
        imagesnoise = np.concatenate(imagesnoise, axis=0)

        logger.log("saving translated images.")
        images = np.concatenate(images, axis=0)

        image = Image.fromarray(images[0].squeeze())
        filepath = os.path.join("/data/yiquan.xu/DATASETS/shuguang/YUV2SAR", f"{j}.png")
        image.save(filepath)
        # imagenoise = Image.fromarray(imagesnoise[index].squeeze())
        # imagenoise.convert('L').save(filepathnoise)
        logger.log(f"    saving: {filepath}")
        j=j+1

    dist.barrier()
    logger.log(f"domain translation complete")
if __name__ == "__main__":
    main()


