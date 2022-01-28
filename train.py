# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
# modified by Hans Brouwer for Maua

import glob
import json
import os
import re
import tempfile

import click
import torch

import dnnlib
import legacy
from metrics import metric_main
from torch_utils import custom_ops, misc, training_stats
from training import training_loop

#----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)


def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


def init_dataset_kwargs(data, video=False):
    try:
        if video:
            dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.video.VideoFramesFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
            dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
            dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
            dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        else:
            dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.image.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
            dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
            dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
            dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
            dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

# Finds the latest pkl file in the `outdir`, including its kimg number.
# Reimplementation of https://github.com/skyflynil/stylegan2/commit/8c57ee4633d334e480a23d7f82433c7649d50866
def locate_latest_pkl(outdir: str):
    allpickles = sorted(glob.glob(os.path.join(outdir, '0*', 'network-*.pkl')))
    latest_pkl = allpickles[-1]
    RE_KIMG = re.compile('network-snapshot-(\d+).pkl')
    latest_kimg = int(RE_KIMG.match(os.path.basename(latest_pkl)).group(1))
    return latest_pkl, latest_kimg

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2', 'fastgan', 'fastgan_lite']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)

# Misc hyperparameters.
@click.option('--projected',    help='Use projected discriminator', metavar='BOOL',             type=bool, is_flag=True, show_default=True)
@click.option('--video',        help='Train on a video dataset', metavar='BOOL',                type=bool, is_flag=True, show_default=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0))
@click.option('--style_mix_p',  help='Style mixing probability', metavar='FLOAT',               type=click.FloatRange(min=0))
@click.option('--pl_weight',    help='Path length regularization weight', metavar='FLOAT',      type=click.FloatRange(min=0))
@click.option('--z-dim',        help='Size of normal z latent vector', metavar='FLOAT',         type=click.IntRange(min=0))
@click.option('--w-dim',        help='Size of disentangled w latent vector', metavar='FLOAT',   type=click.IntRange(min=0))
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0))
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, is_flag=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, is_flag=True, show_default=True)
@click.option('--mirrory',      help='Enable dataset y-flips', metavar='BOOL',                  type=bool, is_flag=True, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--augpipe',      help='Augmentation pipeline',                                   type=click.Choice(['bg', 'bgc']), default='bgc', show_default=True)
@click.option('--resume',       help='Resume from given network pickle (PATH, URL or "latest")', metavar='[PATH|URL|"latest"]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--initstrength', help='Override ADA strength at start',                          type=click.FloatRange(min=0))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, is_flag=True, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, is_flag=True, show_default=True)
@click.option('--allow_tf32',   help='Can improve training speed at the cost of numerical precision', metavar='BOOL', type=bool, is_flag=True, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--restart_every',help='Time interval in seconds to restart code', metavar='INT', type=int, default=9999999, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.

    # Training set.
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, video=opts.video)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.yflip = opts.mirrory

    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus

    # Model configuration
    c.G_kwargs = dnnlib.EasyDict(class_name=None, mapping_kwargs=dnnlib.EasyDict())
    c.G_kwargs.z_dim = opts.z_dim if opts.z_dim else (64 if opts.projected else 512)
    c.G_kwargs.w_dim = opts.w_dim if opts.w_dim else (128 if opts.projected else 512)

    if opts.projected:
        c.D_kwargs = dnnlib.EasyDict(
            class_name='networks.projected_discriminator.ProjectedDiscriminator',
            diffaug=True,
            interp224=(c.training_set_kwargs.resolution < 224),
            backbone_kwargs=dnnlib.EasyDict(),
        )
        c.D_kwargs.backbone_kwargs.cout = 64
        c.D_kwargs.backbone_kwargs.expand = True
        c.D_kwargs.backbone_kwargs.proj_type = 2
        c.D_kwargs.backbone_kwargs.num_discs = 4
        c.D_kwargs.backbone_kwargs.cond = opts.cond
    else:
        c.D_kwargs = dnnlib.EasyDict(class_name='networks.stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
        c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group

    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.stylegan2.StyleGAN2Loss')

    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth if opts.map_depth else (8 if opts.cfg == 'stylegan2' else 2)

    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)

    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'networks.stylegan2.Generator'
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.00025 if opts.projected else 0.0002
        c.D_kwargs.backbone_kwargs.separable = False
    elif opts.cfg in ['fastgan', 'fastgan_lite']:
        c.G_kwargs = dnnlib.EasyDict(class_name='networks.fastgan.Generator', cond=opts.cond, synthesis_kwargs=dnnlib.EasyDict())
        c.G_kwargs.synthesis_kwargs.lite = (opts.cfg == 'fastgan_lite')
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        c.D_kwargs.backbone_kwargs.separable = True
    else:
        c.D_kwargs.backbone_kwargs.separable = False  # TODO does stylegan3 want separable or not?
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.00025
        c.G_kwargs.class_name = 'networks.stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
            c.G_kwargs.channel_base *= 2 # Double the number of feature maps.
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.
            c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32 # Fade out the blur during the first N kimg.

    if opts.glr is not None:
        c.G_opt_kwargs.lr = opts.glr # Override the default learning rate if specified
    if opts.dlr is not None:
        c.D_opt_kwargs.lr = opts.dlr
    c.ema_kimg = c.batch_size * 10 / 32

    # Regularization
    if opts.gamma is None:
        opts.gamma = 0.0002 * (c.training_set_kwargs.resolution ** 2) / opts.batch
    else:
        c.loss_kwargs.r1_gamma = opts.gamma
    
    if opts.style_mix_p is None: 
        c.loss_kwargs.style_mixing_prob = 0 if opts.projected else 0.9
    else:
        c.loss_kwargs.style_mixing_prob = opts.style_mix_p
    if opts.video:
        c.loss_kwargs.style_mixing_prob = 0  # video training doesn't work with style mixing
    
    if opts.pl_weight is None: 
        c.loss_kwargs.pl_weight = 0 if opts.projected else 2
    else:
        c.loss_kwargs.pl_weight = opts.pl_weight

    c.G_reg_interval = 4 # Enable lazy regularization for G.

    # Augmentation.
    if opts.aug != 'noaug':
        if opts.augpipe == 'bg':
            # xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1
            c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.ada.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0)
        else:
            c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.ada.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Initial Augmentation Strength.
    if opts.initstrength is not None:
        assert isinstance(opts.initstrength, float)
        c.augment_p = opts.initstrength
    
    # Hyperparameters & settings.
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = 1
    c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.restart_every = opts.restart_every

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if not opts.projected and c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Performance-related toggles.
    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False
    if 'stylegan2' in opts.cfg:
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        c.loss_kwargs.pl_no_weight_grad = True # Speed up path length regularization by skipping gradient computation wrt. conv2d weights.
    c.allow_tf32 = opts.allow_tf32

    # Resume.
    if opts.resume is not None:
        if opts.resume == "latest":
            c.resume_pkl, c.resume_kimg = locate_latest_pkl(opts.outdir)
        else:
            c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ema_rampup = None  # Disable EMA rampup.

    # Description string.
    desc = f'{opts.cfg:s}'
    if opts.projected:
        desc += '-projected'
    desc += f'-{dataset_name:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    desc += f'-gpus{c.num_gpus:d}-batch{c.batch_size:d}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        # get current number of training images
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg//1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
