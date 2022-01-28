"""Generates a dataset of images using pretrained network pickle."""

import sys; sys.path.extend(['.', 'src'])
import json
import os
import random
import re
import warnings
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf
from src import dnnlib
from src.training.logging import (generate_videos,
                                  save_video_frames_as_frames_parallel,
                                  save_video_frames_as_mp4)
from tqdm import tqdm

import legacy

torch.set_grad_enabled(False)


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network_pkl', help='Network pickle filename', metavar='PATH')
@click.option('--networks_dir', help='Network pickles directory', metavar='PATH')
@click.option('--truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--noise_mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--num_videos', type=int, help='Number of images to generate', default=50000, show_default=True)
@click.option('--batch_size', type=int, help='Batch size to use for generation', default=32, show_default=True)
@click.option('--same_motion', type=bool, help='Should we use the same motion codes for all videos in a row?', default=False, show_default=True)
@click.option('--seed', type=int, help='Random seed', default=42, metavar='DIR')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--save_as_mp4', help='Should we save as independent frames or mp4?', type=bool, default=False, metavar='BOOL')
@click.option('--video_len', help='Number of frames to generate', type=int, default=16, metavar='INT')
@click.option('--fps', help='FPS for mp4 saving', type=int, default=25, metavar='INT')
@click.option('--as_grids', help='Save videos as grids', type=bool, default=False, metavar='BOOl')
@click.option('--overwrite_l_arg', help='Overwrite -l argument in G', type=int, metavar='INT')
@click.option('--img_resolution', help='Image resolution of the generator?', default=256, type=int, metavar='INT')
@click.option('--time_offset', help='Additional time offset', default=0, type=int, metavar='INT')
@click.option('--dataset_path', help='Dataset path', default="", type=str, metavar='PATH')
@click.option('--hydra_cfg_path', help='Config path', default="", type=str, metavar='PATH')
@click.option('--freqs_scale', help='Additional frequencies scaling?', default=1.0, type=float, metavar='FLOAT')
@click.option('--num_weights_to_slice', help='Number of high-frequency coords to remove.', default=0, type=int, metavar='INT')
@click.option('--fps_increase', help='FPS increase factor.', default=1, type=int, metavar='INT')
@click.option('--ckpt_select_metric', help='Which metric to use when selecting the best ckpt?', default='fvd2048_16f', type=str, metavar='STR')

def generate_dataset(
    ctx: click.Context,
    network_pkl: str,
    networks_dir: str,
    truncation_psi: float,
    noise_mode: str,
    num_videos: int,
    batch_size: int,
    same_motion: bool,
    seed: int,
    outdir: str,
    save_as_mp4: bool,
    video_len: int,
    fps: int,
    as_grids: bool,
    overwrite_l_arg: int,
    img_resolution: int,
    time_offset: int,
    dataset_path: os.PathLike,
    hydra_cfg_path: os.PathLike,
    freqs_scale: float,
    num_weights_to_slice: int,
    fps_increase: int,
    ckpt_select_metric: str,
):
    if network_pkl is None:
        output_regex = "^network-snapshot-\d{6}.pkl$"
        ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        # ckpts = sorted([f for f in os.listdir(networks_dir) if ckpt_regex.match(f)])
        # network_pkl = os.path.join(networks_dir, ckpts[-1])
        metrics_file = os.path.join(networks_dir, f'metric-{ckpt_select_metric}.jsonl')
        with open(metrics_file, 'r') as f:
            snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
        best_snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results'][ckpt_select_metric])[0]
        network_pkl = os.path.join(networks_dir, best_snapshot['snapshot_pkl'])
        print(f'Using checkpoint: {network_pkl} with FVD16 of', best_snapshot['results'][ckpt_select_metric])
        # Selecting a checkpoint with the best score
    else:
        assert networks_dir is None, "Cant have both parameters: network_pkl and networks_dir"

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore

    os.makedirs(outdir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # from src.training.networks import Generator
    # G.cfg.z_dim = G.z_dim
    # G.cfg.motion.fourier = G.cfg.motion.embed
    # G_new = Generator(
    #     w_dim=G.cfg.w_dim,
    #     mapping_kwargs=dnnlib.EasyDict(num_layers=G.cfg.get('mapping_net_n_layers', 2), cfg=G.cfg),
    #     synthesis_kwargs=dnnlib.EasyDict(
    #         channel_base=int(G.cfg.get('fmaps', 0.5) * 32768),
    #         channel_max=G.cfg.get('channel_max', 512),
    #         num_fp16_res=4,
    #         conv_clamp=256,
    #     ),
    #     cfg=G.cfg,
    #     img_resolution=img_resolution,
    #     img_channels=3,
    #     c_dim=G.cfg.c_dim,
    # ).to(device).eval()
    # G_new.load_state_dict(G.state_dict())
    # G = G_new

    if num_weights_to_slice > 0:
        if G.synthesis.motion_encoder.time_encoder.weights is None:
            G.synthesis.motion_encoder.time_encoder.weights = torch.ones_like(G.synthesis.motion_encoder.time_encoder.freqs)
        G.synthesis.motion_encoder.time_encoder.weights[:, -num_weights_to_slice:] = 0.0
    if freqs_scale != 1.0:
        G.synthesis.motion_encoder.time_encoder.freqs.data.mul_(freqs_scale)

    all_z = torch.randn(num_videos, G.z_dim, device=device) # [curr_batch_size, z_dim]

    if dataset_path and G.c_dim > 0:
        hydra_cfg_path = hydra_cfg_path or os.path.join(networks_dir, '..', "experiment_config.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path)
        training_set_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.VideoFramesFolderDataset',
            path=dataset_path, cfg=hydra_cfg.dataset, use_labels=True, max_size=None, xflip=False)
        training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
        all_c = [training_set.get_label(random.choice(range(len(training_set)))) for _ in range(num_videos)] # [num_videos, c_dim]
        all_c = torch.from_numpy(np.array(all_c)).to(device) # [num_videos, c_dim]
    elif G.c_dim > 0:
        warnings.warn('Assuming that the conditioning is one-hot!')
        c_idx = torch.randint(low=0, high=G.c_dim, size=(num_videos, 1), device=device)
        all_c = torch.zeros(num_videos, G.c_dim, device=device) # [num_videos, c_dim]
        all_c.scatter_(1, c_idx, 1)
    else:
        all_c = torch.zeros(num_videos, G.c_dim, device=device) # [num_videos, c_dim]

    ts = time_offset + torch.arange(video_len, device=device).float().unsqueeze(0).repeat(batch_size, 1) / fps_increase # [batch_size, video_len]

    if G.cfg.use_video_len_cond:
        l = torch.ones([batch_size], device=device)
        l = l * (overwrite_l_arg if not overwrite_l_arg is None else video_len)
    else:
        l = torch.zeros(batch_size, device=device) # [batch_size]

    if same_motion:
        assert as_grids, "Same motion is supported only for grid visualizations"
        assert batch_size == num_videos, "Same motion is supported only for batch_size == num_videos"
        num_rows = num_cols = int(np.sqrt(num_videos))
        motion_noise = G.synthesis.motion_encoder(
            c=all_c[:num_rows],
            t=ts[:num_rows],
            l=l[:num_rows],
            w=torch.zeros(num_rows, G.w_dim, device=device))['motion_noise'] # [1, *motion_dims]
        motion_noise = motion_noise.repeat_interleave(num_cols, dim=0) # [batch_size, *motion_dims]

        all_z = all_z[:num_cols].repeat(num_rows, 1) # [num_videos, z_dim]
        all_c = all_c[:num_cols].repeat(num_rows, 1) # [num_videos, z_dim]
    else:
        motion_noise = None


    # Generate images.
    for batch_idx in tqdm(range((num_videos + batch_size - 1) // batch_size), desc='Generating the dataset'):
        curr_batch_size = batch_size if batch_size * (batch_idx + 1) <= num_videos else num_videos % batch_size
        z = all_z[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] # [curr_batch_size, z_dim]
        c = all_c[batch_idx * batch_size:batch_idx * batch_size + curr_batch_size] # [curr_batch_size, c_dim]

        videos = generate_videos(
            G, z, c, ts[:curr_batch_size], l=l[:curr_batch_size], motion_noise=motion_noise,
            noise_mode=noise_mode, truncation_psi=truncation_psi, as_grids=as_grids, batch_size_num_frames=128)

        if as_grids:
            videos = [videos]

        for video_idx, video in enumerate(videos):
            if save_as_mp4:
                save_path = os.path.join(outdir, f'{batch_idx * batch_size + video_idx:06d}.mp4')
                save_video_frames_as_mp4(video, fps, save_path)
            else:
                save_dir = os.path.join(outdir, f'{batch_idx * batch_size + video_idx:06d}')
                video = (video * 255).permute(0, 2, 3, 1).to(torch.uint8).numpy() # [video_len, h, w, c]
                save_video_frames_as_frames_parallel(video, save_dir, time_offset=time_offset, num_processes=8)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dataset() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
