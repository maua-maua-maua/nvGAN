"""Temporal Frechet Inception Distance (TFID)"""

import copy
import pickle

import numpy as np
import scipy.linalg
import torch

import dnnlib
from torch_utils import misc

from . import metric_utils

# We use a different video length depending on the resolution
RES_TO_VIDEO_LEN = {
    128: 128,
    256: 128,
    512: 64,
    1024: 32,
}

#----------------------------------------------------------------------------

def compute_rd(opts, max_real: int, num_gen: int):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # TorchScript translation of LaplacianPyramidEmbedder from the src/metrics/laplacian_pyramid_embedder.py file
    detector_url = 'https://www.dropbox.com/s/meaglocg248bx6x/laplacian_pyramid_embedder.pt?dl=1'

    num_frames = RES_TO_VIDEO_LEN[opts.dataset_kwargs.resolution] # [1]
    opts = copy.deepcopy(opts)
    opts.dataset_kwargs.load_n_consecutive = num_frames
    opts.dataset_kwargs.discard_short_videos = True

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs={}, rel_lo=0, rel_hi=0,
        capture_mean_cov=True, max_items=max_real, batch_size=1, feature_stats_cls=FramesDifferencesStats, video_len=num_frames).get_mean_cov()

    if opts.generator_as_dataset:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_dataset
        gen_opts = metric_utils.rewrite_opts_for_gen_dataset(opts)
        gen_opts.dataset_kwargs.load_n_consecutive = num_frames
        gen_opts.dataset_kwargs.discard_short_videos = True
        gen_kwargs = dict()
    else:
        compute_gen_stats_fn = metric_utils.compute_feature_stats_for_generator
        gen_opts = opts
        gen_kwargs = dict(num_video_frames=num_frames)

    mu_gen, sigma_gen = compute_gen_stats_fn(
        opts=gen_opts, detector_url=detector_url, detector_kwargs={}, rel_lo=0, rel_hi=1, capture_mean_cov=True,
        max_items=num_gen, batch_size=1, feature_stats_cls=FramesDifferencesStats, video_len=num_frames, **gen_kwargs).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    misc.assert_shape(mu_real, [num_frames - 1])
    misc.assert_shape(mu_gen, [num_frames - 1])

    # m = np.square(mu_gen - mu_real).sum()
    # s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    # lprd = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    # Now, we want to find the circle with the worst repetitiveness
    lprd = abs(mu_real - mu_gen).max().item() # [1]

    return float(lprd) * 100.0

#----------------------------------------------------------------------------

class FramesDifferencesStats:
    def __init__(self, *args, video_len: int=0, max_items: int=None, diff_batch_size: int=16, **kwargs):
        assert video_len > 0, "Please, specify `video_len` explicitly."
        assert video_len % diff_batch_size == 0, f"video_len={video_len} should be divisible by diff_batch_size={diff_batch_size}"

        self.video_len = video_len
        self.max_items = max_items
        self.diff_batch_size = diff_batch_size
        self.stats = metric_utils.FeatureStats(*args, max_items=max_items, **kwargs)

    def is_full(self):
        return self.stats.is_full()

    def append_torch(self, x, *args, **kwargs):
        """
        We assume that all x are of the same video
        """
        assert x.ndim == 2, f"Bad shape: {x.shape}"
        assert x.shape[0] % self.video_len == 0, f"Bad shape: {x.shape}"

        batch_size, feat_dim = x.shape[0] // self.video_len, x.shape[1]
        x = x.view(batch_size, self.video_len, feat_dim) # [batch_size, video_len, feat_dim]
        all_diffs = []

        for diff_batch_idx in range(self.video_len // self.diff_batch_size):
            x_curr = x[:, diff_batch_idx * self.diff_batch_size : (diff_batch_idx + 1) * self.diff_batch_size] # [batch_size, diff_batch_size, feat_dim]
            diffs = x_curr.unsqueeze(2) - x.unsqueeze(1) # [batch_size, diff_batch_size, video_len, feat_dim]
            diffs = diffs.abs().sum(dim=3) # [batch_size, diff_batch_size, video_len]
            all_diffs.append(diffs)

        all_diffs = torch.cat(all_diffs, dim=1) # [batch_size, video_len, video_len]
        misc.assert_shape(all_diffs, [batch_size, self.video_len, self.video_len])

        mean_diffs = [torch.diagonal(all_diffs, offset=i, dim1=1, dim2=2).median(dim=1)[0] for i in range(1, self.video_len)] # (video_len - 1, [batch_size])
        mean_diffs = torch.stack(mean_diffs).t() # [batch_size, video_len - 1]
        self.stats.append_torch(mean_diffs, *args, **kwargs)

    def get_mean_cov(self):
        return self.stats.get_mean_cov()

    def save(self, *args, **kwargs):
        metric_utils.FeatureStats.save(self, *args, **kwargs)

    @property
    def num_items(self) -> int:
        return self.stats.num_items

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FramesDifferencesStats(capture_all=s.stats.capture_all, max_items=s.max_items, video_len=s.video_len)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------
