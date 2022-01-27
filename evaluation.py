import copy
import json
import os
import sys

import numpy as np
import scipy.linalg
import torch
from prdc import compute_prdc

import dnnlib
import legacy
from metrics import metric_utils

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def compute_fid(mu_real, sigma_real, mu_gen, sigma_gen):
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = float(np.real(m + np.trace(sigma_gen + sigma_real - s * 2)))
    return fid


def compute_kid(real_features, fake_features, num_subsets=100, max_subset_size=1000):
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], fake_features.shape[0]), max_subset_size)
    t = 0
    for _ in range(num_subsets):
        x = fake_features[np.random.choice(fake_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = float(t / num_subsets / m)
    return kid


@torch.no_grad()
def compute_metrics(network_pkl, datapath, n=25000):
    if os.path.exists(network_pkl.replace("pkl", "json")):
        with open(network_pkl.replace("pkl", "json"), "r") as fp:
            metrics = json.load(fp)
        return metrics

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False).to(device)

    opts = metric_utils.MetricOptions(
        G=G,
        dataset_kwargs=dict(
            class_name="training.dataset.ImageFolderDataset", path=datapath, resolution=G.img_resolution
        ),
        num_gpus=1,
        rank=0,
        device="cuda",
        progress=metric_utils.ProgressMonitor(verbose=False),
        cache=True,
    )


    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)

    real_feature_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_all=True,
        capture_mean_cov=True,
        max_items=n,
        data_loader_kwargs=dict(pin_memory=True, num_workers=1, prefetch_factor=2),
    )
    fake_feature_stats = metric_utils.compute_feature_stats_for_generator(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=1,
        capture_all=True,
        capture_mean_cov=True,
        #jit=True,
        max_items=n,
    )

    real_features = real_feature_stats.get_all()
    fake_features = fake_feature_stats.get_all()
    mu_real, sigma_real = real_feature_stats.get_mean_cov()
    mu_gen, sigma_gen = fake_feature_stats.get_mean_cov()

    metrics = compute_prdc(real_features, fake_features, nearest_k=5)
    metrics["fid"] = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    metrics["kid"] = compute_kid(real_features, fake_features)

    with open(network_pkl.replace("pkl", "json"), "w") as fp:
        json.dump(metrics, fp)

    return metrics


if __name__ == "__main__":
    network_pkl = sys.argv[1]
    datapath = sys.argv[2]
    metrics = compute_metrics(network_pkl, datapath)
    print(network_pkl, metrics)
