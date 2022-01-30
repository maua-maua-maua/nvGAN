import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_utils import misc, persistence

from .layers import EqLRConv1d, FullyConnectedLayer


def construct_frequencies(num_freqs: int, min_period_len: int, max_period_len: int) -> Tensor:
    freqs = 2 * np.pi / (2 ** np.linspace(np.log2(min_period_len), np.log2(max_period_len), num_freqs))  # [num_freqs]
    freqs = torch.from_numpy(freqs[::-1].copy().astype(np.float32)).unsqueeze(0)  # [1, num_freqs]
    return freqs


@persistence.persistent_class
class PeriodicFeatsTimeEncoder(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        freqs_dist="linspace",
        num_freqs=256,
        min_period_len=16,
        max_period_len=1024,
        predict_amplitudes=False,
        predict_periods=True,
        predict_phases=True,
        phase_dropout_std=1.0,
        num_opened_dims=256,
        num_feats_per_freq=1,
        use_cosine=True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_feats_per_freq = num_feats_per_freq
        self.use_cosine = use_cosine
        self.num_opened_dims = num_opened_dims
        self.phase_dropout_std = phase_dropout_std

        if freqs_dist == "linspace":
            freqs = construct_frequencies(num_freqs, min_period_len, max_period_len)
        else:
            raise NotImplementedError(f"Unknown freqs dist: {freqs_dist}")

        self.register_buffer("freqs", freqs)  # [1, num_fourier_feats]

        self.weights = None

        # Creating the affine without bias to prevent motion mode collapse
        if predict_amplitudes:
            self.amplitudes_predictor = FullyConnectedLayer(
                latent_dim, freqs.shape[1] * self.num_feats_per_freq * 2, activation="linear", bias=False
            )
        else:
            self.amplitudes_predictor = None

        if predict_periods:
            self.periods_predictor = FullyConnectedLayer(
                latent_dim, freqs.shape[1] * self.num_feats_per_freq, activation="linear", bias=False
            )
        else:
            self.periods_predictor = None

        if predict_phases:
            self.phase_predictor = FullyConnectedLayer(
                latent_dim, freqs.shape[1] * self.num_feats_per_freq, activation="linear", bias=False
            )
        else:
            self.phase_predictor = None

        if not self.phase_predictor is None or self.phase_dropout_std > 0.0:
            period_lens = 2 * np.pi / self.freqs  # [1, num_fourier_feats]
            phase_scales = max_period_len / period_lens  # [1, num_fourier_feats]
            phase_scales = phase_scales.repeat_interleave(
                self.num_feats_per_freq, dim=1
            )  # [1, num_fourier_feats * num_feats_per_freq]
            self.register_buffer("phase_scales", phase_scales)

        self.progressive_update(0)

    def get_dim(self) -> int:
        return self.num_feats_per_freq * self.freqs.shape[1] * (2 if self.use_cosine else 1)

    def progressive_update(self, curr_kimg: int):
        pass


@persistence.persistent_class
class AlignedTimeEncoder(PeriodicFeatsTimeEncoder):
    def __init__(self, latent_dim: int = 512):
        super().__init__(latent_dim=latent_dim)
        out_dim = self.freqs.shape[1] * self.num_feats_per_freq * (2 if self.use_cosine else 1)
        self.aligners_predictor = FullyConnectedLayer(latent_dim, out_dim, activation="linear", bias=False)

    def forward(
        self,
        t: Tensor,
        left_codes: Tensor,
        right_codes: Tensor,
        interp_weights: Tensor,
        t_left: Tensor,
        t_right: Tensor,
        return_coefs: bool = False,
    ):
        batch_size, num_frames, motion_z_dim = left_codes.shape  # [1], [1], [1]

        misc.assert_shape(t, [batch_size, num_frames])
        misc.assert_shape(left_codes, [batch_size, num_frames, None])
        misc.assert_shape(right_codes, [batch_size, num_frames, None])
        misc.assert_shape(interp_weights, [batch_size, num_frames, 1])
        assert t.shape == t_left.shape == t_right.shape, f"Wrong shape: {t.shape} vs {t_left.shape} vs {t_right.shape}"

        left_codes = left_codes.view(batch_size * num_frames, motion_z_dim)  # [batch_size * num_frames, motion_z_dim]
        right_codes = right_codes.view(batch_size * num_frames, motion_z_dim)  # [batch_size * num_frames, motion_z_dim]
        periods = self.periods_predictor(left_codes).tanh() + 1  # [batch_size * num_frames, feat_dim]
        phases = self.phase_predictor(left_codes)  # [batch_size * num_frames, feat_dim]
        aligners_left = self.aligners_predictor(left_codes)  # [batch_size * num_frames, out_dim]
        aligners_right = self.aligners_predictor(right_codes)  # [batch_size * num_frames, out_dim]

        raw_pos_embs = (
            self.freqs * periods * t.view(-1).float().unsqueeze(1) + phases * self.phase_scales
        )  # [bf, feat_dim]
        raw_pos_embs_left = (
            self.freqs * periods * t_left.view(-1).float().unsqueeze(1) + phases * self.phase_scales
        )  # [bf, feat_dim]
        raw_pos_embs_right = (
            self.freqs * periods * t_right.view(-1).float().unsqueeze(1) + phases * self.phase_scales
        )  # [bf, feat_dim]

        if self.use_cosine:
            pos_embs = torch.cat([raw_pos_embs.sin(), raw_pos_embs.cos()], dim=1)  # [bf, out_dim]
            pos_embs_left = torch.cat([raw_pos_embs_left.sin(), raw_pos_embs_left.cos()], dim=1)  # [bf, out_dim]
            pos_embs_right = torch.cat([raw_pos_embs_right.sin(), raw_pos_embs_right.cos()], dim=1)  # [bf, out_dim]
        else:
            pos_embs = raw_pos_embs.sin()  # [bf, out_dim]
            pos_embs_left = raw_pos_embs_left.sin()  # [bf, out_dim]
            pos_embs_right = raw_pos_embs_right.sin()  # [bf, out_dim]

        interp_weights = interp_weights.view(-1, 1)  # [bf, 1]
        aligners_remove = pos_embs_left * (1 - interp_weights) + pos_embs_right * interp_weights  # [bf, out_dim]
        aligners_add = aligners_left * (1 - interp_weights) + aligners_right * interp_weights  # [bf, out_dim]
        time_embs = pos_embs - aligners_remove + aligners_add  # [bf, out_dim]

        if return_coefs:
            return {
                "periods": periods,
                "phases": phases,
                "time_embs": time_embs,
                "aligners_remove": aligners_remove,
                "aligners_add": aligners_add,
            }
        else:
            return time_embs


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MotionEncoder(torch.nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, start_fpm=32, fpm_base=1, kernel_size=11, max_num_frames=1024):
        super().__init__()
        self.z_dim, self.c_dim, self.w_dim = z_dim, c_dim, w_dim
        self.time_encoder = AlignedTimeEncoder(latent_dim=w_dim)
        self.num_frames_per_motion = start_fpm * fpm_base
        self.max_num_frames = max_num_frames

        video_len_dim = 0
        input_dim = z_dim + c_dim + video_len_dim

        self.conv = nn.Sequential(
            EqLRConv1d(input_dim, z_dim, kernel_size, padding=0, activation="lrelu", lr_multiplier=0.01),
            EqLRConv1d(z_dim, w_dim, kernel_size, padding=0, activation="lrelu", lr_multiplier=0.01),
        )
        self.num_additional_codes = (kernel_size - 1) * 2

    def progressive_update(self, curr_kimg: int):
        pass

    def generate_motion(
        self, c: Tensor, t: Tensor, l: Tensor, num_frames_per_motion: int, w: Tensor = None, motion_noise: Tensor = None
    ) -> Dict:
        """
        Arguments:
            - c of shape [batch_size, c_dim]
            - t of shape [batch_size, num_frames]
            - num_frames_per_motion: int
            - w of shape [batch_size, w_dim]
        """
        out = {}
        batch_size, num_frames = t.shape

        # Consutruct trajectories (from code idx for now)
        # TODO: construct batch-wise to save computation
        # max_traj_len = np.ceil((traj_t_max - traj_t_min + 1e-8) / num_frames_per_motion).astype(int) + 1 # [1]
        max_t = max(self.max_num_frames - 1, t.max().item())  # [1]
        max_traj_len = np.ceil(max_t / num_frames_per_motion).astype(int).item() + 2  # [1]

        max_traj_len += self.num_additional_codes  # [1]

        # Now, we should select neighbouring codes for each frame
        left_idx = (t / num_frames_per_motion).floor().long()  # [batch_size, num_frames]
        batch_idx = (
            torch.arange(batch_size, device=c.device).unsqueeze(1).repeat(1, num_frames)
        )  # [batch_size, num_frames]

        if motion_noise is None:
            full_trajs = torch.randn(
                batch_size, max_traj_len, self.z_dim, device=c.device
            )  # [batch_size, max_traj_len, z_dim]
            out["motion_noise"] = full_trajs
        else:
            out["motion_noise"] = motion_noise
            full_trajs = motion_noise[:batch_size, :max_traj_len, : self.z_dim].to(
                c.device
            )  # [batch_size, max_traj_len, z_dim]

        # Construct the conditioning for LSTM
        # We would like to condition it on video lens and c
        cond = torch.zeros(batch_size * num_frames, 0, device=c.device)  # [bf, 0]

        if self.c_dim > 0:
            # Different classes have different motions, so it should be useful to condition on c
            cond = torch.cat([cond, c.repeat_interleave(t.shape[1], dim=0)], dim=1)  # [bf, num_fourier_feats + c_dim]

        cond = cond.view(t.shape[0], t.shape[1], -1)[:, 0, :]  # [batch_size, cond_dim]
        cond = cond.unsqueeze(1).repeat(1, max_traj_len, 1)  # [batch_size, max_traj_len, cond_dim]
        full_trajs = torch.cat([full_trajs, cond], dim=2)  # [batch_size, max_traj_len, z_dim + cond_dim]

        full_trajs = self.conv(full_trajs.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, max_traj_len, w_dim]

        out["full_trajs"] = full_trajs  # [batch_size, max_traj_len, z_dim]
        out["left_codes"] = full_trajs[batch_idx, left_idx]  # [batch_size, num_frames, z_dim]
        out["right_codes"] = full_trajs[batch_idx, left_idx + 1]  # [batch_size, num_frames, z_dim]

        out["t_left"] = t - t % num_frames_per_motion  # [batch_size, num_frames]
        out["t_right"] = out["t_left"] + num_frames_per_motion  # [batch_size, num_frames]
        out["interp_weights"] = (
            ((t % num_frames_per_motion) / num_frames_per_motion).unsqueeze(2).to(torch.float32)
        )  # [batch_size, num_frames, 1]
        motion_z = (
            out["left_codes"] * (1 - out["interp_weights"]) + out["right_codes"] * out["interp_weights"]
        )  # [batch_size, num_frames, z_dim]
        out["motion_z"] = motion_z.view(batch_size * num_frames, motion_z.shape[2]).to(
            torch.float32
        )  # [batch_size * num_frames, z_dim]

        return out

    def get_output_dim(self) -> int:
        return self.w_dim

    def forward(
        self,
        c: Tensor,
        t: Tensor,
        l: Tensor,
        w: Tensor = None,
        motion_noise: Dict = None,
        return_time_embs_coefs: bool = None,
    ) -> Dict:
        assert len(c) == len(t) == len(l) == len(w), f"Wrong shape: {c.shape}, {t.shape}, {l.shape}, {w.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        batch_size, num_frames = t.shape
        out = {}
        motion_info = self.generate_motion(c, t, l, self.num_frames_per_motion, w=w, motion_noise=motion_noise)
        torch.save(motion_info, "/tmp/motion_info.pt")
        motion_z = motion_info["motion_z"].view(t.shape[0] * t.shape[1], -1)  # [batch_size * num_frames, z_dim]

        time_enc_out = self.time_encoder(
            t=t,
            left_codes=motion_info["left_codes"],
            right_codes=motion_info["right_codes"],
            t_left=motion_info["t_left"],
            t_right=motion_info["t_right"],
            interp_weights=motion_info["interp_weights"],
            return_coefs=return_time_embs_coefs,
        )  # <dict or Tensor :(>

        if return_time_embs_coefs:
            out = {**time_enc_out, **out}
            motion_w = time_enc_out["time_embs"]  # [batch_size * num_frames, motion_w_dim]
        else:
            motion_w = time_enc_out  # [batch_size * num_frames, motion_w_dim]

        out["motion_w"] = motion_w + motion_z.sum() * 0.0  # [batch_size * num_frames, w_dim]
        out["motion_noise"] = motion_info["motion_noise"]  # (Any shape)

        return out



@persistence.persistent_class
class TemporalInput(nn.Module):
    def __init__(
        self, resolution, channel_dim: int, w_dim: int, motion_w_dim: int, has_const=True, has_variable_input=False
    ):
        super().__init__()

        self.motion_w_dim = motion_w_dim
        self.resolution = resolution

        # Const input
        if has_const:
            self.const = nn.Parameter(torch.randn(1, channel_dim, resolution, resolution))
        else:
            self.const = None

        # Variable input
        if has_variable_input:
            self.repeat = input["var_repeat"]
            fc_output_dim = channel_dim if self.repeat else channel_dim * resolution ** 2  # [1]
            self.fc = FullyConnectedLayer(w_dim, fc_output_dim, activation="lrelu")
        else:
            self.fc = None

    def get_total_dim(self):
        total_dim = self.motion_w_dim
        total_dim += 0 if self.const is None else self.const.shape[1]
        total_dim += 0 if self.fc is None else self.const.shape[1]

        return total_dim

    def forward(self, t: Tensor, motion_w: Tensor, w: Tensor = None) -> Tensor:
        """
        motion_w: [batch_size, motion_w_dim]
        """
        out = torch.cat(
            [
                self.const.repeat(len(motion_w), 1, 1, 1),
                motion_w.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.resolution, self.resolution),
            ],
            dim=1,
        )  # [batch_size, channel_dim + num_fourier_feats * 2]

        if not self.fc is None:
            if self.repeat:
                res = self.resolution
                var_part = (
                    self.fc(w).unsqueeze(2).unsqueeze(3).repeat(1, 1, res, res)
                )  # [batch_size, channel_dim, h, w]
            else:
                var_part = self.fc(w).view(len(w), -1, *out.shape[2:])  # [batch_size, channel_dim, h, w]

            out = torch.cat([out, var_part], dim=1)  # [batch_size, channel_dim + num_fourier_feats * 2]

        return out


# ----------------------------------------------------------------------------


@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, channel_dim: int, w_dim: int, motion_w_dim: int = None):
        super().__init__()
        self.input = TemporalInput(resolution=4, channel_dim=channel_dim, w_dim=w_dim, motion_w_dim=motion_w_dim)
        self.total_dim = self.input.get_total_dim()

    def forward(self, w: Tensor = None, t: Optional[Tensor] = None, motion_w: Optional[Tensor] = None) -> Tensor:
        x = self.input(t, w=w, motion_w=motion_w)  # [batch_size, d, h, w]
        return x


# ----------------------------------------------------------------------------


def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int = 0) -> Tuple[int, Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats))  # [num_fourier_feats]
    powers = powers[: len(powers) - skip_small_t_freqs]  # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi  # [1, num_fourier_feats]

    return fourier_coefs / time_resolution


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FixedTimeEncoder(nn.Module):
    def __init__(
        self,
        max_num_frames: int,  # Maximum T size
        transformer_pe: bool = False,  # Whether we should use positional embeddings from Transformer
        d_model: int = 512,  # d_model for Transformer PE's
        skip_small_t_freqs: int = 0,  # How many high frequencies we should skip
    ):
        super().__init__()

        assert max_num_frames >= 1, f"Wrong max_num_frames: {max_num_frames}"

        if transformer_pe:
            assert skip_small_t_freqs == 0, "Cant use `skip_small_t_freqs` with `transformer_pe`"
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(
                0
            )  # [1, d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=skip_small_t_freqs)

        self.register_buffer("fourier_coefs", fourier_coefs)  # [1, num_fourier_feats]

    def get_dim(self) -> int:
        return self.fourier_coefs.shape[1] * 2

    def progressive_update(self, curr_kimg):
        pass

    def forward(self, t: Tensor) -> Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        t = t.view(-1).float()  # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1)  # [bf, num_fourier_feats]

        fourier_embs = torch.cat(
            [
                fourier_raw_embs.sin(),
                fourier_raw_embs.cos(),
            ],
            dim=1,
        )  # [bf, num_fourier_feats * 2]

        return fourier_embs


# ----------------------------------------------------------------------------


class TemporalDifferenceEncoder(nn.Module):
    def __init__(self, num_frames_per_sample=2, max_num_frames=1024):
        super().__init__()

        self.num_frames_per_sample = num_frames_per_sample

        if num_frames_per_sample > 1:
            self.d = 256
            self.const_embed = nn.Embedding(max_num_frames, self.d)
            self.time_encoder = FixedTimeEncoder(max_num_frames)

    def get_total_dim(self) -> int:
        if self.num_frames_per_sample == 1:
            return 1
        else:
            return (self.d + self.time_encoder.get_dim()) * (self.num_frames_per_sample - 1)

    def forward(self, t: Tensor) -> Tensor:
        misc.assert_shape(t, [None, self.num_frames_per_sample])

        batch_size = t.shape[0]

        if self.num_frames_per_sample == 1:
            out = torch.zeros(len(t), 1, device=t.device)
        else:
            num_diffs_to_use = self.num_frames_per_sample - 1
            t_diffs = (t[:, 1:] - t[:, :-1]).view(-1)  # [batch_size * (num_frames - 1)]
            # Note: float => round => long is necessary when it's originally long
            const_embs = self.const_embed(t_diffs.float().round().long())  # [batch_size * num_diffs_to_use, d]
            fourier_embs = self.time_encoder(t_diffs.unsqueeze(1))  # [batch_size * num_diffs_to_use, num_fourier_feats]
            out = torch.cat([const_embs, fourier_embs], dim=1)  # [batch_size * num_diffs_to_use, d + num_fourier_feats]
            out = out.view(batch_size, num_diffs_to_use, -1).view(
                batch_size, -1
            )  # [batch_size, num_diffs_to_use * (d + num_fourier_feats)]

        return out

