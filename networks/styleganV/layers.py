import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_utils import misc, persistence
from torch_utils.ops import bias_act, conv2d_resample, upfirdn2d

# ----------------------------------------------------------------------------


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(self, z, c, l=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
        instance_norm=False,  # Should we apply instance normalization to y?
        lr_multiplier=1.0,  # Learning rate multiplier.
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.instance_norm = instance_norm
        self.lr_multiplier = lr_multiplier

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * (self.weight_gain * self.lr_multiplier)
        b = (self.bias.to(x.dtype) * self.lr_multiplier) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.instance_norm:
            x = (x - x.mean(dim=(2, 3), keepdim=True)) / (
                x.std(dim=(2, 3), keepdim=True) + 1e-8
            )  # [batch_size, c, h, w]

        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class EqLRConv1d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        activation: str = "linear",
        lr_multiplier: float = 1.0,
        bias=True,
        bias_init=0.0,
    ):
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features, kernel_size]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features * kernel_size)
        self.bias_gain = lr_multiplier
        self.padding = padding
        self.stride = stride

        assert self.activation in ["lrelu", "linear"]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3, f"Wrong shape: {x.shape}"

        w = self.weight.to(x.dtype) * self.weight_gain  # [out_features, in_features, kernel_size]
        b = self.bias  # [out_features]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        y = F.conv1d(
            input=x, weight=w, bias=b, stride=self.stride, padding=self.padding
        )  # [batch_size, out_features, out_len]
        if self.activation == "linear":
            pass
        elif self.activation == "lrelu":
            y = F.leaky_relu(y, negative_slope=0.2)  # [batch_size, out_features, out_len]
        else:
            raise NotImplementedError
        return y


# ----------------------------------------------------------------------------


@persistence.persistent_class
class TemporalInput(nn.Module):
    def __init__(
        self, resolution, channel_dim: int, w_dim: int, motion_w_dim: int, has_const=True, has_variable_input=False
    ):
        super().__init__()

        self.motion_w_dim = motion_w_dim

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


class TemporalDifferenceEncoder(nn.Module):
    def __init__(self, num_frames_per_sample=2, max_num_frames=1024):
        super().__init__()

        self.num_frames_per_sample = num_frames_per_sample

        if num_frames_per_sample > 1:
            self.d = 256
            self.const_embed = nn.Embedding(max_num_frames, self.d)
            self.time_encoder = FixedTimeEncoder(max_num_frames, transformer_pe=False, skip_small_t_freqs=0)

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


# ----------------------------------------------------------------------------


def remove_diag(x: Tensor) -> Tensor:
    """
    Takes an input of size [..., n, n] and returns a tensor of size [..., n, n-1],
    where we removed diagonal element from the square matrix at the end

    Based on https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    """
    assert x.shape[-1] == x.shape[-2], f"The end dims should be a matrix: {x.shape}"

    core_shape = x.shape[:-2]  # The shape which is not fixed during the manipulations
    n = x.shape[-1]  # Matrix size
    y = x.reshape(*core_shape, -1)[..., 1:]  # Removed the very first element
    y = y.reshape(*core_shape, n - 1, n + 1)  # Reshaped in such a way that diagonal elements are at the end
    y = y[..., :-1]  # Removed the diagonal elements
    y = y.reshape(*core_shape, n, n - 1)  # Reshaped to the proper size

    return y


# ----------------------------------------------------------------------------


def ema(x: Tensor, alpha, dim: int = -1):
    """
    Adapted / copy-pasted from https://stackoverflow.com/a/42926270
    """
    # alpha = 2 / (window + 1.0)
    assert 0.0 <= alpha < 1.0, f"Bad alpha value: {alpha}. It should be in [0, 1)"
    assert dim == -1, f"Not implemented for dim: {dim}"
    assert x.size(dim) <= 1024, f"Too much points for a vectorized implementation: {x.shape}"

    alpha_rev = 1.0 - alpha  # [1]
    num_points = x.size(dim)  # [1]
    pows = alpha_rev ** (torch.arange(num_points + 1, device=x.device))  # [num_points + 1]
    scale_arr = 1 / pows[:-1].double()  # [num_points]

    # Note: broadcast from the last dimension
    offset = x[..., [0]] * pows[1:]  # [..., num_points]
    pw0 = alpha * (alpha_rev ** (num_points - 1))  # [1]
    cumsums = (x * pw0 * scale_arr).to(x.dtype).cumsum(dim=dim)  # [..., num_points]
    x_ema = offset + (cumsums * scale_arr.flip(0)).to(x.dtype)  # [..., num_points]

    return x_ema


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
