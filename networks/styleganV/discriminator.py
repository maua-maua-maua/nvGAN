import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_utils import misc, persistence
from torch_utils.ops import upfirdn2d

from .generator import SynthesisLayer
from .layers import (Conv2dLayer, FullyConnectedLayer, MappingNetwork,
                     TemporalDifferenceEncoder, remove_diag)


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
        c_dim=0,  # Embedding size for t.
        agg_concat_res=16,
        num_frames_per_sample=2,
        contr_resolutions=[],
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()

        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()
        conv0_in_channels = in_channels if in_channels > 0 else tmp_channels

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            conv0_in_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if int(self.resolution) in [int(r) for r in contr_resolutions] and num_frames_per_sample > 1:
            assert (
                agg_concat_res < self.resolution
            ), f"Cant compute similarities after concatenation: {agg_concat_res} > {self.resolution}"
            self.contr = GroupwiseContrastiveLayer(
                in_channels=conv0_in_channels,
                c_dim=c_dim,
                resolution=self.resolution,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                num_frames_per_sample=num_frames_per_sample,
            )
            conv1_in_channels = self.contr.get_output_dim()
        else:
            self.contr = None
            conv1_in_channels = tmp_channels

        self.conv1 = Conv2dLayer(
            conv1_in_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                conv0_in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, c, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == "skip":
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == "skip" else None

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = x if self.contr is None else self.contr(x, c)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = x if self.contr is None else self.contr(x, c)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [N(C+1)HW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class GroupwiseContrastiveLayer(torch.nn.Module):
    """
    This layer compares images of the same video with one another
    and concatenates the similarity scores back to their original activations
    """

    def __init__(
        self,
        in_channels: int,
        resolution: int,
        c_dim: int,
        conv_clamp: int = None,
        channels_last: bool = False,
        num_frames_per_sample=2,
        dim=256,
        diff_based=True,
        kernel_size=3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.group_size = num_frames_per_sample
        self.dim = dim
        self.scale = 1 if self.dim <= 3 else ((self.dim - 2) ** 2 / self.dim) ** 0.5
        self.diff_based = diff_based

        self.transform = SynthesisLayer(
            in_channels=in_channels,
            out_channels=dim,
            w_dim=c_dim,
            resolution=resolution,
            kernel_size=kernel_size,
            activation="lrelu",
            conv_clamp=conv_clamp,
            channels_last=channels_last,
        )

    def get_output_dim(self) -> int:
        if self.diff_based:
            return self.in_channels + (self.group_size - 1) * (self.group_size - 2)
        else:
            return self.in_channels + self.group_size - 1

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        bn, c_in, h, w = x.shape
        num_groups = bn // self.group_size
        y = self.transform(x, c)  # [bn, dim, h, w]
        y = y.reshape(num_groups, self.group_size, self.dim, h, w)  # [num_groups, group_size, dim, h, w]

        if self.diff_based:
            # We first compute differences between frames features
            # and then we compute similarities between those vectors
            # This should show D how the pixels moved
            d = y[:, 1:] - y[:, :-1]  # [num_groups, group_size - 1, dim, h, w]
            # Unfortunately, we cannot normalize the diffs because R1 penalty falls with NaNs...
            # So, normalize the scale at least somehow...
            d_sims = (
                (1.0 / np.sqrt(d.shape[2])) * d.permute(0, 3, 4, 1, 2) @ d.permute(0, 3, 4, 2, 1)
            )  # [num_groups, h, w, group_size - 1, group_size - 1]
            assert not torch.isnan(d_sims).any(), "There are NaNs in the diffs tensor"
            d_sims = remove_diag(d_sims)  # [num_groups, h, w, group_size - 1, group_size - 2]
            y = d_sims.unsqueeze(1).repeat(
                1, self.group_size, 1, 1, 1, 1
            )  # [num_groups, group_size, h, w, group_size - 1, group_size - 2]
            y = y.view(bn, h, w, (self.group_size - 1) * (self.group_size - 2)).permute(
                0, 3, 1, 2
            )  # [bn, (group_size - 1) * (group_size - 2), h, w]
        else:
            y = F.normalize(y, dim=2)  # [num_groups, group_size, dim, h, w]
            y = y.permute(0, 3, 4, 1, 2) @ y.permute(0, 3, 4, 2, 1)  # [num_groups, h, w, group_size, group_size]
            y = y * self.scale  # [num_groups, h, w, group_size, group_size]
            y = remove_diag(y)  # [num_groups, h, w, group_size, group_size - 1]
            y = y.permute(0, 3, 4, 1, 2)  # [num_groups, group_size, group_size - 1, h, w]
            y = y.reshape(bn, self.group_size - 1, h, w)  # [bn, group_size - 1, h, w]

        y = torch.cat([x, y], dim=1)  # [bn, in_channel + d, h, w]

        return y


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FeatDiffLayer(torch.nn.Module):
    """
    Computes differences between consecutive frames features
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        resolution: int,
        c_dim: int,
        conv_clamp: int = None,
        channels_last: bool = False,
        num_frames_per_sample=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.group_size = num_frames_per_sample
        self.dim = dim

        self.transform = SynthesisLayer(
            in_channels=in_channels,
            out_channels=self.dim,
            w_dim=c_dim,
            resolution=resolution,
            kernel_size=3,
            activation="lrelu",
            conv_clamp=conv_clamp,
            channels_last=channels_last,
        )

    def get_output_dim(self) -> int:
        return (self.group_size - 1) * self.dim

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        bn, c_in, h, w = x.shape
        num_groups = bn // self.group_size
        y = self.transform(x, c)  # [bn, dim, h, w]
        y = y.reshape(num_groups, self.group_size, self.dim, h, w)  # [num_groups, group_size, dim, h, w]
        d = y[:, 1:] - y[:, :-1]  # [num_groups, group_size - 1, dim, h, w]
        d = d.view(num_groups, (self.group_size - 1) * self.dim, h, w)  # [num_groups, (group_size - 1) * c, h, w]

        return d


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()

        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = (
            MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels)
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp
        )
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

        self.dist_predictor = None

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == "skip":
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)
        dist_preds = None if self.dist_predictor is None else self.dist_predictor(x.flatten(1))  # [batch_size]
        x = self.fc(x.flatten(1))
        x = self.out(x)  # [batch_size, out_dim]

        if not self.dist_predictor is None:
            # If one uncomments this, then we'll encounter a DDP consistency error for some reason
            x = x + dist_preds.sum() * 0.0

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))  # [batch_size, 1]

        assert x.dtype == dtype
        return x, dist_preds


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
        num_frames_per_sample=2,
        agg_concat_res=16,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.num_frames_per_sample = num_frames_per_sample
        self.agg_concat_res = agg_concat_res

        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        self.time_encoder = TemporalDifferenceEncoder()

        if self.c_dim == 0 and self.time_encoder is None:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        conditioning_dim = c_dim + (0 if self.time_encoder is None else self.time_encoder.get_total_dim())
        cur_layer_idx = 0

        num_frames_div_factor = 2
        self.diff_transform = FeatDiffLayer(
            in_channels=channels_dict[agg_concat_res] // num_frames_div_factor,
            dim=0,
            resolution=agg_concat_res,
            c_dim=conditioning_dim,
            conv_clamp=conv_clamp,
            num_frames_per_sample=num_frames_per_sample,
        )

        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]

            # Adjust numbers of channels
            if res // 2 == agg_concat_res:
                out_channels = out_channels // num_frames_div_factor

            if res == agg_concat_res:
                in_channels = (in_channels // num_frames_div_factor) * num_frames_per_sample
                in_channels += 0 if self.diff_transform is None else self.diff_transform.get_output_dim()

            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                c_dim=conditioning_dim,
                agg_concat_res=agg_concat_res,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers

        if self.c_dim > 0 or not self.time_encoder is None:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=conditioning_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs
        )

    def forward(self, img, c, t, **block_kwargs):
        # TODO: pass img in [b, c, t, h, w] shape instead of [b * t, c, h, w]

        assert len(img) == t.shape[0] * t.shape[1], f"Wrong shape: {img.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        if not self.time_encoder is None:
            t_embs = self.time_encoder(t.view(-1, self.num_frames_per_sample))  # [batch_size, t_dim]
            c_orig = torch.cat([c, t_embs], dim=1)  # [batch_size, c_dim + t_dim]
            c = c_orig.repeat_interleave(t.shape[1], dim=0)  # [batch_size * num_frames, c_dim + t_dim]

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            if res == self.agg_concat_res:
                d = (
                    None if self.diff_transform is None else self.diff_transform(x, c)
                )  # [batch_size, num_frames - 1, diff_c, h, w]
                x = x.view(-1, self.num_frames_per_sample, *x.shape[1:])  # [batch_size, num_frames, c, h, w]
                x = x.view(x.shape[0], -1, *x.shape[3:])  # [batch_size, num_frames * c, h, w]
                x = (
                    x if self.diff_transform is None else torch.cat([x, d], dim=1)
                )  # [batch_size, num_frames * c + (num_frames - 1) * d_dim, h, w]
                c = c_orig
            x, img = block(x, img, c, **block_kwargs)

        cmap = None
        if self.c_dim > 0 or not self.time_encoder is None:
            assert c.shape[1] > 0
        if c.shape[1] > 0:
            cmap = self.mapping(None, c)
        x, dist_preds = self.b4(x, img, cmap)
        x = x.squeeze(1)  # [batch_size]

        return {"image_logits": x, "dist_preds": dist_preds}
