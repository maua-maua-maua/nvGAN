import numpy as np
import torch
from torch import Tensor

from torch_utils import misc, persistence
from torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d

from .layers import Conv2dLayer, FullyConnectedLayer, MappingNetwork
from .motion import GenInput, MotionEncoder

# ----------------------------------------------------------------------------


@misc.profiled_function
def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1 / np.sqrt(in_channels * kh * kw) / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight
        )
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
        instance_norm=False,  # Use instance norm?
        use_noise=False,
        fmm_resolutions=[],
        fmm_activation="demod",
    ):
        super().__init__()

        self.resolution = resolution
        self.use_noise = use_noise
        self.up = up
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.fmm_activation = fmm_activation

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.instance_norm = instance_norm

    def forward(self, x, w, noise_mode="random", fused_modconv=True, gain=1):
        assert noise_mode in ["random", "const", "none"]
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = (
                torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        flip_weight = self.up == 1  # slightly faster

        if self.instance_norm:
            x = x / (x.std(dim=[2, 3], keepdim=True) + 1e-8)  # [batch_size, c, h, w]

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        motion_w_dim,  # Motion code size
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        input_resolution=4,
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim

        if resolution <= input_resolution:
            self.resolution = input_resolution
            self.up = 1
            self.input_resolution = input_resolution
        else:
            self.resolution = resolution
            self.up = 2
            self.input_resolution = resolution // 2

        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.kernel_size = 3
        self.use_instance_norm = False

        if in_channels == 0:
            self.input = GenInput(out_channels, w_dim, motion_w_dim=motion_w_dim)
            conv1_in_channels = self.input.total_dim
        else:
            conv0_in_channels = in_channels

            # We are not using instance norm in conv0, because we concatenate coords to it (sometimes)
            # and some coords can be all-zero
            self.conv0 = SynthesisLayer(
                conv0_in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=self.resolution,
                up=self.up,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                kernel_size=self.kernel_size,
                instance_norm=False,
                **layer_kwargs,
            )
            self.num_conv += 1
            conv1_in_channels = out_channels

        self.conv1 = SynthesisLayer(
            conv1_in_channels,
            out_channels,
            w_dim=w_dim,
            resolution=self.resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            kernel_size=self.kernel_size,
            instance_norm=self.use_instance_norm,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(
                out_channels, img_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=self.up,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, ws, t=None, motion_w=None, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or (isinstance(x, Tensor) and int(x.shape[0]) == 1)
                )

        # Input.
        if self.in_channels == 0:
            conv1_w = next(w_iter)
            x = self.input(conv1_w, t=t, motion_w=motion_w)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.input_resolution, self.input_resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, conv1_w, fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            conv0_w = next(w_iter)

            x = self.conv0(x, conv0_w, fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.input_resolution, self.input_resolution])

            if self.up == 2:
                img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.motion_encoder = MotionEncoder(z_dim, c_dim, w_dim)
        self.motion_w_dim = self.motion_encoder.get_output_dim()

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=self.w_dim,
                motion_w_dim=self.motion_w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv

            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{res}", block)

    def forward(self, ws, t=None, c=None, l=None, motion_noise=None, motion_w=None, **block_kwargs):
        assert len(ws) == len(c) == len(t), f"Wrong shape: {ws.shape}, {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        block_ws = []

        if motion_w is None:
            motion_info = self.motion_encoder(c, t, l=l, w=ws[:, 0], motion_noise=motion_noise)  # [batch_size * num_frames, motion_w_dim]
            motion_w = motion_info["motion_w"]  # [batch_size * num_frames, motion_w_dim]

        ws = ws.repeat_interleave(t.shape[1], dim=0)  # [batch_size * num_frames, num_ws, w_dim]

        with torch.autograd.profiler.record_function("split_ws"):
            ws = ws.to(torch.float32)
            w_idx = 0

            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{res}")
            x, img = block(x, img, cur_ws, t=t, motion_w=motion_w, **block_kwargs)

        return img


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()

        self.sampling_dict = {
            "fps": 24,
            "max_num_frames": 1024,
            "type": "random",
            "dists": [1, 2, 4, 8, 16, 32],
            "num_frames_per_sample": 2,
        }
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, t, l, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        assert len(z) == len(c) == len(t), f"Wrong shape: {z.shape}, {c.shape}, {t.shape}"
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        ws = self.mapping(z, c, l=l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)  # [batch_size, num_ws, w_dim]
        img = self.synthesis(ws, t=t, c=c, l=l, **synthesis_kwargs)  # [batch_size * num_frames, c, h, w]

        return img
