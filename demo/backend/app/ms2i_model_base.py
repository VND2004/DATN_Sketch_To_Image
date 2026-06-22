import torch
import torch.nn as nn
import torch.nn.functional as F

from .ms2i_model import (
    RepConv3, Block, DownSample, UpSample, SkipConnection, RepAttnConfig, FFNConfig
)

class Head(nn.Module):
    def __init__(self, in_channels, out_channels=3, deploy=False, last_act=None):
        super().__init__()
        hidden = max(in_channels // 2, 8)
        self.pre = RepConv3(in_channels, hidden, 1, deploy=deploy)
        self.to_rgb = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.last_act = last_act if last_act is not None else nn.Identity()

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.pre, RepConv3):
            self.pre.fuse()

    def forward(self, x):
        x = F.gelu(self.pre(x))
        return self.last_act(self.to_rgb(x))

class MS2I(nn.Module):
    """Structure-based disentangled MS2I generator."""

    def __init__(
        self,
        input_shape=(3, 256, 256),
        deploy=False,
        dims=[48, 96, 192, 384],
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 2, 4],
        bias=True,
        last_act=None,
    ):
        super().__init__()
        assert len(dims) == len(num_blocks) == len(num_heads), "Length of dims, num_blocks and num_heads must be the same"
        self.input_shape = input_shape
        self.deploy = deploy
        self.dims = dims
        self.num_blocks = num_blocks
        self.bias = bias
        self.num_heads = num_heads

        # Structure encoder: sketch is the only input here, so early features stay layout-dominant.
        self.stem = nn.Conv2d(3, dims[0], kernel_size=7, stride=4, padding=3, bias=bias)

        layers = []
        down_convs = []
        for idx in range(len(dims)):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            block = Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias')
            if idx < len(dims) - 1:
                down_convs.append(DownSample(dims[idx]))
            layers.append(block)
        self.bottleneck = layers[-1]
        self.encoder = nn.ModuleList(layers[:-1])
        self.downsample = nn.ModuleList(down_convs)

        layers = []
        up_convs = []
        skip_connections = []
        for stage, idx in enumerate(range(len(dims) - 2, -1, -1)):
            attn_cfg, ffn_cfg = self.build_cfg(dims[idx], num_heads[idx])
            up_convs.append(UpSample(dims[idx + 1]))
            skip_connections.append(SkipConnection(dims[idx]))
            layers.append(Block(num_blocks[idx], attn_cfg, ffn_cfg, norm_type='WithBias'))
        self.decoder = nn.ModuleList(layers)
        self.up_sample = nn.ModuleList(up_convs)
        self.skip = nn.ModuleList(skip_connections)

        self.head = Head(dims[0], out_channels=3, deploy=deploy, last_act=last_act)

    @torch.no_grad()
    def fuse(self):
        for block in self.encoder:
            block.fuse()
        self.bottleneck.fuse()
        for block in self.decoder:
            block.fuse()
        self.head.fuse()

    def build_cfg(self, dim, head):
        attn_cfg = RepAttnConfig(
            dim=dim,
            num_heads=head,
            deploy=self.deploy,
        )
        ffn_cfg = FFNConfig(dim=dim, expansion_factor=1)
        return attn_cfg, ffn_cfg

    def forward(self, sketch, return_latents=False):

        x = self.stem(sketch)
        structure_feats = []
        for blk, down in zip(self.encoder, self.downsample):
            x = blk(x)
            structure_feats.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for blk, up, skip in zip(self.decoder, self.up_sample, self.skip):
            x = up(x)
            cur_feat = structure_feats.pop()
            x = skip(x, cur_feat)
            x = blk(x)

        x = F.interpolate(x, size=self.input_shape[1:], mode='bilinear', align_corners=False)
        out = self.head(x)
        if return_latents:
            return out, {'structure': x}
        return out
