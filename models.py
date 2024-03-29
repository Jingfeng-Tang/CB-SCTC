import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

import math
from info_nce import InfoNCE

__all__ = [
    'deit_small_MCTformerV1_patch16_224'
]


class MCTformerV1(VisionTransformer):
    def __init__(self, last_opt='average', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        if last_opt == 'fc':
            self.head = nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=self.embed_dim, groups=self.num_classes)
            self.head.apply(self._init_weights)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12, shallow_block=9, sparsity=10):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        attn_weights = []
        # print(f'222shallow_block: {shallow_block}')
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x, sparsity)
            if i == shallow_block:  # 10改9----------------------------------------------------------------------
                # print(f'333shallow_block: {shallow_block}')
                # a = []
                # b = a[10]
                block11_x = x.clone()
            elif i == 11:
                block12_x = x.clone()
            if len(self.blocks) - i <= n:
                attn_weights.append(weights_i)

        return x[:, 0:self.num_classes], attn_weights, block11_x, block12_x

    def forward(self, x, n_layers=12, return_att=False, shallow_block=9, sparsity=10):
        # print(f'111shallow_block: {shallow_block}')
        # a = []
        # b = a[10]
        x, attn_weights, block11_x, block12_x = self.forward_features(x, shallow_block=shallow_block, sparsity=sparsity)
        # print(f'block11_x.shape: {block11_x.shape}')  # 64*216*384
        # block_11_clstoken = block11_x[:, 0:20, :]
        # block_12_clstoken = block12_x[:, 0:20, :]
        block11_x = block11_x[:, 0:20, :]
        block12_x = block12_x[:, 0:20, :]
        # print(f'block_11_clstoken.shape: {block_11_clstoken.shape}')    # 64*20*384
        # # 加投影层
        # block_11_x_pj = self.pj1(block11_x)
        # block_11_x_pj = self.act(block_11_x_pj)
        # block_11_clstoken = self.pj2(block_11_x_pj)
        #
        # block_12_x_pj = self.pj3(block12_x)
        # block_12_x_pj = self.act(block_12_x_pj)
        # block_12_clstoken = self.pj4(block_12_x_pj)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:]
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x.mean(-1)

        if return_att:
            return x_cls_logits, mtatt, patch_attn
        else:
            return x_cls_logits, block11_x, block12_x


@register_model
def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model