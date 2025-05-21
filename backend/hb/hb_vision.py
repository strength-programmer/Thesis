#!/usr/bin/env python
import os
import logging
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

import torch.utils.checkpoint as checkpoint

import math

# On P1, model extracted from https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
MODEL_PATH = ''
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "ViCLIP-L_InternVid-FLT-10M.pth"),
    "ViT-B/16": os.path.join(MODEL_PATH, "ViCLIP-B-InternVid-FLT-10M.pth"),
}

# YOLOS: Added Custom MLP
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
# PeFT: Added LoRA    
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class TransformerMLPwithLoRA(nn.Module):
    def __init__(self, d_model, dropout=0., det_token_num=0):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.drop1 = nn.Dropout(dropout)
        self.c_proj = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)
        
        self.det_token_num = det_token_num
        self.det_lora_fc = LoRALayer(d_model, d_model * 4)
        self.det_lora_proj = LoRALayer(d_model * 4, d_model)
        
    def forward(self, x):
        x, x_det = x[:-self.det_token_num], x[-self.det_token_num:]
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.c_proj(x)
        x = self.drop2(x)
        
        x_det = self.c_fc(x_det) + self.det_lora_fc(x_det)
        x_det = self.gelu(x_det)
        x_det = self.drop1(x_det)
        x_det = self.c_proj(x_det) + self.det_lora_proj(x_det)
        x_det = self.drop2(x_det)
        
        x = torch.cat([x, x_det], dim=0)
        
        return x

class TransformerMLP(nn.Module):
    def __init__(self, d_model, dropout=0.):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.drop1 = nn.Dropout(dropout)
        self.c_proj = nn.Linear(d_model * 4, d_model)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.drop1(x)
        x = self.c_proj(x)
        x = self.drop2(x)
        return x
        
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., dropout=0., det_token_num=100, lora=False):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        if lora and det_token_num > 0:
            self.mlp = TransformerMLPwithLoRA(d_model, dropout, det_token_num)
        else:
            self.mlp = TransformerMLP(d_model, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.det_token_num = det_token_num

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0., det_token_num=100, lora=False, num_frames=9):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        in_frame = num_frames
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width,
                                                         heads,
                                                         drop_path=dpr[idx],
                                                         dropout=dropout,
                                                         det_token_num=det_token_num,
                                                         lora=lora))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim=None, 
        kernel_size=1, num_frames=9, drop_path=0, checkpoint_num=0, dropout=0.,
        temp_embed=True, det_token_num=100, lora=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width, 
            (kernel_size, patch_size, patch_size), 
            (kernel_size, patch_size, patch_size), 
            (0, 0, 0), bias=False
        )

        self.input_resolution = input_resolution
        self.patch_size = patch_size
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))
        
        self.lora = lora
        self.transformer = Transformer(
            width, layers, heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout, det_token_num=det_token_num, lora=self.lora, num_frames=num_frames)

        self.ln_post = nn.LayerNorm(width)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(width, output_dim))
            if self.lora:
                self.det_proj_lora = LoRALayer(width, output_dim)
        else:
            self.proj = None
        
        self.dropout = nn.Dropout(dropout)
        
        #YOLOS: Added MLP for Human logits, BBoxes and Action logits
        self.human_embed = MLP(width, width, 2, 3)
        self.bbox_embed = MLP(width, width, 4, 3)
        
        
        self.det_token_num = det_token_num
        # YOLOS: Added [DET] Tokens
        if self.det_token_num > 0:
            self.det_token = nn.Parameter(torch.zeros(self.det_token_num, width))
            nn.init.normal_(self.det_token, std=0.02)
            
            # YOLOS: Added PE for [DET] Tokens
            self.det_positional_embedding = nn.Parameter(scale * torch.randn(self.det_token_num, width))
            nn.init.normal_(self.det_positional_embedding, std=0.02)

    def get_num_layers(self):
        return len(self.transformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding', 'temporal_positional_embedding'}
    
    def mask_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape

        # This is different from text as we are masking a fix number of tokens
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()

        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])

    # YOLOS: Added Online Interpolation for Positional Embedding 
    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:,None]
        patch_pos_embed = pos_embed[:, 1:, :]
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape


        P_H, P_W = self.input_resolution // self.patch_size, self.input_resolution // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        return scale_pos_embed

    def forward(self, x, masking_prob=0.0):
        _, _, _, in_H, in_W = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # [CLS] Token and spatial pos
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # YOLOS: interpolate PE on-the-fly
        # interpolate init pe
        temp_pos_embed = self.positional_embedding.to(x.dtype).unsqueeze(0)
        if self.positional_embedding.shape[1] != x.shape[1]:
            temp_pos_embed = self.InterpolateInitPosEmbed(temp_pos_embed, img_size=(in_H, in_W))
        else:
            temp_pos_embed = self.positional_embedding
        x = x + temp_pos_embed

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob)

        # [DET] Tokens and corresponding pos
        # x = torch.cat((cls_tokens, x), dim=1) Removed [CLS] Tokens
        if self.det_token_num > 0:
            det_tokens = self.det_token + self.det_positional_embedding
            det_tokens = det_tokens + torch.zeros(B, det_tokens.shape[0], det_tokens.shape[1]).to(det_tokens.device)
            x = torch.cat((x, det_tokens), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)

        x = self.ln_post(x)

        if self.det_token_num > 0:
            x = self.dropout(x[-self.det_token_num:])
        else:
            x = self.dropout(x.reshape(H*W, T, B, C).mean(1))
        
        x = x.permute(1, 0, 2) #NBD -> BND
        
        if self.proj is not None:
            class_scores = x @ self.proj
            if self.lora:
                class_scores += self.det_proj_lora(x)
        else:
            class_scores = x
        
        bboxes = checkpoint.checkpoint(self.bbox_embed, x, use_reentrant=False).sigmoid()
        if self.det_token_num == 0:
            box_bias = F.pad(torch.stack(torch.meshgrid(torch.linspace(0,1,W), torch.linspace(0,1,H))).reshape(2, W*H).permute(1,0), (0,2), 'constant', 0)
            bboxes = bboxes + box_bias.to(bboxes.device)
        human_scores = checkpoint.checkpoint(self.human_embed, x, use_reentrant=False)
        
        out = {'pred_logits': class_scores, 'pred_boxes': bboxes, 'human_logits': human_scores}

        return out

def inflate_weight(weight_2d, time_dim, center=True):
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d

def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                continue
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict['positional_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict['positional_embedding'] = new_pos_embed
    
    message = model.load_state_dict(state_dict, strict=False)

def clip_joint_b16(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0.,
    lora=False,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16, 
        width=768, layers=12, heads=12, output_dim=512,
        kernel_size=kernel_size, num_frames=num_frames, 
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        lora=lora,
    )
    # raise NotImplementedError
    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-B/16"
        
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()

def clip_joint_l14(
    pretrained=False, input_resolution=224, kernel_size=1,
    center=True, num_frames=9, drop_path=0., checkpoint_num=0, det_token_num=100,
    dropout=0.,
    lora=False,
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        kernel_size=kernel_size, num_frames=num_frames, 
        checkpoint_num=checkpoint_num,
        drop_path=drop_path, det_token_num=det_token_num,
        lora=lora,
    )
    
    if pretrained:
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            model_name = "ViT-L/14"
        state_dict = torch.load(_MODELS[model_name], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()
