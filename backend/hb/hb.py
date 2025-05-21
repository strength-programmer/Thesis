import os
import logging
from pkg_resources import packaging

import torch
import numpy as np
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import math

from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .hb_vision import clip_joint_l14, clip_joint_b16
from .hb_text import clip_text_l14, clip_text_b16

class HB(nn.Module):
    def __init__(self,  
                 tokenizer=None, 
                 size='l',
                 pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
                 text_lora=False,
                 det_token_num=100,
                 num_frames=9):
        super(HB, self).__init__()
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = _Tokenizer()
        self.max_txt_l = 32

        if size.lower() == 'l':
            self.vision_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.vision_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")
    
        self.vision_encoder_pretrained = False
        self.inputs_image_res = 224
        self.vision_encoder_kernel_size = 1
        self.vision_encoder_center = True
        self.video_input_num_frames = num_frames
        self.vision_encoder_drop_path_rate = 0 #0.1
        self.vision_encoder_checkpoint_num = 24
        self.is_pretrain = pretrain
        self.vision_width = 1024
        self.text_width = 768 
        self.embed_dim = 768 
        self.masking_prob = 0 #0.9
        
        # Extra techniques for vision encoder
        self.det_token_num = det_token_num
        if self.det_token_num > 0:
            print('Type: [DET] regression')
        else:
            print('Type: [PATCH] regression')
        
        if size.lower() == 'l':
            self.text_encoder_name = 'vit_l14'
        elif size.lower() == 'b':
            self.text_encoder_name = 'vit_b16'
        else:
            raise NotImplementedError(f"Size {size} not implemented")
        
        self.text_encoder_pretrained = False#'bert-base-uncased'
        self.text_encoder_d_model = 768

        self.text_encoder_vocab_size = 49408
        
        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.text_encoder = self.build_text_encoder(lora=text_lora)

        if pretrain:
            state_dict = torch.load(pretrain, map_location='cpu')['model']
            state_dict = interpolate_pos_embed_vit(state_dict, self) # interpolate temporal embeddings
            self.load_state_dict(state_dict, strict=False)
        
        # Freeze text weights
        if text_lora:
            print('Frozen: ViCLIP text encoder and added LoRA')
            self.freeze_text(text_lora=text_lora)
        else:
            print('Frozen: ViCLIP text encoder')
            self.freeze_text()

        if size.lower() == 'l':
            self.embed_dim = 768
        elif size.lower() == 'b':
            self.embed_dim = 512
        else:
            raise NotImplementedError(f"Size {size} not implemented")
        
        # Learnable temperature param
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07)) #np.log(5))
        self.logit_bias = nn.Parameter(torch.tensor(0.))

    def freeze_text(self, text_lora=False):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False
            
        if text_lora:
            for n, p in self.text_encoder.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )

        return ret

    def forward(self, video, raw_text, log_generation=None):
        """forward and calculate loss.

        Args:
            video (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            input_ids (torch.Tensor): The tokenized texts. Shape: [B, L].

        Returns: TODO

        """
        # temperature
        logit_scale = self.logit_scale.exp() #torch.clamp(self.logit_scale.exp(), max=100)

        outputs = self.encode_vision(video)
        text_embeds = self.encode_text(raw_text)
            
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # convert vision logits to closed-set logits
        outputs['pred_logits'] = F.normalize(outputs['pred_logits'], dim=-1) @ text_embeds.T
        if self.training:
            outputs['pred_logits'] = logit_scale * outputs['pred_logits'] + self.logit_bias
        
        return outputs
        

    def encode_vision(self, image, test=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,T,L,C].
            - pooled_vision_embeds (torch.Tensor): The pooled features. Shape: [B,T,C].

        """
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)

        if not test and self.masking_prob > 0.0:
            return self.vision_encoder(
                image, masking_prob=self.masking_prob
            )

        return self.vision_encoder(image)

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        device = next(self.text_encoder.parameters()).device
        text = self.tokenize(
            text, context_length=self.max_txt_l
        ).to(device)
        text_embeds = self.text_encoder(text)
        return text_embeds

    def build_vision_encoder(self, lora=False):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = self.vision_encoder_name
        if encoder_name == "vit_l14":
            vision_encoder = clip_joint_l14(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
            )
        elif encoder_name == "vit_b16":
            vision_encoder = clip_joint_b16(
                pretrained=self.vision_encoder_pretrained,
                input_resolution=self.inputs_image_res,
                kernel_size=self.vision_encoder_kernel_size,
                center=self.vision_encoder_center,
                num_frames=self.video_input_num_frames,
                drop_path=self.vision_encoder_drop_path_rate,
                checkpoint_num=self.vision_encoder_checkpoint_num,
                det_token_num=self.det_token_num,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")
            
        return vision_encoder

    def build_text_encoder(self, lora=False):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.text_encoder_name
        
        if encoder_name == "vit_l14":
            text_encoder = clip_text_l14(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
                lora=lora,
            )
        elif encoder_name == "vit_b16":
            text_encoder = clip_text_b16(
                pretrained=self.text_encoder_pretrained,
                context_length=self.max_txt_l,
                vocab_size=self.text_encoder_vocab_size,
                checkpoint_num=0,
                lora=lora,
            )
        else:
            raise NotImplementedError(f"Not implemented: {encoder_name}")

        return text_encoder

    def tokenize(self, texts, context_length='auto', truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        if context_length == 'auto':
            context_length = max(len(tokens) for tokens in all_tokens)
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
        
def interpolate_temporal_pos_embed(temp_embed_old, num_frames_new):
    """
    temp_embed_old: (1, num_frames_old, 1, d)
    Returns:
        temp_embed_new: (1, num_frames_new, 1, d)
    """
    temp_embed_old = temp_embed_old.squeeze(2).permute(
        0, 2, 1
    )  # (1, d, num_frames_old)
    temp_embed_new = F.interpolate(
        temp_embed_old, num_frames_new, mode="linear"
    )  # (1, d, num_frames_new)
    temp_embed_new = temp_embed_new.permute(0, 2, 1).unsqueeze(
        2
    )  # (1, num_frames_new, 1, d)
    return temp_embed_new

def load_temp_embed_with_mismatch(temp_embed_old, temp_embed_new, add_zero=False):
    """
    Add/Remove extra temporal_embeddings as needed.
    https://arxiv.org/abs/2104.00650 shows adding zero paddings works.

    temp_embed_old: (1, num_frames_old, 1, d)
    temp_embed_new: (1, num_frames_new, 1, d)
    add_zero: bool, if True, add zero, else, interpolate trained embeddings.
    """
    # TODO zero pad
    num_frms_new = temp_embed_new.shape[1]
    num_frms_old = temp_embed_old.shape[1]
    if num_frms_new > num_frms_old:
        if add_zero:
            temp_embed_new[
                :, :num_frms_old
            ] = temp_embed_old  # untrained embeddings are zeros.
        else:
            temp_embed_new = interpolate_temporal_pos_embed(temp_embed_old, num_frms_new)
    elif num_frms_new < num_frms_old:
        temp_embed_new = temp_embed_old[:, :num_frms_new]
    else:  # =
        temp_embed_new = temp_embed_old
    return temp_embed_new

def interpolate_pos_embed_vit(state_dict, new_model):
    key = "vision_encoder.temporal_positional_embedding"
    if key in state_dict:
        vision_temp_embed_new = new_model.state_dict()[key]
        vision_temp_embed_new = vision_temp_embed_new.unsqueeze(2)  # [1, n, d] -> [1, n, 1, d]
        vision_temp_embed_old = state_dict[key]
        vision_temp_embed_old = vision_temp_embed_old.unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            vision_temp_embed_old, vision_temp_embed_new, add_zero=False
        ).squeeze(2)

    key = "text_encoder.positional_embedding"
    if key in state_dict:
        text_temp_embed_new = new_model.state_dict()[key]
        text_temp_embed_new = text_temp_embed_new.unsqueeze(0).unsqueeze(2)  # [n, d] -> [1, n, 1, d]
        text_temp_embed_old = state_dict[key]
        text_temp_embed_old = text_temp_embed_old.unsqueeze(0).unsqueeze(2)

        state_dict[key] = load_temp_embed_with_mismatch(
            text_temp_embed_old, text_temp_embed_new, add_zero=False
        ).squeeze(2).squeeze(0)
    return state_dict
