from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from .hb import HB
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
import numpy as np
import cv2
import os

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

def get_hb(size='l', 
              pretrain=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViClip-InternVid-10M-FLT.pth"),
              text_lora=False,
              det_token_num=100,
              num_frames=9):
    
    tokenizer = _Tokenizer()
    hb_model = HB(tokenizer=tokenizer,
                        size=size,
                        pretrain=pretrain,
                        det_token_num=det_token_num,
                        num_frames=num_frames,
                        text_lora=text_lora)
    m = {'hb':hb_model, 'tokenizer':tokenizer}
    
    return m
        
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)
        
        prob = out_logits
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for i in range(bs):
            human_idx = torch.where(human_labels[i] == 0) # obtain boxes where human is detected
            human_scores_kept = human_scores[i][human_idx] # filter boxes where human is detected
            prob_kept = prob[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            
            human_idx = torch.where(human_scores_kept >= human_conf) # obtain boxes where human conf > thresh (default=0.7)
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx]
            
            human_idx = batched_nms(boxes_kept, human_scores_kept, torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5) # extra NMS
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx].int()

            final_scores = []
            final_labels = []
            finalboxes = []
            for i in range(len(human_idx)):
                box = boxes_kept[i]
                gt = torch.where(prob_kept[i] >= thresh)[0]
                gt_conf = (prob_kept[i][gt] + 1) / 2
                final_scores.extend(gt_conf)
                final_labels.extend(gt)
                for _ in range(len(gt)):
                    finalboxes.append(box)
            final_scores = torch.stack(final_scores) if len(final_scores) != 0 else torch.empty(0)
            final_labels = torch.stack(final_labels) if len(final_labels) != 0 else torch.empty(0)
            finalboxes = torch.stack(finalboxes) if len(finalboxes) != 0 else torch.empty(0)
            
            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        return results

class PostProcessViz(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, imgsize, human_conf=0.7, Aaug=None, thresh=0.25):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_human, out_logits, out_bbox = outputs['human_logits'], outputs['pred_logits'], outputs['pred_boxes']

        if Aaug is not None:
            out_logits = out_logits @ Aaug

        human_prob = F.softmax(out_human, -1)
        human_scores, human_labels = human_prob[...,].max(-1)
        
        #prob = F.sigmoid(out_logits)
        prob = out_logits
        #scores, labels = prob.max(-1)
        
        boxes = box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([imgsize[1], imgsize[0], imgsize[1], imgsize[0]]).to(boxes.device)
        boxes = boxes * scale_fct

        results = []
        bs = out_human.shape[0]
        for i in range(bs):
            human_idx = torch.where(human_labels[i] == 0) # obtain boxes where human is detected
            human_scores_kept = human_scores[i][human_idx] # filter boxes where human is detected
            #scores_kept = scores[i][human_idx] # obtain boxes where human is detected
            #labels_kept = labels[i][human_idx]
            prob_kept = prob[i][human_idx]
            boxes_kept = boxes[i][human_idx]
            
            human_idx = torch.where(human_scores_kept >= human_conf) # obtain boxes where human conf > thresh (default=0.7)
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            #scores_kept = scores_kept[human_idx] # obtain boxes where human is detected
            #labels_kept = labels_kept[human_idx]
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx]
            
            human_idx = batched_nms(boxes_kept, human_scores_kept, torch.zeros(len(human_scores_kept)).to(human_scores_kept.device), 0.5) # extra NMS
            human_scores_kept = human_scores_kept[human_idx] # filter boxes where human is detected
            #scores_kept = scores_kept[human_idx] # obtain boxes where human is detected
            #labels_kept = labels_kept[human_idx]
            prob_kept = prob_kept[human_idx]
            boxes_kept = boxes_kept[human_idx].int()

            final_scores = []
            final_labels = []
            finalboxes = []
            for i in range(len(human_idx)):
                box = boxes_kept[i]
                gt = torch.where(prob_kept[i] >= thresh)[0]
                gt_conf = (prob_kept[i][gt] + 1) / 2
                final_scores.append(gt_conf)
                final_labels.append(gt)
                finalboxes.append(box)
            #final_scores = torch.stack(final_scores)
            #final_labels = torch.stack(final_labels)
            #finalboxes = torch.stack(finalboxes)
            
            results.append({'scores': final_scores,
                            'labels': final_labels,
                            'boxes': finalboxes})

        return results
