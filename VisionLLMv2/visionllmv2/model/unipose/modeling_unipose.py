# ------------------------------------------------------------------------
# ED-Pose
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from torchvision.ops.boxes import nms
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Optional
from timm.models.layers import DropPath
from .utils.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from .utils import box_ops
from torch import Tensor
import numpy as np
from .utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from .utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                        accuracy, get_world_size, interpolate,
                        is_dist_avail_and_initialized, inverse_sigmoid, is_main_process)
from .utils.model_utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position, get_sine_pos_embed, _get_clones
from .utils.model_utils import sigmoid_focal_loss, OKSLoss
from .utils.utils import targets_to


from .ops.modules import MSDeformAttn
from .configuration_unipose import UniPoseConfig

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput


#------------------- for internimage -----------------------
import warnings

import torch.nn as nn
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
from mmcv.runner import _load_checkpoint
from mmcv.cnn import constant_init, trunc_normal_init
from mmdet.utils import get_root_logger
try:
    from ..ops_dcnv3 import modules as opsm
except:
    warnings.warn("please compile DCNv3 operator if taking InternImage as backbone!")

@dataclass
class UniPoseOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    # logits: torch.FloatTensor = None           
    pred_logits: torch.FloatTensor = None       # [bs, nq, k]
    pred_boxes: torch.FloatTensor = None        # [bs, nq, 4]
    pred_keypoints: torch.FloatTensor = None    # [bs, nq, 68*3]
    # dn_meta: Dict = None

class UniPose(PreTrainedModel):
    """ This is the Cross-Attention Detector module that performs object detection """
    config_class = UniPoseConfig

    def __init__(self, config: UniPoseConfig):
        super().__init__(config)

        self.backbone = build_backbone(config)
        self.transformer = build_deformable_transformer(config)

        self.num_queries = config.num_queries
        self.num_classes = config.num_classes  # 2
        self.hidden_dim = hidden_dim = self.transformer.d_model
        self.num_feature_levels = config.num_feature_levels #1
        self.nheads = config.nheads
        self.use_label_enc = config.use_label_enc
        # NOTE: use encoded_text, not nn.Embedding
        # if config.use_label_enc:
        #     self.label_enc = nn.Embedding(config.dn_labelbook_size + 1, hidden_dim)
        # else:
        #     raise NotImplementedError
        #     self.label_enc = None
        self.max_text_len = 256
        self.binary_query_selection = config.binary_query_selection  # false
        self.sub_sentence_present = config.sub_sentence_present      # true, not used

        # setting query dim
        self.query_dim = config.query_dim
        assert config.query_dim == 4
        self.random_refpoints_xy = config.random_refpoints_xy        # false
        self.fix_refpoints_hw = config.fix_refpoints_hw              # -1

        # for dn training
        self.num_patterns = config.num_patterns                      # 0
        self.dn_number = config.dn_number                            # 100
        self.dn_box_noise_scale = config.dn_box_noise_scale          # 1.0
        self.dn_label_noise_ratio = config.dn_label_noise_ratio      # 0.5
        self.dn_labelbook_size = config.dn_labelbook_size            # 2000
        self.use_cdn = config.use_cdn                                # true

        # for project queries from llm
        self.projection_llava = MLP(config.l_hidden_size, hidden_dim, hidden_dim, 3)
        self.projection_kpt_llava = MLP(config.l_hidden_size, hidden_dim, hidden_dim, 3)

        # prepare input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert config.two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn
                )])

        self.aux_loss = config.aux_loss  # true
        self.box_pred_damping = box_pred_damping = None  # not used

        self.iter_update = True
        assert self.iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = config.dec_pred_class_embed_share  # true
        self.dec_pred_bbox_embed_share = config.dec_pred_bbox_embed_share    # true
        # prepare class & box embed
        _class_embed = ContrastiveAssign()



        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _pose_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_embed.layers[-1].bias.data, 0)

        # box
        if config.dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(self.transformer.num_decoder_layers)]
        if config.dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(self.transformer.num_decoder_layers)]

        # pose, predict pose at the last 5 decoder layers
        if config.dec_pred_bbox_embed_share:

            pose_embed_layerlist = [_pose_embed for i in
                                    range(self.transformer.num_decoder_layers - config.num_box_decoder_layers + 1)]
        else:
            pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in
                                    range(self.transformer.num_decoder_layers - config.num_box_decoder_layers + 1)]

        pose_hw_embed_layerlist = [_pose_hw_embed for i in
                                   range(self.transformer.num_decoder_layers - config.num_box_decoder_layers)]


        self.num_box_decoder_layers = config.num_box_decoder_layers  # 2
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.num_body_points = config.num_body_points                # 68, max num_kpts for an instance
        self.pose_embed = nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed = nn.ModuleList(pose_hw_embed_layerlist)

        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.pose_hw_embed = self.pose_hw_embed

        self.transformer.decoder.num_body_points = config.num_body_points


        # two stage
        self.two_stage_type = config.two_stage_type
        self.two_stage_add_query_num = config.two_stage_add_query_num
        assert config.two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(config.two_stage_type)  # standard
        if config.two_stage_type != 'no':
            if config.two_stage_bbox_embed_share:
                assert config.dec_pred_class_embed_share and config.dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if config.two_stage_class_embed_share:
                assert config.dec_pred_class_embed_share and config.dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(config.two_stage_add_query_num)

        self.decoder_sa_type = config.decoder_sa_type  # sa
        assert config.decoder_sa_type in ['sa', 'ca_label', 'ca_content']  
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca
        if config.decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(config.num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None


        # ----------------------------------------------------------
        # matcher and criterion
        self.matcher = build_matcher(config)
        weight_dict = {'loss_ce': config.cls_loss_coef, 'loss_bbox': config.bbox_loss_coef}
        weight_dict['loss_giou'] = config.giou_loss_coef
        weight_dict['loss_keypoints'] = config.keypoint_loss_coef
        weight_dict['loss_oks'] = config.keypoint_loss_coef
        # add aux weight
        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(config.dec_layers - 1):
                for k, v in weight_dict.items():
                    if (i < self.num_box_decoder_layers and 'keypoints' in k) or (i < self.num_box_decoder_layers and 'oks' in k):
                        continue  # first layer does not predict pose
                    aux_weight_dict.update({k + f'_{i}': v})
            aux_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items() if 'keypoints' not in k and 'oks' not in k})
            weight_dict.update(aux_weight_dict)
        # add dn weight
        if self.use_cdn:
            weight_dict_dn = {"loss_ce_dn": config.cls_loss_coef, "loss_bbox_dn": config.bbox_loss_coef,
                              "loss_giou_dn": config.giou_loss_coef}  # dn not predict pose
            aux_weight_dict_dn = {}
            for i in range(config.dec_layers - 1):
                aux_weight_dict_dn.update({k + f"_{i}": v for k, v in weight_dict_dn.items()})
            weight_dict.update(weight_dict_dn)
            weight_dict.update(aux_weight_dict_dn)

        losses = ['labels', 'boxes', 'keypoints']
        self.criterion = DNSetCriterion(matcher=self.matcher, weight_dict=weight_dict,
                             focal_alpha=config.focal_alpha, losses=losses,
                             num_body_points=config.num_body_points, num_box_decoder_layers=config.num_box_decoder_layers
        )
        

        self._reset_parameters()

    def open_set_transfer_init(self):
        for name, param in self.named_parameters():
            if 'fusion_layers' in name:
                continue
            if 'ca_text' in name:
                continue
            if 'catext_norm' in name:
                continue
            if 'catext_dropout' in name:
                continue
            if "text_layers" in name:
                continue
            if 'bert' in name:
                continue
            if 'bbox_embed' in name:
                continue
            if 'label_enc.weight' in name:
                continue
            if 'feat_map' in name:
                continue
            if 'enc_output' in name:
                continue

            param.requires_grad_(False)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        # false
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        # -1, pass
        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets: List = None, text_query: Dict = None,  img_metas: List[Dict] = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        '''
            text_query: {
                obj_querys: [bs, max_num_obj, num_embs, c]
                obj_query_masks: [bs, max_num_obj]
                kpt_querys: [bs, max_num_kpt, num_embs, c]
                kpt_query_masks: [bs, max_num_kpt]
            }
            e.g., max_num_obj=1, max_num_kpt=100

            TO Modify:
            tensor_list; kpts_embedding_text; kpt_mask
        '''

        # convert the class labels during training
        if self.training:
            assert img_metas is not None
            assert targets is not None
            new_targets = []
            for img_meta, target in zip(img_metas, targets):
                # img_meta, label: dict
                id2index = img_meta["id2index"]  # dict, store the class id to location index mapping 
                class_labels = copy.deepcopy(target["class_labels"])  # [num_gt_i,]
                class_labels_list = class_labels.tolist()
                new_class_labels = torch.LongTensor([id2index[label_i] for label_i in class_labels_list]).to(self.device)
                target["class_labels"] = new_class_labels
                new_targets.append(target)
            targets = new_targets

        # -----------------------------------------------------------------------
        # project obj and kpt embeddings
        bs, max_num_obj = text_query['obj_querys'].shape[:2]

        # for obj embedding
        encoded_text = self.projection_llava(text_query['obj_querys']).mean(-2)  # [bs, max_num_obj, c]
        # FIXME: do we need pad to 256 classes
        # encoded_text = torch.zeros((bs, 256, self.hidden_dim), dtype=samples.tensors.dtype).to(self.device)                              # [bs, 256, c], max_text_len=256
        # for kpt embedding
        kpt_embeddings_specific = torch.zeros((bs, self.num_body_points, self.hidden_dim), dtype=samples.tensors.dtype).to(self.device)  # [bs, 68, c], num_body_points=68
        for i in range(bs):
            # obj embedding
            # FIXME: do we need pad to 256 classes
            # n_obj_class = text_query['obj_query_masks'].sum()  # e.g., 1
            # encoded_text[i, :n_obj_class, :] = self.projection_llava(text_query['obj_querys'][i]).mean(-2)[:n_obj_class, :] # [max_num_obj, c] -> [n_obj_class, c]
            # kpt embedding
            n_kpt = text_query['kpt_query_masks'][i].sum()        # e.g., 17
            assert n_kpt <= self.num_body_points, f"The max number for keypoints is {self.num_body_points}, {n_kpt}"
            kpt_embeddings_specific[i, :n_kpt, :] = self.projection_kpt_llava(text_query['kpt_querys'][i]).mean(-2)[:n_kpt, :] # [max_num_kpt, c] -> [n_kpt, c]
        
        kpt_vis = text_query["kpt_query_masks"][:, :self.num_body_points]  # [bs, 68]
        kpt_mask = torch.cat((torch.ones_like(kpt_vis, device=kpt_vis.device)[..., 0].unsqueeze(-1), kpt_vis), dim=-1) # [bs, 69], +1 for box. this is for dn. valid is 1.

        # FIXME: do we need pad to 256 classes
        # for obj embedding
        # num_classes = encoded_text.shape[1] # 256
        # text_self_attention_masks = torch.eye(num_classes).unsqueeze(0).expand(bs, -1, -1).bool().to(self.device)  # [bs, 256, 256], valid is 1
        # text_token_mask = torch.zeros(bs,num_classes).to(self.device)>0 # [bs, 256], valid is 1
        # position_ids = torch.zeros(bs, num_classes).to(self.device)     # [bs, 256]
        # # generate from obj_query_masks
        # obj_text_self_attention_masks, obj_position_ids = generate_masks_with_text_query_masks(text_query['obj_query_masks'])
        # text_self_attention_masks[:, :max_num_obj, :max_num_obj] = obj_text_self_attention_masks.bool()  # valid is 1
        # position_ids[:, :max_num_obj] = obj_position_ids
        # text_token_mask[:, :max_num_obj] = text_query['obj_query_masks'].bool()  # valid is 1

        # for obj embedding
        text_self_attention_masks, position_ids = generate_masks_with_text_query_masks(text_query['obj_query_masks'])
        text_token_mask = text_query['obj_query_masks'].bool()  # valid is 1

        # for i in range(bs):
        #     text_token_mask[i,:1]=True #TODO
        # position_ids = torch.zeros(bs, num_classes).to(self.device)
        # for i in range(bs):
        #     position_ids[i,:1]= 1 #TODO

        text_dict = {
            'encoded_text': encoded_text, # bs, 195, d_model
            'text_token_mask': text_token_mask, # bs, 195
            'position_ids': position_ids, # bs, 195
            'text_self_attention_masks': text_self_attention_masks # bs, 195,195
        }

        # ----------------------------------------------
        # start unipose
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples, size_divisibility=32)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # for dn training
        # if self.label_enc is not None:
        #     label_enc = self.label_enc
        # else:
        #     raise NotImplementedError
        #     label_enc = encoded_text
        if self.training and self.dn_number > 0 :
            # input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta =\
            #     prepare_for_cdn_v1(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
            #                     training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
            #                     hidden_dim=self.hidden_dim,label_enc=self.label_enc,kpt_mask=kpt_mask,
            #                     num_body_points=self.num_body_points)
            input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta = self.prepare_for_cdn(
                targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale, self.num_queries, self.hidden_dim, 
                self.dn_labelbook_size, encoded_text, kpt_mask=kpt_mask, num_body_points=self.num_body_points, num_heads=self.nheads
            )
        else:
            # assert targets is None
            input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta = self.prepare_for_mask(kpt_mask)


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss,
                                                                                 input_query_label, attn_mask, attn_mask2,
                                                                                 text_dict, dn_meta,targets,kpt_embeddings_specific)

        # In case num object=0
        # if self.label_enc is not None:
        #     hs[0] += self.label_enc.weight[0, 0] * 0.0

        num_group = 50  # pose query number
        effective_dn_number = dn_meta['pad_size'] if (self.training and dn_meta is not None) else 0
        outputs_coord_list = []
        outputs_class = []

        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_cls_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, self.class_embed, hs)):

            # first 2 layers
            if dec_lid < self.num_box_decoder_layers:
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                layer_cls = layer_cls_embed(layer_hs, text_dict)
                outputs_coord_list.append(layer_outputs_unsig.to(torch.float32))  # [bs, dn + num_queries, 4]
                outputs_class.append(layer_cls.to(torch.float32))                 # [bs, dn + num_queries, k]

            # last 4 layers
            else:
                # dn first
                layer_hs_bbox_dn = layer_hs[:, :effective_dn_number, :]  # [bs, num_dn, c]
                layer_hs_bbox_norm = layer_hs[:, effective_dn_number:, :][:, 0::(self.num_body_points + 1), :]  # [bs, num_group, c]
                bs = layer_ref_sig.shape[0]
                reference_before_sigmoid_bbox_dn = layer_ref_sig[:, :effective_dn_number, :]
                reference_before_sigmoid_bbox_norm = layer_ref_sig[:, effective_dn_number:, :][:,
                                                     0::(self.num_body_points + 1), :] #box
                layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                layer_outputs_unsig_dn = layer_delta_unsig_dn + inverse_sigmoid(reference_before_sigmoid_bbox_dn)
                layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                layer_outputs_unsig_norm = layer_delta_unsig_norm + inverse_sigmoid(reference_before_sigmoid_bbox_norm)
                layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                layer_outputs_unsig = torch.cat((layer_outputs_unsig_dn, layer_outputs_unsig_norm), dim=1)  
                layer_cls_dn = layer_cls_embed(layer_hs_bbox_dn, text_dict)
                layer_cls_norm = layer_cls_embed(layer_hs_bbox_norm, text_dict)
                layer_cls = torch.cat((layer_cls_dn, layer_cls_norm), dim=1)   # concat dn and num_queries
                outputs_class.append(layer_cls.to(torch.float32))                 # [bs, dn + num_group, k]
                outputs_coord_list.append(layer_outputs_unsig.to(torch.float32))  # [bs, dn + num_group, 4]

        # update keypoints
        outputs_keypoints_list = []
        outputs_keypoints_hw = []
        kpt_index = [x for x in range(num_group * (self.num_body_points + 1)) if x % (self.num_body_points + 1) != 0]  # box and kpt, [bs, num_group * (1 + num_kpts)]
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_res = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points * 3)) #没有输出，所以是0
                outputs_keypoints_list.append(layer_res.to(torch.float32))    # [bs, num_queries, num_kp*3]
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_kpt = layer_hs[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index, device=layer_hs.device))  # [bs, num_group * num_kpts, c]
                delta_xy_unsig = self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_kpt)         # [bs, num_group * num_kpts, 2]
                layer_ref_sig_kpt = layer_ref_sig[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index, device=layer_hs.device))
                layer_outputs_unsig_keypoints = delta_xy_unsig + inverse_sigmoid(layer_ref_sig_kpt[..., :2])  # [bs, num_group * num_kpts, 2]
                vis_xy_unsig = torch.ones_like(layer_outputs_unsig_keypoints,
                                               device=layer_outputs_unsig_keypoints.device)                   # [bs, num_group * num_kpt, 1], 1, keypoint score
                xyv = torch.cat((layer_outputs_unsig_keypoints, vis_xy_unsig[:, :, 0].unsqueeze(-1)), dim=-1) # [bs, num_group * num_kpt, 3]
                xyv = xyv.sigmoid()  # last dim is score
                layer_res = xyv.reshape((bs, num_group, self.num_body_points, 3)).flatten(2, 3)               # [bs, num_group, num_kpt*3]
                layer_hw = layer_ref_sig_kpt[..., 2:].reshape(bs, num_group, self.num_body_points, 2).flatten(2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)                                              # [bs, num_group, num_kpts*3], xyzxyz -> xyxyzz
                outputs_keypoints_list.append(layer_res.to(torch.float32))
                outputs_keypoints_hw.append(layer_hw)

        # for dn training
        if self.training and self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list, outputs_keypoints_list, dn_meta = self.dn_post_process(
                outputs_class, outputs_coord_list, outputs_keypoints_list, dn_meta
            )
            # outputs_class, outputs_coord_list = \
            #     post_process(outputs_class, outputs_coord_list,
            #                     dn_meta, self.aux_loss, self._set_aux_loss)
            

        outputs = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1],
               'pred_keypoints': outputs_keypoints_list[-1], 'text_query_masks': text_query['obj_query_masks']}


        if self.training:
            # convert pred_keypoints to its original kpt classes order
            # pred_keypoints: [bs, 50, num_kpt*3], num_kpt=68, xyxyvv
            new_pred_keypoints = []
            pred_keypoints = outputs['pred_keypoints']
            Z = pred_keypoints[..., :self.num_body_points*2]
            X = Z[..., 0::2]  # [bs, 50, 68]
            Y = Z[..., 1::2]  # [bs, 50, 68]
            V = pred_keypoints[..., self.num_body_points*2:]  # [bs, 50, 68]
            for batch_idx, (img_meta, X_per_batch, Y_per_batch, V_per_batch) in enumerate(zip(
                img_metas, X, Y, V
            )):
                kpt_id2index = img_meta['kpt_id2index']  # dict, original pose class id to shuffled pose class id
                kpt_ori_ids = torch.as_tensor(list(kpt_id2index.keys()), dtype=torch.long, device=pred_keypoints.device)
                kpt_shuffle_ids = torch.as_tensor(list(kpt_id2index.values()), dtype=torch.long, device=pred_keypoints.device)
                new_X_per_batch = torch.zeros_like(X_per_batch)
                new_Y_per_batch = torch.zeros_like(Y_per_batch)
                new_V_per_batch = torch.zeros_like(V_per_batch)
                new_X_per_batch[..., kpt_ori_ids] = X_per_batch[..., kpt_shuffle_ids]
                new_Y_per_batch[..., kpt_ori_ids] = Y_per_batch[..., kpt_shuffle_ids]
                new_V_per_batch[..., kpt_ori_ids] = V_per_batch[..., kpt_shuffle_ids]
                new_pred_keypoints_per_batch = torch.stack([new_X_per_batch, new_Y_per_batch, new_V_per_batch], dim=-1).flatten(1, 2)  # [50, 68, 3] -> [50, 68*3]
                new_pred_keypoints_per_batch = keypoint_xyzxyz_to_xyxyzz(new_pred_keypoints_per_batch)  # [50, 68*3], xyxyvv
                new_pred_keypoints.append(new_pred_keypoints_per_batch)
            outputs['pred_keypoints'] = torch.stack(new_pred_keypoints, dim=0)


            # add aux outputs 
            if self.aux_loss:
                aux_outputs = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_keypoints_list)  # list[dict]
                # convert pred_keypoints to its original kpt classes order
                new_aux_outputs = []
                for aux_out in aux_outputs:
                    # aux_out: dict
                    new_pred_keypoints = []
                    pred_keypoints = aux_out['pred_keypoints']
                    Z = pred_keypoints[..., :self.num_body_points*2]
                    X = Z[..., 0::2]  # [bs, 50, 68]
                    Y = Z[..., 1::2]  # [bs, 50, 68]
                    V = pred_keypoints[..., self.num_body_points*2:]  # [bs, 50, 68]
                    for batch_idx, (img_meta, X_per_batch, Y_per_batch, V_per_batch) in enumerate(zip(
                        img_metas, X, Y, V
                    )):
                        kpt_id2index = img_meta['kpt_id2index']  # dict, original pose class id to shuffled pose class id
                        kpt_ori_ids = torch.as_tensor(list(kpt_id2index.keys()), dtype=torch.long, device=pred_keypoints.device)
                        kpt_shuffle_ids = torch.as_tensor(list(kpt_id2index.values()), dtype=torch.long, device=pred_keypoints.device)
                        new_X_per_batch = torch.zeros_like(X_per_batch)
                        new_Y_per_batch = torch.zeros_like(Y_per_batch)
                        new_V_per_batch = torch.zeros_like(V_per_batch)
                        new_X_per_batch[..., kpt_ori_ids] = X_per_batch[..., kpt_shuffle_ids]
                        new_Y_per_batch[..., kpt_ori_ids] = Y_per_batch[..., kpt_shuffle_ids]
                        new_V_per_batch[..., kpt_ori_ids] = V_per_batch[..., kpt_shuffle_ids]
                        new_pred_keypoints_per_batch = torch.stack([new_X_per_batch, new_Y_per_batch, new_V_per_batch], dim=-1).flatten(1, 2)  # [50, 68, 3] -> [50, 68*3]
                        new_pred_keypoints_per_batch = keypoint_xyzxyz_to_xyxyzz(new_pred_keypoints_per_batch)  # [50, 68*3], xyxyvv
                        new_pred_keypoints.append(new_pred_keypoints_per_batch)
                    aux_out['pred_keypoints'] = torch.stack(new_pred_keypoints, dim=0)
                    new_aux_outputs.append(aux_out)
                aux_outputs = new_aux_outputs
                del new_aux_outputs

                outputs['aux_outputs'] = aux_outputs
                for out in outputs['aux_outputs']:
                    out['text_query_masks'] = text_query['obj_query_masks']
                
            # add enc outputs
            if hs_enc is not None:
                interm_coord = ref_enc[-1]
                interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
                interm_pose = torch.zeros_like(outputs_keypoints_list[0])
                outputs['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord, 'pred_keypoints': interm_pose}
                outputs['interm_outputs']['text_query_masks'] = text_query['obj_query_masks']
            # add text_query_masks to dn_meta
            if dn_meta is not None:
                dn_meta["output_known_lbs_bboxes"]["text_query_masks"] = text_query['obj_query_masks']
                for x in dn_meta["output_known_lbs_bboxes"]["aux_outputs"]:
                    x["text_query_masks"] = text_query['obj_query_masks']

            loss_dict = self.criterion(outputs, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        else:
            loss_dict = None
            loss = None

        return UniPoseOutput(
            loss = loss, 
            loss_dict = loss_dict,
            pred_logits = outputs['pred_logits'],
            pred_boxes = outputs['pred_boxes'],
            pred_keypoints = outputs['pred_keypoints'],
            # dn_meta = out['dn_meta']
        )

    # dn training
    def prepare_for_cdn(self, targets, dn_number, label_noise_ratio, box_noise_scale, num_queries, hidden_dim, dn_labelbook_size, label_enc,
                    kpt_mask, num_body_points, num_heads):
        """
        Args:
            targets (list[dict]): training labels
            dn_number (int): dn query number
            label_noise_ratio: 0.5
            box_noise_scale: 1.0
            num_queries: 900
            hidden_dim: 256
            dn_labelbook_size: rand classes label size
            label_enc (tensor): [bs, num_patches, c]
            kpt_mask: [bs, 69]
            num_body_points: 68
            num_heads: 8
        """
        max_num_patches = label_enc.shape[1]
        dn_labelbook_size = min(dn_labelbook_size, max_num_patches)  # in case the labelbook size is too large

        if dn_number <= 0:
            return None, None, None, None, None
        # e.g. class_labels: [[0,1,2], [1,4]]

        # positivie and negative dn queries
        dn_number = dn_number * 2 
        known = [(torch.ones_like(t["class_labels"])).cuda() for t in targets]  # e.g. [(1, 1, 1), (1, 1)]
        batch_size = len(known)
        known_num = [sum(k) for k in known]  # list[int], e.g. [3, 2]
        if int(max(known_num)) == 0:
            return None, None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2)) # num of dn-group
        # dn_number is shared in a batch, here dn_number is num_groups

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)            # [num_all_gt,], value is 1
        labels = torch.cat([t["class_labels"] for t in targets]) # [num_all_gt,], e.g. [0, 1, 2, 1, 4]
        boxes = torch.cat([t["boxes"] for t in targets])         # [num_all_gt, 4]
        batch_idx = torch.cat(
            [torch.full_like(t["class_labels"].long(), i) for i, t in enumerate(targets)]
        )  # e.g. [0, 0, 0, 1, 1]

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)  # e.g. [0, 1, 2, 3, 4]

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)  # [num_all_gt x 2 x dn_number,], e.g. [0, 1, 2, 3, 4, 0, 1, 2, 3, 4...]
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)        # [num_all_gt x 2 x dn_number,]
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)        # [num_all_gt x 2 x dn_number,]
        known_bboxs = boxes.repeat(2 * dn_number, 1)                   # [num_all_gt x 2 x dn_number, 4]
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # label jittering
        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of box prob
            new_label = torch.randint_like(
                chosen_indice, 0, dn_labelbook_size
            )  # [num_rand,], randomly put a new class
            known_labels_expaned.scatter_(0, chosen_indice, new_label)  # [num_all_gt x 2 x dn_number,]
        single_padding = int(max(known_num))  # value of max_num_gt in a batch

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )  # [num_all_gt x dn_number, 4]
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)  
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes) # [pos, neg, pos, neg,...]
        # for [num_all_gt x 2 x dn_number,], pos, neg, pos, neg...

        # box jittering
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)  # [num_all_gt x 2 x dn_number, 4]
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2 # (x1, y1)
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2 # (x2, y2)

            diff = torch.zeros_like(known_bboxs) # [num_all_gt x 2 x dn_number, 4]
            diff[:, :2] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)
            diff[:, 2:] = known_bboxs[:, 2:] / 2 # (0.5w, 0.5h)

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            ) # -1 or 1
            rand_part = torch.rand_like(known_bboxs) # [0, 1]
            rand_part[negative_idx] += 1.0 # negative [1, 2], negatvie boxes noise scale is larger
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # transform back to the cxcywh format
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        # label_embed is from the corresponding location index of text_query 
        input_label_embed = label_enc[known_bid.long(), known_labels_expaned.long()] # [n, c]
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)  # [n, 4], n = num_gt_all x 2 x dn_number

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()  # [max_num_gt x 2 x dn_number, C]
        padding_bbox = torch.zeros(pad_size, 4).cuda()            # [max_num_gt x 2 x dn_number, C]

        input_query_label = padding_label.repeat(batch_size, 1, 1).to(label_enc.dtype)  # [bs, N, C], N = max_num_gt x 2 x dn_number
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1).to(label_enc.dtype)    # [bs, N, 4]
        
        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # e.g. [0, 1, 2, 0, 1], [num_all_gt,]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()  # [num_all_gt x 2 x dn_number,]
        if len(known_bid):
            # (batch idx, known_idx)
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed  # this the box coordinate cxcywh, not need sigmoid

        tgt_size = pad_size + num_queries  # dn first
        """
        For a binary mask, a ``True`` value indicates that the corresponding position is not allowed to attend. 
        For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend. 
        For a float mask, the mask values will be added to the attention weight.
        
        Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.
        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.
        """
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0  # False
        # match query can not see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "pad_size": pad_size,
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        # -------------------------------------------
        # this is for attn_mask2, used in the last 5 decoder layers (box + pose)
        num_group = 50  # pose num_queries
        tgt_size2 = pad_size + num_group * (num_body_points + 1)
        no_dn_part_size = num_group * (num_body_points + 1)
        attn_mask2 = torch.ones(batch_size, num_heads, tgt_size2, tgt_size2).to("cuda") < 0  # false
        attn_mask2_no_dn_part = torch.ones(batch_size, num_heads, no_dn_part_size, no_dn_part_size).to('cuda') < 0  # false

        group_bbox_kpt = (num_body_points + 1)  # 69

        for matchj in range(num_group * group_bbox_kpt):
            sj = (matchj // group_bbox_kpt) * group_bbox_kpt
            ej = (matchj // group_bbox_kpt + 1) * group_bbox_kpt
            if sj > 0:
                attn_mask2_no_dn_part[:,:,matchj, :sj] = True
            if ej < num_group * group_bbox_kpt:
                attn_mask2_no_dn_part[:,:,matchj, ej:] = True

        bs, length = kpt_mask.shape  # [bs, 69]
        equal_mask = kpt_mask[:, :, None] == kpt_mask[:, None, :]
        equal_mask= equal_mask.unsqueeze(1).repeat(1,num_heads,1,1)
        for idx in range(num_group):
            start_idx = idx * length
            end_idx = (idx + 1) * length
            attn_mask2_no_dn_part[:, :,start_idx:end_idx, start_idx:end_idx][equal_mask] = False
            attn_mask2_no_dn_part[:, :,start_idx:end_idx, start_idx:end_idx][~equal_mask] = True

        # dn_part:
        attn_mask2[:,:,:pad_size, :pad_size] = \
            attn_mask[:pad_size, :pad_size].reshape(1,1,pad_size,pad_size).repeat(attn_mask2.shape[0],num_heads,1,1)  
        # no_dn_part:
        attn_mask2[:,:, pad_size:, pad_size:] = attn_mask2_no_dn_part
        # match query cannot see the reconstruct
        attn_mask2[:,:, pad_size:, :pad_size] = True
        attn_mask2 = attn_mask2.flatten(0, 1)

        return input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, outputs_keypoint, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            # dn part
            output_known_class = [outputs_class_i[:, :padding_size, :] for outputs_class_i in outputs_class]
            output_known_coord = [outputs_coord_i[:, :padding_size, :] for outputs_coord_i in outputs_coord]
            # matching part
            outputs_class = [outputs_class_i[:, padding_size:, :] for outputs_class_i in outputs_class]
            outputs_coord = [outputs_coord_i[:, padding_size:, :] for outputs_coord_i in outputs_coord]
            outputs_keypoint = outputs_keypoint
            # dn out
            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            # dn aux, only cls + box
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord, outputs_keypoints=None)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_keypoint, dn_metas

    def prepare_for_mask(self, kpt_mask):
        # this is for inference, since num_body_points=68, larger than coco 17 keypoints
        tgt_size2 = 50 * (1 + self.num_body_points)
        attn_mask2 = torch.ones(kpt_mask.shape[0], self.nheads, tgt_size2, tgt_size2).to('cuda') < 0
        group_bbox_kpt = self.num_body_points
        num_group = 50
        for matchj in range(num_group * group_bbox_kpt):
            sj = (matchj // group_bbox_kpt) * group_bbox_kpt
            ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
            if sj > 0:
                attn_mask2[:,:,matchj, :sj] = True
            if ej < num_group * group_bbox_kpt:
                attn_mask2[:,:,matchj, ej:] = True


        bs, length = kpt_mask.shape
        equal_mask = kpt_mask[:, :, None] == kpt_mask[:, None, :]
        equal_mask= equal_mask.unsqueeze(1).repeat(1,self.nheads,1,1)
        for idx in range(num_group):
            start_idx = idx * length
            end_idx = (idx + 1) * length
            attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][equal_mask] = False
            attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][~equal_mask] = True

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

        return input_query_label, input_query_bbox, attn_mask, attn_mask2.flatten(0,1), dn_meta

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_keypoints=None):
        if outputs_keypoints is not None:
            return [{'pred_logits': a, 'pred_boxes': b,'pred_keypoints': c}
                    for a, b,c in zip(outputs_class[:-1], outputs_coord[:-1],outputs_keypoints[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def generate_masks_with_text_query_masks(text_query_masks):
    """
    Args:
        text_query_masks (bool): [bs, max_num_patches,], valid is 1
    Returns:
        self_attention_mask (bool): [bs, max_num_patches, max_num_patches], padding is 1
        position_ids (long): [bs, max_num_patches]
    """
    batch_size, num_token = text_query_masks.shape
    self_attention_mask = torch.eye(num_token, device=text_query_masks.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), dtype=torch.long, device=text_query_masks.device)

    for batch_idx in range(batch_size):
        num_valid = text_query_masks[batch_idx].sum()  # [num_patches_i,]
        self_attention_mask[batch_idx, :num_valid, :num_valid] = True
        position_ids[batch_idx, :num_valid] = torch.arange(0, num_valid, device=text_query_masks.device)

    return self_attention_mask, position_ids
  
class ContrastiveAssign(nn.Module):
    def __init__(self, project=False, cal_bias=None, max_text_len=256):
        """
        :param x: query
        :param y: text embed
        :param proj:
        :return:
        """
        super().__init__()
        self.project = project
        self.cal_bias = cal_bias
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict['encoded_text'] #y: bs, 350, 256 x: bs, nq, 256

        max_text_len = y.shape[1]

        text_token_mask = text_dict['text_token_mask']

        if self.cal_bias is not None:
            raise NotImplementedError
            return x @ y.transpose(-1, -2) + self.cal_bias.weight.repeat(x.shape[0], x.shape[1], 1)
        res = x @ y.transpose(-1, -2) #bs, nq, 256 @ bs, 256, 350 -> bs, nq, 350
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf')) #bs, 350 -> bs , 1, 350

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], max_text_len), float('-inf'), device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res

# PosEmbed
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # if os.environ.get("SHILONG_AMP", None) == '1':
            #     eps = 1e-4
            # else:
            #     eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingSineHW(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

# Backbone

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

# SwinTransformer

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)).to(dtype=x.dtype)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        dilation (bool): if True, the output size if 16x downsample, ow 32x downsample.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 dilation=False,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation

        # if use_checkpoint:
        #     print("use_checkpoint!!!!!!!!!!!!!!!!!!!!!!!!")

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            num_features[-1] = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                # dim=int(embed_dim * 2 ** i_layer),
                dim=num_features[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                downsample=downsamplelist[i_layer],
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False



    def forward_raw(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # outs:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]
        return tuple(outs)


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors

        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # out:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]

        # collect for nesttensors        
        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

#fuse_model
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000) # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights_l.masked_fill_(attention_mask_v, float('-inf')).to(v.dtype)

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights.masked_fill_(attention_mask_l, float('-inf')).to(v.dtype)
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)


        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

#transformer_vanilla
class TextTransformer(nn.Module):
    def __init__(self, num_layers, d_model=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.norm = None

        single_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = _get_clones(single_encoder_layer, num_layers)


    def forward(self, memory_text:torch.Tensor, text_attention_mask:torch.Tensor):
        """        

        Args:
            text_attention_mask: bs, num_token
            memory_text: bs, num_token, d_model

        Raises:
            RuntimeError: _description_

        Returns:
            output: bs, num_token, d_model
        """

        output = memory_text.transpose(0, 1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=text_attention_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)
        q = k = self.with_pos_embed(src, pos)
        q = q.to(src.dtype)
        k = k.to(src.dtype)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gam_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gam_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gam_v * delta_v)
        l = l + self.drop_path(self.gam_l * delta_l)
        return v, l

#deformable_transformer 
class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='standard',
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 use_text_enhancer=False,
                 use_fusion_layer=False,
                 use_checkpoint=False,
                 use_transformer_ckpt=False,
                 use_text_cross_attention=False,
                 text_dropout=0.1,
                 fusion_dropout=0.1,
                 fusion_droppath=0.0,

                 binary_query_selection=False,
                 ffn_extra_layernorm=False,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.ffn_extra_layernorm = ffn_extra_layernorm
        assert query_dim == 4

        self.binary_query_selection = binary_query_selection
        if self.binary_query_selection:
            self.binary_query_selection_layer = nn.Linear(d_model, 1)
        # assert not binary_query_selection, 'binary_query_selection not implemented yet'

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout
            )
        else:
            text_enhance_layer = None

        #import pdb
        #pdb.set_trace()
        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, d_model=d_model,
            num_queries=num_queries,
            enc_layer_share=enc_layer_share,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_text_cross_attention=use_text_cross_attention,
                                                              ffn_extra_layernorm=ffn_extra_layernorm, )

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          modulate_hw_attn=modulate_hw_attn,
                                          num_feature_levels=num_feature_levels,
                                          deformable_decoder=deformable_decoder,
                                          decoder_query_perturber=decoder_query_perturber,
                                          dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                          dec_layer_share=dec_layer_share,
                                          use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                          )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type == 'standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            #self.enc_output.requires_grad = False
            #self.enc_output_norm.requires_grad = False

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                           0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[
                           0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            # assert len(rm_self_attn_layers) == num_decoder_layers
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, attn_mask2=None, text_dict=None,
                dn_meta=None,targets=None,kpt_embed=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer

        """
        # if self.two_stage_type != 'no' and self.two_stage_add_query_num == 0:
        #     assert refpoint_embed is None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict['encoded_text'],
            text_attention_mask=~text_dict['text_token_mask'],
            # we ~ the mask . False means use the token; True means pad the token
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['text_self_attention_masks'],
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################
        text_dict['encoded_text'] = memory_text

        if self.two_stage_type == 'standard':
            #print('???')
            #import pdb
            #pdb.set_trace()
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes,
                                                                           input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            if self.binary_query_selection:
                topk_logits = self.binary_query_selection_layer(output_memory).squeeze(-1)
            else:
                if text_dict is not None:
                    enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
                else:
                    enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

                topk_logits = enc_outputs_class_unselected.max(-1)[0]
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries

            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1,
                                             topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries,
                                                                             1)  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if refpoint_embed.isnan().any() | refpoint_embed.isinf().any():
        #         import ipdb; ipdb.set_trace()
        #     if tgt.isnan().any() | tgt.isinf().any():
        #         import ipdb; ipdb.set_trace()

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios, tgt_mask=attn_mask,
            tgt_mask2=attn_mask2,
            memory_text=text_dict['encoded_text'],
            text_attention_mask=~text_dict['text_token_mask'],
            text_dict=text_dict,
            dn_meta=dn_meta,
            targets=targets,
            kpt_embed=kpt_embed
            # we ~ the mask . False means use the token; True means pad the token
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None

class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, d_model=256,
                 num_queries=300,
                 enc_layer_share=False,
                 text_enhance_layer=None,
                 feature_fusion_layer=None,
                 use_checkpoint=False,
                 use_transformer_ckpt=False,
                 ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        #import pdb 
        #pdb.set_trace()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(text_enhance_layer, num_layers, layer_share=enc_layer_share)
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(feature_fusion_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_,  device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_,  device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                # for images
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                # for texts
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - memory_text: bs, n_text, 256
            - text_attention_mask: bs, n_text
                False for no padding; True for padding
            - pos_text: bs, n_text, 256

            - position_ids: bs, n_text
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """

        #dtype = next(self.parameters()).dtype
        output = src

        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = torch.arange(n_text, device=memory_text.device).float().unsqueeze(0).unsqueeze(-1).repeat(bs,
                                                                                                                     1,
                                                                                                                     1)
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(position_ids[..., None], num_pos_feats=256, exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                if self.use_checkpoint:
                    output, memory_text = checkpoint.checkpoint(
                        self.fusion_layers[layer_id],
                        output,
                        memory_text,
                        key_padding_mask,
                        text_attention_mask
                    )
                else:
                    output, memory_text = self.fusion_layers[layer_id](v=output, l=memory_text,
                                                                       attention_mask_v=key_padding_mask,
                                                                       attention_mask_l=text_attention_mask)

            if self.text_layers:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None)
                ).transpose(0, 1)

            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask
                )
            else:
                output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes,
                               level_start_index=level_start_index, key_padding_mask=key_padding_mask)

        return output, memory_text

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False,
                 num_box_decoder_layers=2,
                 num_body_points=68,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.pose_embed = None
        self.pose_hw_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None
        self.num_body_points = num_body_points

        self.hw = nn.Embedding(17, 2)
        self.num_box_decoder_layers = num_box_decoder_layers
        self.kpt_index = [x for x in range(50 * (self.num_body_points + 1)) if x % (self.num_body_points + 1) != 0]
        self.hw_append = nn.Embedding(self.num_body_points-17, 2)
        #self.hw_append.requires_grad = False

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                tgt_mask2: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                # for text
                memory_text: Optional[Tensor] = None,
                text_attention_mask: Optional[Tensor] = None,
                text_dict: Optional[Tensor] = None,
                dn_meta: Optional[Tensor] = None,
                targets: Optional[Tensor] = None,
                kpt_embed: Optional[Tensor] = None
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
            - kpt_embed
        """
        #一开始的tgt: 900 + num_dn

        output = tgt
        output += self.hw.weight[0, 0] * 0.0


        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        effect_num_dn = dn_meta['pad_size'] if (dn_meta is not None and self.training) else 0
        inter_select_number = 50
        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]).to(tgt.dtype)  # nq, bs, 256*2

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if query_pos.isnan().any() | query_pos.isinf().any():
            #         import ipdb; ipdb.set_trace()

            # NOTE: main process
            #第一次：output.shape[0] = 1100 = 200(pad_size) + 900 
            #>=第三次： output.shape[0] = 3650 = 200 + 50 * (68 + 1)
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,

                memory_text=memory_text,
                text_attention_mask=text_attention_mask,

                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,

                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)


            intermediate.append(self.norm(output))
            # iter update, layer 0
            if layer_id < self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

            # select # ref points as anchors, layer 1
            if layer_id == self.num_box_decoder_layers - 1:
                dn_output = output[:effect_num_dn]                              
                dn_new_reference_points = new_reference_points[:effect_num_dn]  # [num_dn, bs, 4]
                class_unselected = self.class_embed[layer_id](output.transpose(0, 1), text_dict)[:,
                                   effect_num_dn:].transpose(0, 1)              # [num_queries, bs, k ]
                topk_proposals = torch.topk(class_unselected.max(-1)[0], inter_select_number, dim=0)[1] # [topk, bs], topk = num_group = 50, for pose
                new_reference_points_for_box = torch.gather(new_reference_points[effect_num_dn:], 0, 
                                                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  #900, bs, 4 -> 50, bs, 4
                new_output_for_box = torch.gather(output[effect_num_dn:], 0, 
                                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)) #900, bs, 256 -> 50, bs, 256
                keypoint_embed=kpt_embed.transpose(0, 1) # num_kpts, bs, d

                # nkptclass=68
                new_output_for_keypoint = keypoint_embed[None, :, :, :].repeat(new_output_for_box.shape[0],1,1,1) #50, nkptclass, bs, d
                delta_xy = self.pose_embed[-1](new_output_for_keypoint)[..., :2] #pose_embed: MLPs -> 50, nkptclasses, bs, 2
                keypoint_xy = (inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy).sigmoid() #50, nkptclass, bs, 2
                num_queries, _, bs, _ = keypoint_xy.shape
                aa = torch.cat((self.hw.weight,self.hw_append.weight),dim=0)  # [self.num_keypoint, 2]
                keypoint_wh_weight = aa.unsqueeze(0).unsqueeze(-2).repeat(num_queries, 1, bs, 1).sigmoid()  # [50, 68, bs, 2]
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None] #50, nkptclasses, bs, 2
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1) #50, nkptclass, bs, 4, keypoint reference points dim should=4
                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1).flatten(0, 1) #50, 1+nkptclass, bs, 4 
                output = torch.cat((new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1).flatten(0, 1) # [50, 1+num_kpt, bs, c]
                #output:  两部分
                # 1. 50, bs, 256: 50个query的输出
                # 2. 50, nkptclass, bs, 256: 50个kpt query的输出
                # 拼接起来: 50, (1+nkptclass), bs, 256

                # concat dn
                new_reference_points = torch.cat((dn_new_reference_points, new_reference_points), dim=0) 
                #new_reference_points: 两部分
                # 1. dn_num, bs, 4: dn的输出
                # 2. 50, (1+nkptclass), bs, 4: 50个query的输出


                output = torch.cat((dn_output, output), dim=0)
                tgt_mask = tgt_mask2  # 

            # last 4 layers
            if layer_id >= self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_dn = output[:effect_num_dn]
                output_bbox_norm = output[effect_num_dn:][0::(self.num_body_points + 1)] #所有的box output
                reference_before_sigmoid_bbox_dn = reference_before_sigmoid[:effect_num_dn]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[effect_num_dn:][
                                                     0::(self.num_body_points + 1)]
                delta_unsig_dn = self.bbox_embed[layer_id](output_bbox_dn)
                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_dn = delta_unsig_dn + reference_before_sigmoid_bbox_dn
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_dn = outputs_unsig_dn.sigmoid()
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid()
                output_kpt = output[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index, device=output.device))
                delta_xy_unsig = self.pose_embed[layer_id - self.num_box_decoder_layers](output_kpt) #pose_embed: MLPs
                outputs_unsig = reference_before_sigmoid[effect_num_dn:].index_select(0, torch.tensor(self.kpt_index,
                                                                                                      device=output.device)).clone()  ##
                delta_hw_unsig = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid()
                bs = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat((new_reference_points_for_box_norm.unsqueeze(1),
                                                       new_reference_points_for_keypoint.view(-1, self.num_body_points,
                                                                                              bs, 4)), dim=1).flatten(0,
                                                                                                                      1)
                new_reference_points = torch.cat((new_reference_points_for_box_dn, new_reference_points_norm), dim=0)

            if self.rm_detach and 'dec' in self.rm_detach:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()

            # if layer_id != self.num_layers - 1:
            if self.use_detached_boxes_dec_out:
                ref_points.append(reference_points)
            else:
                ref_points.append(new_reference_points)

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, 
                              spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_text_feat_guide=False,
                 use_text_cross_attention=False,
                 ffn_extra_layernorm=False
                 ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)
        if ffn_extra_layernorm:
            raise NotImplementedError('ffn_extra_layernorm not implemented')
            self.norm_ext = nn.LayerNorm(d_ffn)
        else:
            self.norm_ext = None

        self.key_aware_proj = None
        self.use_text_feat_guide = use_text_feat_guide
        assert not use_text_feat_guide
        self.use_text_cross_attention = use_text_cross_attention

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt, ipdb_flag=False):

        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                memory_text: Optional[Tensor] = None,  # bs, num_token, d_model
                text_attention_mask: Optional[Tensor] = None,  # bs, num_token

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if tgt.isnan().any() | tgt.isinf().any() :
            #         import ipdb; ipdb.set_trace()

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(self.with_pos_embed(tgt, tgt_query_pos), memory_text.transpose(0, 1),
                                memory_text.transpose(0, 1), key_padding_mask=text_attention_mask)[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

            # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()

            # if tgt.isnan().any() | tgt.isinf().any() :
            #     import ipdb; ipdb.set_trace()

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), 
                               memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     tgtk = tgt.clone()
        #     if tgt.isnan().any() | tgt.isinf().any() :
        #         import ipdb; ipdb.set_trace()

        # ffn
        tgt = self.forward_ffn(tgt)
        # if os.environ.get("SHILONG_AMP_INFNAN_DEBUG") == '1':
        #     if tgt.isnan().any() | tgt.isinf().any() :
        #         tgtk = self.forward_ffn(tgtk, ipdb_flag=True)
        #         import ipdb; ipdb.set_trace()

        return tgt

#loss
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25, 
                 cost_keypoints: float = 1, cost_oks: float = 1, num_body_points: int = 68):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.cost_keypoints = cost_keypoints 
        self.cost_oks = cost_oks
        self.num_body_points = num_body_points
        if num_body_points == 68:
            self.sigmas = np.array([
                .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                1.07, .87, .87, .89, .89, .25, .25, .25, .25, .25, .25, .25, .25,
                .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
                .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
                .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25, .25,
            ], dtype=np.float32) / 10.0
        else:
            raise ValueError(f'Unsupported keypoints number {num_body_points}')


        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)             # [batch_size * num_queries, 4]
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)    # [batch_size * num_queries, num_kpts*3]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["class_labels"] for v in targets])                        # [num_all_gt,]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).to(torch.float32)            # [num_all_gt, 4]
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets]).to(torch.float32)   # [num_all_gt, num_kpts*3]
        tgt_area = torch.cat([v["area"] for v in targets]).to(torch.float32)             # [num_all_gt]

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # FIXME: whether need keypoints cost
        Z_pred = out_keypoints[:, 0:(self.num_body_points*2)]
        V_pred = out_keypoints[:, (self.num_body_points*2):]
        Z_gt = tgt_keypoints[:, 0:(self.num_body_points*2)]
        V_gt: torch.Tensor = tgt_keypoints[:, (self.num_body_points*2):]
        if Z_pred.sum() > 0:
            sigmas = Z_pred.new_tensor(self.sigmas)  # [num_kpts]
            variances = (sigmas * 2) ** 2
            kpt_preds = Z_pred.reshape(-1, Z_pred.size(-1) // 2, 2)  # [bs*nq, num_kpts, 2]
            kpt_gts = Z_gt.reshape(-1, Z_gt.size(-1) // 2, 2)        # [all_num_gt, 2]
            squared_distance = (kpt_preds[:, None, :, 0] - kpt_gts[None, :, :, 0]) ** 2 + \
                               (kpt_preds[:, None, :, 1] - kpt_gts[None, :, :, 1]) ** 2
            squared_distance0 = squared_distance / (tgt_area[:, None] * variances[None, :] * 2)
            squared_distance1 = torch.exp(-squared_distance0)
            squared_distance1 = squared_distance1 * V_gt
            oks = squared_distance1.sum(dim=-1) / (V_gt.sum(dim=-1) + 1e-6)
            oks = oks.clamp(min=1e-6)
            cost_oks = 1 - oks
            cost_keypoints = torch.abs(Z_pred[:, None, :] - Z_gt[None])
            cost_keypoints = cost_keypoints * V_gt.repeat_interleave(2, dim=1)[None]
            cost_keypoints = cost_keypoints.sum(-1)
        else:
            cost_oks = torch.zeros_like(cost_bbox)
            cost_keypoints = torch.zeros_like(cost_bbox)

        # Final cost matrix
        # FIXME: whether need keypoints cost
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_keypoints * cost_keypoints + self.cost_oks * cost_oks
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha, cost_keypoints=args.set_cost_keypoint, cost_oks=args.set_cost_oks, 
            num_body_points=args.num_body_points
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))    

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, focal_alpha, losses, num_body_points, num_box_decoder_layers):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        # self.num_classes = num_classes  # not used
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.oks = OKSLoss(linear=True,
                 num_keypoints=num_body_points,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0)
        
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "pred_logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["pred_logits"]  # [bs, num_queries, 256]

        # loss_labels according to vl
        idx = self._get_src_permutation_idx(indices)  # (batch_idx, src_idx)
        if len(idx[0]) == 0:
            loss_ce = source_logits.sum() * 0.0
            losses = {"loss_ce": loss_ce}
            return losses

        target_classes_onehot = torch.zeros(source_logits.size(),
                    dtype=source_logits.dtype,
                    layout=source_logits.layout,
                    device=source_logits.device
        )  # [bs, num_queries, 256]
        class_labels = [x["class_labels"] for x in targets] # [num_gt_1, num_gt_2,..]
        # loop over batch size
        for batch_idx, (src_idxs, tgt_idxs) in enumerate(indices):
            # loop over each gt object
            for (src_idx, tgt_idx) in zip(src_idxs, tgt_idxs):
                target_classes_onehot[batch_idx, src_idx, class_labels[batch_idx][tgt_idx]] = 1

        loss_ce = token_sigmoid_binary_focal_loss(
            source_logits, target_classes_onehot, text_mask=outputs["text_query_masks"]) / num_boxes
        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

        if len(src_keypoints) == 0:
            device = outputs["pred_logits"].device
            losses = {
                'loss_keypoints': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
                'loss_oks': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
            }
            return losses
        
        Z_pred = src_keypoints[:, 0:(self.num_body_points * 2)]   # [all_num_gt, num_kpts*2], xy
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(torch.float32)  # [all_num_gt, num_kpts*3]
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(torch.float32)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points * 2)] # [all_num_gt, ]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points * 2):]
        oks_loss=self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')     # l1 loss
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)  # invisible kpt loss=0
        losses = {}
        losses['loss_keypoints'] = pose_loss.sum() / num_boxes        
        losses['loss_oks'] = oks_loss.sum() / num_boxes
        return losses


    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(torch.float32)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'keypoints': self.loss_keypoints
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'interm_outputs'}
        device=next(iter(outputs.values())).device

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        indices = self.matcher(outputs_without_aux, targets)
        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss in ['keypoints'] and idx < self.num_box_decoder_layers:
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if loss in ['keypoints']:
                    continue
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

class DNSetCriterion(SetCriterion):
    def forward(self, outputs, targets, dn_metas=None):
        losses = super(DNSetCriterion, self).forward(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # compute all the requested losses
        aux_num = 0 
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])  # aux decoder layer number
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)
        return losses
    
    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            # loop over batchsize
            for i in range(len(targets)):
                if len(targets[i]["class_labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["class_labels"])).long().cuda()  # [num_gt_i,]
                    t = t.unsqueeze(0).repeat(dn_num, 1) # shape: (dn_num, n)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                if loss not in ["labels", "boxes", "masks"]:
                    continue
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num
                    )
                )
            
            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        # dn aux loss
        for i in range(aux_num):
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    if loss not in ["labels", "boxes", "masks"]:
                        continue
                    l_dict.update(
                        self.get_loss(
                            loss, output_known_lbs_bboxes_aux, targets, dn_idx, num_boxes * dn_num
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict[f"loss_bbox_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
                l_dict[f"loss_giou_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
                l_dict[f"loss_class_dn_{i}"] = torch.as_tensor(0.0).to("cuda")
            losses.update(l_dict)
        return losses

    
def token_sigmoid_binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, text_mask=None, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256
    assert (inputs.dim() == 3)
    assert (targets.dim() == 3)
    assert (text_mask is not None and text_mask.dim() == 2)
    bs, num_queries, max_text_len = inputs.size()
    bs, max_num_patches = text_mask.size()

    text_mask_pad = torch.zeros((bs, max_text_len), dtype=torch.bool, device=inputs.device)
    text_mask_pad[:, :max_num_patches] = text_mask
    text_mask = text_mask_pad

    text_mask = (text_mask > 0).unsqueeze(1)         # (bs, 1, max_seq_len)
    text_mask = text_mask.repeat(1, num_queries, 1)  # [bs, num_queries, max_seq_len]
    inputs = torch.masked_select(inputs, text_mask)
    targets = torch.masked_select(targets, text_mask)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction:
        return loss.sum()
    else:
        return loss

#dn mask generate
def prepare_for_cdn_v1(dn_args, training, num_queries, num_classes, hidden_dim, label_enc, kpt_mask, num_body_points):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    

    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }

        num_group = 50
        tgt_size2 = pad_size + num_group * (num_body_points + 1)
        no_dn_part_size = num_group * (num_body_points + 1)
        attn_mask2 = torch.ones(batch_size, 8, tgt_size2, tgt_size2).to('cuda') < 0
        attn_mask2_no_dn_part = torch.ones(batch_size, 8, no_dn_part_size, no_dn_part_size).to('cuda') < 0


        group_bbox_kpt = (num_body_points + 1)
        
        for matchj in range(num_group * group_bbox_kpt):
            sj = (matchj // group_bbox_kpt) * group_bbox_kpt
            ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
            if sj > 0:
                attn_mask2_no_dn_part[:,:,matchj, :sj] = True
            if ej < num_group * group_bbox_kpt:
                attn_mask2_no_dn_part[:,:,matchj, ej:] = True

        bs, length = kpt_mask.shape
        equal_mask = kpt_mask[:, :, None] == kpt_mask[:, None, :]
        equal_mask= equal_mask.unsqueeze(1).repeat(1,8,1,1)
        for idx in range(num_group):
            start_idx = idx * length
            end_idx = (idx + 1) * length
            attn_mask2_no_dn_part[:, :,start_idx:end_idx, start_idx:end_idx][equal_mask] = False
            attn_mask2_no_dn_part[:, :,start_idx:end_idx, start_idx:end_idx][~equal_mask] = True

        # dn_part:
        attn_mask2[:,:,:pad_size, :pad_size] = \
            attn_mask[:pad_size, :pad_size].reshape(1,1,pad_size,pad_size).repeat(attn_mask2.shape[0],8,1,1)  
        # no_dn_part:
        attn_mask2[:,:, pad_size:, pad_size:] = attn_mask2_no_dn_part
        # match query cannot see the reconstruct
        attn_mask2[:,:, pad_size:, :pad_size] = True

        #import pdb
        #pdb.set_trace()

    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        attn_mask2 = None
        dn_meta = None
    #print(attn_mask.shape)
    #print(attn_mask2.shape)
    if attn_mask2 is not None:
        attn_mask2 = attn_mask2.flatten(0,1)
        
    return input_query_label, input_query_bbox, attn_mask, attn_mask2, dn_meta

def prepare_for_mask(kpt_mask):


    tgt_size2 = 50 * 69
    attn_mask2 = torch.ones(kpt_mask.shape[0], 8, tgt_size2, tgt_size2).to('cuda') < 0
    group_bbox_kpt = 69
    num_group=50
    for matchj in range(num_group * group_bbox_kpt):
        sj = (matchj // group_bbox_kpt) * group_bbox_kpt
        ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
        if sj > 0:
            attn_mask2[:,:,matchj, :sj] = True
        if ej < num_group * group_bbox_kpt:
            attn_mask2[:,:,matchj, ej:] = True


    bs, length = kpt_mask.shape
    equal_mask = kpt_mask[:, :, None] == kpt_mask[:, None, :]
    equal_mask= equal_mask.unsqueeze(1).repeat(1,8,1,1)
    for idx in range(num_group):
        start_idx = idx * length
        end_idx = (idx + 1) * length
        attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][equal_mask] = False
        attn_mask2[:, :,start_idx:end_idx, start_idx:end_idx][~equal_mask] = True




    input_query_label = None
    input_query_bbox = None
    attn_mask = None
    dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, attn_mask2.flatten(0,1), dn_meta

def post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):

    if dn_meta and dn_meta['pad_size'] > 0:

        output_known_class = [outputs_class_i[:, :dn_meta['pad_size'], :] for outputs_class_i in outputs_class]
        output_known_coord = [outputs_coord_i[:, :dn_meta['pad_size'], :] for outputs_coord_i in outputs_coord]

        outputs_class = [outputs_class_i[:, dn_meta['pad_size']:, :] for outputs_class_i in outputs_class]
        outputs_coord = [outputs_coord_i[:, dn_meta['pad_size']:, :] for outputs_coord_i in outputs_coord]

        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord



def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_swin_transformer(modelname, pretrain_img_size, **kw):
    assert modelname in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']

    model_para_dict = {
        'swin_T_224_1k': dict(
            embed_dim=96,
            depths=[ 2, 2, 6, 2 ],
            num_heads=[ 3, 6, 12, 24],
            window_size=7
        ),        
        'swin_B_224_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=7
        ),
        'swin_B_384_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=12
        ),
        'swin_L_224_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=7
        ),
        'swin_L_384_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=12
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = SwinTransformer(pretrain_img_size=pretrain_img_size, **kw_cgf)
    return model

def build_swin_transformer(modelname, pretrain_img_size, **kw):
    assert modelname in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']

    model_para_dict = {
        'swin_T_224_1k': dict(
            embed_dim=96,
            depths=[ 2, 2, 6, 2 ],
            num_heads=[ 3, 6, 12, 24],
            window_size=7
        ),        
        'swin_B_224_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=7
        ),
        'swin_B_384_22k': dict(
            embed_dim=128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=12
        ),
        'swin_L_224_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=7
        ),
        'swin_L_384_22k': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=12
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    model = SwinTransformer(pretrain_img_size=pretrain_img_size, **kw_cgf)
    return model

def build_backbone(args: UniPoseConfig):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    elif args.backbone == 'internimage_h':
        backbone = build_internimage_h(load_path=getattr(args,'backbone_load_path',None))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    return model

def build_position_encoding(args:UniPoseConfig):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSineHW(
            N_steps, 
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True
        )
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    binary_query_selection = False
    try:
        binary_query_selection = args.binary_query_selection
    except:
        binary_query_selection = False

    ffn_extra_layernorm = False
    try:
        ffn_extra_layernorm = args.ffn_extra_layernorm
    except:
        print('ffn_extra_layernorm not found, set to False')
        ffn_extra_layernorm = False

    #import pdb
    #pdb.set_trace()

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,

        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,

        binary_query_selection=binary_query_selection,
        ffn_extra_layernorm=ffn_extra_layernorm,
    )

# ------------------ InternImage -------------------
class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale) # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.mlp = MLPLayer(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 post_norm_block_ids=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None: # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x) # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class InternImage(nn.Module):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether tfo use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=64,
                 depths=[3, 4, 18, 5],
                 groups=[3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 level2_post_norm=False,  # for InternImage-H/G
                 level2_post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = [int(channels * 2 ** i) for i in range(self.num_levels)]
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        logger = get_root_logger()
        logger.info(f'using core type: {core_op}')
        logger.info(f'using activation layer: {act_layer}')
        logger.info(f'using main norm layer: {norm_layer}')
        logger.info(f'using dpr: {drop_path_type}, {drop_path_rate}')
        logger.info(f"level2_post_norm: {level2_post_norm}")
        logger.info(f"level2_post_norm_block_ids: {level2_post_norm_block_ids}")
        logger.info(f"res_post_norm: {res_post_norm}")

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                i == 2) else None # for InternImage-H/G
            level = InternImageBlock(
                core_op=getattr(opsm, core_op),
                channels=int(channels * 2**i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
            )
            self.levels.append(level)

        self.num_layers = len(depths)
        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(self.init_cfg['checkpoint'],
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            meg = self.load_state_dict(state_dict, False)
            print('loading internimage weights')
            logger.info(meg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(opsm, self.core_op)):
            m._reset_parameters()

    def forward_raw(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            x, x_ = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_.permute(0, 3, 1, 2).contiguous())
        return seq_out

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        outs = self.forward_raw(x)

        outs_dict = {}
        for idx, out_i in enumerate(outs):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=out_i.shape[-2:]).to(torch.bool)[0]
            outs_dict[idx] = NestedTensor(out_i, mask)

        return outs_dict

def build_internimage_h(load_path=None):
    cfg = dict(
        core_op='DCNv3',
        channels=320,
        depths=[6, 6, 32, 6],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.,
        norm_layer='LN',
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        dw_kernel_size=5,  # for InternImage-H/G
        res_post_norm=True,  # for InternImage-H/G
        level2_post_norm=True,  # for InternImage-H/G
        level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
        center_feature_scale=True,  # for InternImage-H/G
        with_cp=True,
        out_indices=(1, 2, 3),
        init_cfg=None # dict(type='Pretrained', checkpoint=pretrained)
    )
    if load_path is not None:
        print(f'--------------backbone load path: {load_path} ----------------')
        cfg['init_cfg'] = dict(type='Pretrained', checkpoint=load_path)
    backbone = InternImage(**cfg)
    backbone.num_features = [320,640,1280,2560]
    return backbone