from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
import torch
import copy
import math
from typing import Optional, List
from torch import nn, Tensor
import torch.nn as nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class Resnet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(Resnet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = resnet50(pretrained)
        del model.avgpool
        del model.fc
        self.model = model
        self.ps = nn.PixelShuffle(2)

    def forward(self, image, face):
        # image = self.model.conv1(image)
        # image = self.model.bn1(image)
        # image = self.model.relu(image)
        # image = self.model.maxpool(image)
        #
        # image = self.model.layer1(image)
        # feat1 = self.model.layer2(image)
        # feat2 = self.model.layer3(feat1)
        # feat3 = self.model.layer4(feat2)

        # face = self.model.conv2(face)
        # face = self.model.bn2(face)
        # face = self.model.relu(face)
        # face = self.model.maxpool(face)
        #
        # face = self.model.layer1(face)
        # face = self.model.layer2(face)
        # face = self.model.layer3(face)
        # face = self.model.layer4(face)

        image = self.model.layer5_scene(image)
        face = self.model.layer5_face(face)


        return image, face


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Pixel_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_shuffle, self).__init__()

        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(2),
            conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x, ):
        x = self.pixel_shuffle(x)
        return x


def get_gt_gaze_cone(eyes, points, size):
    bs = len(eyes)
    eyes = eyes * np.array([size, size])

    points = points * np.array([size, size])
    gt_gaze_cone = np.zeros((bs, 1, size, size), dtype=np.float32)

    apha = 60

    for gaze_cone, eye, point in zip(gt_gaze_cone, eyes, points):
        # eye坐标和gaze point坐标
        eye_x = eye[0]
        eye_y = eye[1]
        gaze_x = point[0]
        gaze_y = point[1]

        # 计算eye到gaze point的向量
        vector_x = gaze_x - eye_x
        vector_y = gaze_y - eye_y


        for y in range(size):

            for x in range(size):
                energy = (vector_x * (x - eye_x) + vector_y * (y - eye_y)) / (
                        np.sqrt(vector_x ** 2 + vector_y ** 2) * np.sqrt((x - eye_x) ** 2 + (y - eye_y) ** 2))
                energy = np.clip(energy, -1.0, 1.0)
                energy = np.arccos(energy)
                m = 60 * energy / 180


                gaze_cone[:, y, x] = max(1 - m, 0)


    return gt_gaze_cone


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

def get_head_channel(pred_head, gt_heads, orig_target_sizes):

    head_channels = []
    is_head = []
    for i, boxes in enumerate(pred_head):
        box = boxes['boxes']
        conf = boxes['scores']
        h, w = orig_target_sizes[i]
        if len(gt_heads) == 0:
            if gt_heads is None or len(conf) != 0 and conf > 0.5:
                box1 = box[0]
                is_head.append(1)
                head_c = torch.zeros(1, 224, 224)
                xmin, ymin, xmax, ymax = box1
                xmin = xmin * 224
                ymin = ymin * 224
                xmax = xmax * 224
                ymax = ymax * 224
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                head_c[0, xmin:xmax + 1, ymin:ymax + 1] = 1
                head_channels.append(head_c)
            else:
                head_c = torch.zeros(1, 224, 224)
                is_head.append(0)
                head_channels.append(head_c)
        else:
            if box.numel() == 0:
                head_c = torch.zeros(1, 224, 224)
                is_head.append(0)
                head_channels.append(head_c)
            else:
                box1 = box[0]
                box2 = gt_heads[i]

                box2[0] *= w
                box2[1] *= h
                box2[2] *= w
                box2[3] *= h
                # 计算相交区域的左上角和右下角坐标
                x1 = max(box1[0], box2[0])

                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                # 计算相交区域面积
                intersection = max(0, x2 - x1) * max(0, y2 - y1)

                # 计算两个框各自面积
                area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                # 计算并集面积
                union = area_box1 + area_box2 - intersection

                # 计算 IOU
                iou = intersection / union
                if iou > 0.5:
                    is_head.append(1)
                    head_c = torch.zeros(1, 224, 224)
                    xmin, ymin, xmax, ymax = box1
                    xmin = xmin * 224
                    ymin = ymin * 224
                    xmax = xmax * 224
                    ymax = ymax * 224
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    head_c[0, xmin:xmax + 1, ymin:ymax + 1] = 1
                    head_channels.append(head_c)
                else:
                    head_c = torch.zeros(1, 224, 224)
                    is_head.append(0)
                    head_channels.append(head_c)

    head_channels = torch.cat([items.unsqueeze(0) for items in head_channels], 0)

    head_channels_ = head_channels.cuda()

    return head_channels_, is_head

class GaTectorBody(nn.Module):
    def __init__(self, args):
        super(GaTectorBody, self).__init__()

        self.backbone = Resnet(phi='resnet50', pretrained=True)
        # self.backbone = darknet53(None)

        # GOO network
        # common



        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # head pathway
        self.conv_face_scene = nn.Conv2d(2560, 2048, kernel_size=1)
        self.conv_trblock = nn.Conv2d(args.gaze_hidden_size, 256, kernel_size=1)
        self.trblock_bn = nn.BatchNorm2d(256)

        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU()
        )
        # attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)


        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)
        self.compress_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn3 = nn.BatchNorm2d(256)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # gaze cone
        self.gaze_cone_head = nn.ConvTranspose2d(256, 1, kernel_size=3, stride=2)
        self.gaze_cone_head_bn = nn.BatchNorm2d(1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=3)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        self.totrans_conv = nn.Conv2d(256, args.gaze_hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.totrans_conv_bn1 = nn.BatchNorm2d(args.gaze_hidden_size)

        self.transformer_layer = transformer_layer(d_model=args.gaze_hidden_size, dropout=args.dropout,
                                                   nhead=args.nheads, dim_feedforward=args.dim_feedforward,
                                                   num_encoder_layers=0, num_decoder_layers=1,
                                                   normalize_before=args.pre_norm, return_intermediate_dec=True,
                                                   )

    def forward(self, tensor_list, face, head_boxes, head_targets, orig_target_sizes, od_memory, od_pos, od_mask):
        #  backbone

        # [2, 3, 224, 224] --> [2, 1024, 7, 7]
        scene_feat, face_feat = self.backbone(tensor_list, face)

        # GOO
        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        # get head channel
        head, is_head = get_head_channel(head_boxes, head_targets, orig_target_sizes)

        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        head_conv = self.conv_block(head)

        # attn_weights = torch.ones(attn_weights.shape)/49.0
        scene_feat = torch.cat((scene_feat, head_conv), 1)
        attn_applied_scene_feat = torch.mul(attn_weights,
                                            scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        scene_face_feat = self.conv_face_scene(scene_face_feat)  # (b, 2048, 7, 7)

        cone_feat = scene_face_feat.detach().clone()

        encoding = self.compress_conv1(scene_face_feat) # (b, 1024, 7, 7)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding) # (b, 512, 7, 7)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv3(encoding)  # (b, 256, 7, 7)
        encoding = self.compress_bn3(encoding)
        encoding = self.relu(encoding)

        # cross K V
        bs, c, h, w = encoding.shape
        device = tensor_list.device

        # decoder Q
        # src = encoding

        src = self.totrans_conv(encoding) # (b, 256, 7, 7 )
        src = self.totrans_conv_bn1(src)
        src = self.relu(src)

        # gaze cone
        cone_encoding = self.compress_conv1(cone_feat)  # (b, 1024, 7, 7)
        cone_encoding = self.compress_bn1(cone_encoding)
        cone_encoding = self.relu(cone_encoding)
        cone_encoding = self.compress_conv2(cone_encoding)  # (b, 512, 7, 7)
        cone_encoding = self.compress_bn2(cone_encoding)
        cone_encoding = self.relu(cone_encoding)
        cone_encoding = self.compress_conv3(cone_encoding)  # (b, 256, 7, 7)
        cone_encoding = self.compress_bn3(cone_encoding)
        cone_encoding = self.relu(cone_encoding)

        cone_src = self.totrans_conv(cone_encoding)  # (b, 256, 7, 7 )
        cone_src = self.totrans_conv_bn1(cone_src)
        cone_src = self.relu(cone_src)

        gaze_cone = self.gaze_cone_head(cone_src)
        gaze_cone = self.gaze_cone_head_bn(gaze_cone)

        mask_shape = (bs, h, w)  # 3 dimensions with size 4, 5 and 6 respectively
        mask = torch.zeros(mask_shape, dtype=torch.bool).to(device)
        # mask = tensor_list.mask
        pos = make_pos(mask, c/2)

        hs = self.transformer_layer(src, mask, pos, od_memory, od_pos, od_mask)
        hs = hs.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        # scene + face feat -> encoding -> decoding
        cross_features = self.conv_trblock(hs)
        cross_features = self.trblock_bn(cross_features)
        cross_features = self.relu(cross_features)

        x = self.deconv2(cross_features)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x, gaze_cone

class transformer_layer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        #
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, od_memory, od_pos, od_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        enc_output = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        tgt = enc_output
        query_embed = pos_embed

        # cross#
        # # cat #
        # od_memory = torch.cat((od_memory, od_memory), dim=2)
        # od_pos = torch.cat((od_pos, od_pos), dim=2)


        hs = self.decoder(tgt, od_memory.permute(1, 0, 2), memory_key_padding_mask=od_mask,
                          pos=od_pos.permute(1, 0, 2), query_pos=query_embed)

        hs = hs.transpose(1, 2)

        return hs[-1]
#
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
#
#
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)
#
#
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
#
#
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.hidden_size = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.fc_to512 = nn.Linear(256, self.hidden_size)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Liner #
        # k 256 -> 512
        od_k = self.with_pos_embed(memory, pos) #(1045, 2, 256)
        # od_k = od_k.permute(1, 0, 2)
        # batch_size, input_size, seq_len = od_k.size()
        # od_k = od_k.contiguous().view(batch_size * input_size, seq_len)
        # od_k = self.fc_to512(od_k)
        # od_k = od_k.view(batch_size, input_size, self.hidden_size)  # (2,1045,512)
        # od_k = od_k.permute(1, 0, 2)
        # v 256 -> 512
        od_v = memory #(1045, 2, 256)
        # od_v = od_v.permute(1, 0, 2)
        # batch_size, input_size, seq_len = od_v.size()
        # od_v = od_v.contiguous().view(batch_size * input_size, seq_len)
        # od_v = self.fc_to512(od_v)
        # od_v = od_v.view(batch_size, input_size, self.hidden_size)  # (2,1045,512)
        # od_v = od_v.permute(1, 0, 2)
        # # cat#
        # od_k = self.with_pos_embed(memory, pos)
        # od_v = memory

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=od_k,
                                   value=od_v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def make_pos(mask, hidden_dim):

    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    scale = 2 * math.pi

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_tx = torch.arange(hidden_dim, dtype=torch.float32, device=mask.device)
    dim_tx = 20 ** (2 * (dim_tx // 2) / hidden_dim)
    pos_x = x_embed[:, :, :, None] / dim_tx

    dim_ty = torch.arange(hidden_dim, dtype=torch.float32, device=mask.device)
    dim_ty = 20 ** (2 * (dim_ty // 2) / hidden_dim)
    pos_y = y_embed[:, :, :, None] / dim_ty

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)



    return pos


def build_gatector(args):


    return GaTectorBody(args)