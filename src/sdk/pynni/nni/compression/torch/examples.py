import logging
import os
import copy

from collections import OrderedDict

import torch
import torch.nn as nn

from nni.compression.torch import apply_compression_results
from nni.compression.torch.utils.mask_conflict import fix_mask_conflict
from nni.compression.torch.speedup import ModelSpeedup

__all__ = []

logger = logging.getLogger('torch filter pruners')

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        mid_channel = int(channel // reduction)

        self.se = nn.Sequential(OrderedDict([
            ("reduction", nn.Conv2d(self.channel, mid_channel, 1, 1, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("expand", nn.Conv2d(mid_channel, self.channel, 1, 1, 0)),
            ("activation", nn.Sigmoid())
        ]))

    def forward(self, inputs):
        out = inputs.mean(3, keepdim=True).mean(2, keepdim=True)
        out = self.se(out)
        return inputs * out

class MobileNetV3Block(torch.nn.Module):
    def __init__(self, expansion, C, C_out, 
                 stride, 
                 kernel_size,
                 affine,
                 activation="relu",
                 use_se=False,
                 se=None,
                 ):
        super(MobileNetV3Block, self).__init__()
        self.expansion = expansion
        self.C = C
        self.C_out = C_out 
        self.C_inner = int(C * expansion)
        self.stride = stride
        self.kernel_size = kernel_size
        self.act_fn = nn.ReLU
        self.use_se = use_se
        
        self.inv_bottleneck = None
        if expansion != 1:
            self.inv_bottleneck = nn.Sequential(
                nn.Conv2d(C, self.C_inner, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.C_inner),
                self.act_fn(inplace=True)
            )
        
        self.depth_wise = nn.Sequential(
            nn.Conv2d(self.C_inner, self.C_inner, self.kernel_size, stride, self.kernel_size // 2, groups=self.C_inner, bias=False),
            nn.BatchNorm2d(self.C_inner),
            self.act_fn(inplace=True)
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(self.C_inner, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.se = None
        if self.use_se:
            self.se = SEModule(self.C_inner)
        
        self.shortcut = None
        if stride == 1 and C == C_out:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        out = inputs
        if self.inv_bottleneck:
            out = self.inv_bottleneck(out)
        out = self.depth_wise(out)
        if self.se:
            out = self.se(out)
        out = self.point_linear(out)
        if self.shortcut is not None:
            out = out + self.shortcut(inputs)
        return out


class MobileNetStage(nn.Module):
    def __init__(self, C, C_out, expansion):
        super().__init__()
        self.blocks = nn.Sequential(    
            MobileNetV3Block(expansion, C, C_out, 2, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 2, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
            MobileNetV3Block(expansion, C_out, C_out, 1, 3, True, use_se=True),
        )

    def forward(self, inputs):
        return self.blocks(inputs)

import numpy as np
def count_state_dict(state_dict):
    res = 0
    for k, v in state_dict.items():
        res += np.prod(v.shape)
    return res


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.copy_(torch.rand(m.weight.shape))
        nn.init.constant_(m.bias, 1)
        

def speedup(model, dummy, fixed_mask, trace=None):
    m_speedup = ModelSpeedup(model, dummy, fixed_mask, trace)
    m_speedup.speedup_model()
    return m_speedup.bound_model


def count_zero(model, verbose=False):
    all_count = 0
    all_total = 0
    for name, param in model.named_parameters():
        count = torch.sum((param == 0).int()).data
        all_count += count
        all_total += param.numel()
        if verbose:
            print(
                "{}: shape {}, zeros {}, total {}, ratio {:.3f}".format(name, list(param.shape), count, param.numel(),
                                                                        float(count) / param.numel()))
    print(
        "[Count Zero] total number of zeros: {}, total number of param values: {}, overall ratio: {:.6f}".format(
            all_count,
            all_total,
            float(all_count) / all_total))


def update_sparsity_by_sensitivity(config_list, ori_metric, metric_thres, sensitivity):
    layer_sparsity = 0.0
    for layername in sensitivity:
        for sparsity in sorted(sensitivity[layername].keys()):
            if ori_metric - sensitivity[layername][sparsity] >  metric_thres:
                new_cfg = {'sparsity': layer_sparsity, 'op_types': ['Conv2d'], 'op_names': [layername]}
                config_list.insert(0, new_cfg)
                break
            else:
                layer_sparsity = sparsity
    return config_list


def compress(model, dummy, pruner_cls, config_list, ori_metric=1.00, metric_thres=0.01, sensitivity=None, trace=None,
             verbose=True):
    if sensitivity:
        config_list = update_sparsity_by_sensitivity(config_list, ori_metric, metric_thres, sensitivity)
    compressed_model = copy.deepcopy(model)
    pruner = pruner_cls(compressed_model, config_list)
    compressed_model = pruner.compress()

    mask_path = "/tmp/mask.pth"
    pruner.export_model(model_path='/tmp/model.pth', mask_path=mask_path)
    pruner._unwrap_model()

    print("fixing mask conflict...")
    fixed_mask = fix_mask_conflict(mask_path, compressed_model, dummy, trace)
    mask = torch.load(mask_path)

    compressed_model.load_state_dict(model.state_dict())
    apply_compression_results(compressed_model, fixed_mask)
    if verbose:
        count_zero(compressed_model, verbose=False)
        from thop import profile
        macs, params = profile(compressed_model, inputs=dummy, verbose=False)
        print("MACs: {} G, Params: {} M".format(macs / 1000000000, params / 100000))
    speedup_model = speedup(compressed_model, dummy, fixed_mask, trace)
    if verbose:
        count_zero(speedup_model, verbose=False)
    return speedup_model, fixed_mask


def load_state_dict(model, dummy, model_state_dict, fixed_mask, strict=False):
    m_speedup = speedup(model, dummy, fixed_mask)
    m_speedup.bound_model.load_state_dict(model_state_dict)
    return m_speedup


def save_state_dict(m_speedup, fixed_mask, model_path, mask_path):
    torch.save(m_speedup, model_path)
    torch.save(fixed_mask, mask_path)


from .pruning.one_shot import L1FilterPruner, SlimPruner

# model = MobileNetStage(16, 24, 6)
# dummy = torch.rand(2, 16, 28, 28)
# params = count_state_dict(model.state_dict())
# print("param count:", params)

# model.apply(init_weight)
# config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}, {'sparsity': 0.5, 'op_types': ['Conv2d'], 'op_names': ['blocks.0.inv_bottleneck.0']}, {'exclude': True, 'op_names': ['backbone.classifier']}]

# m_speedup, masks = compress(model, dummy, L1FilterPruner, config_list)

# params = count_state_dict(m_speedup.state_dict())
# print("new param count:", params)