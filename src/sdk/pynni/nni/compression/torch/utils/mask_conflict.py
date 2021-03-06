# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import torch
import numpy as np
from .shape_dependency import ChannelDependency, GroupDependency, CatPaddingDependency
logging.basicConfig(level = logging.DEBUG)
_logger = logging.getLogger('FixMaskConflict')

def fix_mask_conflict(masks, model=None, dummy_input=None, traced=None):
    """
    MaskConflict fix the mask conflict for the channel dependencies
    and group dependency.
    Parameters
    ----------
    model : torch.nn.Module
        model to fix the mask conflict
    dummy_input : torch.Tensor
        input example to trace the model
    masks : dict/str
        a dict object that stores the masks or the path of the mask file
    traced : torch._C.torch.jit.TopLevelTracedModule
        the traced model of the target model, is this parameter is not None,
        we donnot use the model and dummpy_input to get the trace graph.
    """
    if isinstance(masks, str):
        # if the input is the path of the mask_file
        assert os.path.exists(masks)
        masks = torch.load(masks)
    # if the user uses the model and dummy_input to trace the model, we
    # should get the traced model handly, so that, we only trace the
    # model once, GroupMaskConflict and ChannelMaskConflict will reuse
    # this traced model.
    if traced is None:
        assert model is not None and dummy_input is not None
        with torch.onnx.set_training(model, False):
            # We need to trace the model in this way, else it will have problems
            traced = torch.jit.trace(model, dummy_input)

    print("fixing group conflict...")
    fix_group_mask = GroupMaskConflict(masks, model, dummy_input, traced)
    masks = fix_group_mask.fix_mask()

    print("fixing channel conflict...")
    fix_channel_mask = ChannelMaskConflict(masks, model, dummy_input, traced)
    masks = fix_channel_mask.fix_mask()

    print("fixing cat padding conflict...")
    padding_cat_mask = CatMaskPadding(masks, model, dummy_input, traced)
    masks = padding_cat_mask.fix_mask()

    channel_depen = ChannelDependency(model,dummy_input, traced)
    sets = channel_depen.dependency_sets 
    print("*******", sets)
    # for dep in sets:
    #     sub_masks = [(d, masks[d]["weight"].sum(-1).sum(-1).sum(-1).to(torch.bool)) for d in dep]
    #     _, mask_0 = sub_masks[0]
    #     for name, m in sub_masks:
    #         res = all(m == mask_0)
    #         assert res, f'{name}, {m}'
    #         if 'depth_wise' in name:
    #             assert all(m == torch.Tensor(masks[name]["group"]).to(torch.bool)), f'{name}, {m}, {masks[name]["group"]}'
    return masks

class MaskFix:
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        # check if the parameters are valid
        parameter_valid = False
        if traced is not None:
            parameter_valid = True
        elif (model is not None) and (dummy_input is not None):
            parameter_valid = True
        if not parameter_valid:
            raise Exception('The input parameters is invalid!')
        self.model = model
        self.dummy_input = dummy_input
        self.traced = traced
        self.masks = masks

    def fix_mask(self):
        raise NotImplementedError

    def export(self, path):
        """
        Export the masks after fixing the conflict to file.
        """
        torch.save(self.masks, path)

class CatMaskPadding(MaskFix):
    def __init__(self, masks, model, dummy_input=None, traced=None):
        super(CatMaskPadding, self).__init__(masks, model, dummy_input, traced)

    def fix_mask(self):
        cat_padding_depen = CatPaddingDependency(self.model, self.dummy_input, self.traced)
        name_to_module = {}
        for name, module in self.model.named_modules():
            name_to_module[name] = module
        depen = cat_padding_depen.dependency_sets
        for layers in depen:
            device = None
            count = 0
            for layer in layers:
                if layer in self.masks:
                    count += 1
                    if device is None:
                        device = self.masks[layer]['weight'].device
            if count == 0:
                # no layer is pruned
                continue
            elif count == len(layers):
                # all the layers have been pruned
                continue
            # pad the mask for the non-pruned layers
            for layer in layers:
                if layer not in self.masks:
                    module = name_to_module[layer]
                    w_shape = module.weight.data.size()
                    w_mask = torch.ones(w_shape).to(device)
                    b_mask = None
                    if hasattr(module, 'bias') and module.bias is not None:
                        b_shape = module.bias.data.size()
                        b_mask = torch.ones(b_shape).to(device)
                    self.masks[layer] = {'weight':w_mask, 'bias':b_mask}
        return self.masks



class GroupMaskConflict(MaskFix):
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        """
        GroupMaskConflict fix the mask conflict between the layers that
        has group dependecy with each other.
        Parameters
        ----------
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        masks : dict
            a dict object that stores the masks
        traced : torch._C.torch.jit.TopLevelTracedModule
            the traced model of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        super(GroupMaskConflict, self).__init__(masks, model, dummy_input, traced)


    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has group dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        group_depen = GroupDependency(self.model, self.dummy_input, self.traced)
        depens = group_depen.dependency
        _logger.info(depens)
        for layername in depens:
            group = depens[layername]
            if layername not in self.masks:
                # this layer not pruned
                continue
            w_mask = self.masks[layername]['weight']
            shape = w_mask.size()
            # output channel
            count = np.prod(shape[1:])
            # unpruned channels
            all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
            # pruned channels
            all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
            if len(all_ones) + len(all_zeros) < w_mask.size(0):
                # In fine-grained pruning, skip this layer
                _logger.info('Layers %s using fine-grained pruning', layername)
                continue
            assert shape[0] % group == 0
            assert group == 1 or group == shape[0], "only normal conv and depth-wise conv are supported now."
            # Find the number of masked filter for each group (mini_masked).
            # Because we have to keep the pruned filter can still
            # be divided into the same number of groups, so we only can
            # prune mini_masked filters for each group.
            step = shape[0] / group
            group_masked = []
            # divide pruned channels into different groups
            for i in range(group):
                _start = step * i
                _end = step * (i+1)
                _tmp_list = list(filter(lambda x: _start <= x and x < _end, all_zeros))
                group_masked.append(_tmp_list)
            mini_masked = min([len(x) for x in group_masked if len(x) > 0] or [0])
            """
            the number of pruned channels accords to the most minimal pruned-number group
            e.g. group = 4, channel = 16
            group_masked: [[0, 1], [5], [8, 9, 10], [14, 15]]
            -> [[0], [5], [8], [14]]
            Moreover, because both of the input/output channels are divided into G groups,
            each group can be considered as an independent conv layer.
            i.e. 
                original conv: Ci * Co
                group conv: (Ci/G * Co/G) * G
            If all channels in the same group are pruned, then this whole group can be pruned,
            including the input channels.
            """
            # TODO: only support normal conv and depth-wise conv, i.e. conv.groups = weight.shape[0]
            self.masks[layername]['group'] = [1] * group
            for k, gm in enumerate(group_masked):
                if len(gm) == step:
                    # the whole group is pruned -> reduce the number of groups.
                    # input mask should be also update at the same time.
                    self.masks[layername]['group'][k] = 0
                    continue
                for i in range(mini_masked, len(gm)):
                    # To keep the output channel number still being divisible to
                    # groups, we set the masks of following filters to be zero.
                    pos = gm[i]
                    self.masks[layername]['weight'][pos] = torch.ones(shape[1:])
                    if hasattr(self.masks[layername], 'bias') or 'bias' in self.masks[layername] and self.masks[layername]['bias'] is not None:
                        self.masks[layername]['bias'][pos] = 1
        return self.masks



class ChannelMaskConflict(MaskFix):
    def __init__(self, masks, model=None, dummy_input=None, traced=None):
        """
        ChannelMaskConflict fix the mask conflict between the layers that
        has channel dependecy with each other.
        Parameters
        ----------
        model : torch.nn.Module
            model to fix the mask conflict
        dummy_input : torch.Tensor
            input example to trace the model
        masks : dict
            a dict object that stores the masks
        graph : torch._C.torch.jit.TopLevelTracedModule
            the traced graph of the target model, is this parameter is not None,
            we donnot use the model and dummpy_input to get the trace graph.
        """
        super(ChannelMaskConflict, self).__init__(masks, model, dummy_input, traced)

    def _fix_mask_for_one_depen(self, dset):
        if len(dset) == 1:
            # This layer has no channel dependency with other layers
            return
        channel_remain = set()
        fine_grained = False
        out_channels = None
        # A flag that represents if all the layers in
        # the dependency set are pruned
        all_pruned = True
        for name in dset:
            # only conv2d and linear op
            if name not in self.masks:
                # this layer is not pruned
                all_pruned = False
                continue
            w_mask = self.masks[name]['weight']
            group = self.masks[name].get('group')
            # TODO: depth-wise conv is pruned as BN
            if group is not None and len(group) == w_mask.shape[0]:
                # group conv
                # depth-conv is considered as bn
                continue
            if out_channels is None:
                out_channels = w_mask.size(0)
            shape = w_mask.size()
            count = np.prod(shape[1:])
            all_ones = (w_mask.flatten(1).sum(-1) == count).nonzero().squeeze(1).tolist()
            all_zeros = (w_mask.flatten(1).sum(-1) == 0).nonzero().squeeze(1).tolist()
            if len(all_ones) + len(all_zeros) < w_mask.size(0):
                # In fine-grained pruning, there is no need to check
                # the shape conflict
                _logger.info('Layers %s using fine-grained pruning', ','.join(dset))
                fine_grained = True
                break
            # reserve the intersection of all pruned channels
            channel_remain.update(all_ones)
            _logger.debug('Layer: %s ', name)
            _logger.debug('Original pruned filters: %s', str(all_zeros))
        # Update the masks for the layers in the dependency set
        if fine_grained:
            return
        if not all_pruned:
            # if some layer are not pruned at all
            # then all the layers in this dependency set
            # cannot be pruned due to the shape dependency.
            channel_remain.update(range(out_channels))
        self._set_mask_for_dset(dset, channel_remain)

    def _set_mask_for_dset(self, dset, channel_remain):
        ori_channels = 0
        for name in dset:
            if name not in self.masks:
                # this layer is not pruned at all
                # in this case, all_pruned is False
                # and the other layers in the same dset
                # will not be pruned either.
                continue
            mask = self.masks[name]
            # TODO: only support normal conv and depth-wise conv
            assert len(mask['group']) == 1 or len(mask['group']) == mask['weight'].shape[0], mask['group']
            w_shape = mask['weight'].size()
            for i in range(w_shape[0]):
                value = 1 if i in channel_remain else 0
                mask['weight'][i] = torch.ones(w_shape[1:]) * value
                if mask.get('bias') is not None:
                    mask['bias'][i] = value
                if mask.get('group') is not None and len(mask['group']) == mask['weight'].shape[0]:
                    mask['group'][i] = value
                
        _logger.info(','.join(dset))
        _logger.info('Pruned Filters after fixing conflict:')
        pruned_filters = set(list(range(ori_channels))) - channel_remain
        _logger.info(str(sorted(pruned_filters)))

    def fix_mask(self):
        """
        Fix the mask conflict before the mask inference for the layers that
        has shape dependencies. This function should be called before the
        mask inference of the 'speedup' module.
        """
        channel_depen = ChannelDependency(self.model, self.dummy_input, self.traced)
        depen_sets = channel_depen.dependency_sets
        for dset in depen_sets:
            self._fix_mask_for_one_depen(dset)
        return self.masks