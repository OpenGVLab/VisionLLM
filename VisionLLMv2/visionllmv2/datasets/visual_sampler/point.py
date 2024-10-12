import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class Point:
    def __init__(self, is_train=True):
        self.max_points = 20
        self.max_eval = 20
        self.is_train = is_train

    def draw(self, mask=None, box=None):
        # mask: [h, w], box: [4]
        if mask.sum() < 10:
            return torch.zeros(mask.shape).bool() # if mask is empty
        
        # for inference
        if not self.is_train:
            return self.draw_eval(mask=mask, box=box)
        # for training, randomly select some points as pos points
        max_points = min(self.max_points, mask.sum().item()) # max number of points no more than total mask number
        num_points = random.randint(1, max_points) # get a random number of points 
        h,w = mask.shape
        view_mask = mask.view(-1)
        non_zero_idx = view_mask.nonzero()[:,0] # get non-zero index of mask, [num_nonzero,]
        selected_idx = torch.randperm(len(non_zero_idx))[:num_points] # select id
        non_zero_idx = non_zero_idx[selected_idx] # select non-zero index
        rand_mask = torch.zeros(view_mask.shape).bool() # init rand mask
        rand_mask[non_zero_idx] = True # get non zero place to zero
        # dilate
        # struct = ndimage.generate_binary_structure(2, 2)
        # rand_mask = torch.from_numpy((ndimage.binary_dilation(rand_mask.reshape(h, w).numpy(), structure=struct, iterations=5).astype(rand_mask.numpy().dtype)))
        # return rand_mask
        return rand_mask.reshape(h, w)  
    
    def draw_eval(self, mask=None, box=None):
        background = ~mask
        neg_num = min(self.max_eval // 2, background.sum().item())        
        pos_num = min(self.max_eval - neg_num, mask.sum().item()-1) + 1   

        h,w = mask.shape
        view_mask = mask.view(-1)
        non_zero_idx_pos = view_mask.nonzero()[:,0] # get non-zero index of mask
        selected_idx_pos = torch.randperm(len(non_zero_idx_pos))[:pos_num] # select id
        non_zero_idx_pos = non_zero_idx_pos[selected_idx_pos] # select non-zero index
        pos_idx = torch.ones(non_zero_idx_pos.shape)

        view_background = background.view(-1)
        non_zero_idx_neg = view_background.nonzero()[:,0] # get non-zero index of mask
        selected_idx_neg = torch.randperm(len(non_zero_idx_neg))[:neg_num] # select id
        non_zero_idx_neg = non_zero_idx_neg[selected_idx_neg] # select non-zero index
        neg_idx = torch.ones(non_zero_idx_neg.shape) * -1

        non_zero_idx = torch.cat([non_zero_idx_pos, non_zero_idx_neg])  # [num_pos + num_zero,]
        idx = torch.cat([pos_idx, neg_idx])  # [1, 1, 1, -1, -1, -1]
        rand_idx = torch.cat([torch.zeros(1), torch.randperm(len(non_zero_idx)-1) + 1]).long()  # random sample idx, at least one pos
        non_zero_idx = non_zero_idx[rand_idx]
        idx = idx[rand_idx]

        rand_masks = []
        for i in range(0, len(non_zero_idx)): 
            rand_mask = torch.zeros(view_mask.shape) # init rand mask
            rand_mask[non_zero_idx[0:i+1]] = idx[0:i+1] # get non zero place to zero， 1 or -1
            # struct = ndimage.generate_binary_structure(2, 2)
            # rand_mask = torch.from_numpy((ndimage.binary_dilation(rand_mask.reshape(h, w).numpy(), structure=struct, iterations=5).astype(rand_mask.numpy().dtype)))
            rand_masks += [rand_mask.reshape(h, w)]

        # kernel_size = 3
        rand_masks = torch.stack(rand_masks)  # [num_select, h, w], forground 1, background -1
        # rand_masks = F.conv2d(rand_masks[:,None], torch.ones(1,1,kernel_size,kernel_size), padding=kernel_size//2)[:,0]
        # rand_masks[rand_masks>0] = 1
        # rand_masks[rand_masks<0] = -1
        return rand_masks
    
    def __repr__(self,):
        return 'point'