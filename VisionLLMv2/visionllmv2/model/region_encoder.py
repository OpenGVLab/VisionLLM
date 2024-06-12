import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
# From https://github.com/facebookresearch/detectron2/blob/3ff5dd1cff4417af07097064813c9f28d7461d3c/projects/PointRend/point_rend/point_features.py#L19
def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

# random sample for each batch
def rand_sample(x, divisor, max_len):
    if len(x.nonzero()) == 0:
        return x.nonzero()  # [0, 3]

    non_zero_point_index = (x.nonzero()/divisor).t()   
    mask_ids = non_zero_point_index[0].unique().long() 

    # compute probability for each samle
    probs = torch.zeros_like(non_zero_point_index[0])   # [n_nonzero,]
    for idx in mask_ids:
        prob = 1./(len(mask_ids)*((non_zero_point_index[0:1]==idx).sum()))
        probs[non_zero_point_index[0]==idx] = prob
    
    indices = torch.multinomial(probs, num_samples=min(max_len, len(probs)), replacement=False).sort()[0]
    non_zero_point_index = non_zero_point_index[:,indices]  
    return non_zero_point_index.t() 


class RegionEncoder(nn.Module):
    def __init__(self, hidden_dim, embed_dim, out_dim, patch_size=14, mask_pool_type='mean'):
        super().__init__()
        # NOTE: patch_size=7x2, since clip patch_size=14
        assert patch_size % 2 == 0
        kernel_size = patch_size // 2
        self.patch_size = patch_size
        self.mask_embedding = nn.Sequential(                
            nn.Conv2d(4, hidden_dim // 4, kernel_size=kernel_size, stride=kernel_size),
            LayerNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=2, stride=2), 
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )
        
        self.mask_pool_type = mask_pool_type  # 'mean', 'cross_attn', 'grid_sample'
        assert mask_pool_type in ['mean', 'cross_attn', 'grid_sample']
        if self.mask_pool_type == 'cross_attn':
            # using a learnable query 
            self.region_query = nn.Embedding(1, embed_dim)
            self.region_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0., batch_first=True)
        elif self.mask_pool_type == 'grid_sample':
            self.num_points = 12544  # 112 x 112
        self.up_dim = nn.Linear(embed_dim, out_dim)

    def forward(self, images, masks, image_features):
        assert images.shape[-2:] == masks.shape[-2:] 

        # encode mask features
        masks = masks.to(images.dtype)
        masks_in = torch.cat([images, masks], dim=1)  
        masks_out = self.mask_embedding(masks_in)    
        bs = len(images)
        h, w = masks_out.shape[-2:]  

        outs = []
        for image_features_per_level in image_features:
            if image_features_per_level.dim() == 3: 
                image_features = image_features_per_level.reshape(bs, h, w, -1)
                image_features = image_features.permute(0, 3, 1, 2)  
            else:
                image_features = image_features_per_level
            assert masks_out.shape[-2:] == image_features.shape[-2:]
            masks_out = masks_out + image_features

            # get region features
            if self.mask_pool_type == 'mean':
                masks_binary = F.interpolate(masks.float(), size=(h, w), mode='bilinear', align_corners=False) > 0.5
                masks_out = masks_out * masks_binary      
                out = masks_out.mean(-1).mean(-1)       
                out = self.up_dim(out)                     
            elif self.mask_pool_type == 'cross_attn':
                masks_out = masks_out.flatten(-2).transpose(1, 2)                         
                region_query = self.region_query.weight.unsqueeze(0).repeat(bs, 1, 1)     
                out = self.region_attn(region_query, masks_out, masks_out)[0].squeeze(1) 
                out = self.up_dim(out)                 
            elif self.mask_pool_type == 'grid_sample':
                ori_h, ori_w = masks.shape[-2:]
                divisor = torch.tensor([1, ori_h, ori_w], device=masks.device)[None,]     
                non_zero_pos_point = [rand_sample(m, divisor, self.num_points) for m in masks]  
                non_zero_pos_index = [m[:,0:1].long() for m in non_zero_pos_point]       
                non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)        
                non_zero_pos_index = nn.utils.rnn.pad_sequence(non_zero_pos_index, padding_value=-1).permute(1,0,2)[:,:,0]  
                non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) >= 0)     
                point_coords = non_zero_pos_point[:, :, -2:].flip(dims=[-1])           
                dtype = masks_out.dtype
                sample_features = point_sample(masks_out.float(), point_coords.float(), align_corners=False).permute(0, 2, 1)  
                sample_features = sample_features.to(dtype)
                # get region features, [bs, c]
                out = sample_features * non_zero_pos_mask.unsqueeze(-1)    
                out = (out.sum(1) / non_zero_pos_mask.sum(1).unsqueeze(-1)).nan_to_num()  
                out = self.up_dim(out)
            else:
                raise NotImplementedError
            outs.append(out)
        out = torch.stack(outs).mean(dim=0)  # [bs, c], mean on multi-scale features       
        return out