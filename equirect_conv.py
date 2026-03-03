"""
Equirectangular-aware convolution module (SkyCloudNet-EQC).

Implements latitude-adaptive convolutions that adjust the sampling grid based on
the spherical geometry of equirectangular projections. At higher latitudes (near
poles), horizontal pixels are stretched, so the convolution kernel compensates by
adjusting horizontal offsets according to 1/cos(latitude).

Reference: Tateno et al., "Distortion-Aware Convolutional Filters for Dense
Prediction in Panoramic Images" (ECCV 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EquirectConv2d(nn.Module):
    """
    Equirectangular-aware 2D convolution.
    
    Uses deformable-convolution-style offset sampling to adapt the kernel grid
    to the local geometry of an equirectangular projection. The horizontal offsets
    are scaled by 1/cos(latitude) to compensate for the varying pixel density.
    
    Can be used as a drop-in replacement for nn.Conv2d.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(EquirectConv2d, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Standard conv weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        
        # Cache for latitude scale factors
        self._cached_h = None
        self._cached_w = None
        self._cached_offsets = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _compute_offsets(self, h, w, device):
        """
        Compute latitude-dependent sampling offsets for the convolution kernel.
        
        For each spatial location and each kernel position, compute the adjusted
        offset that accounts for the equirectangular distortion.
        """
        if self._cached_h == h and self._cached_w == w and self._cached_offsets is not None:
            return self._cached_offsets.to(device)
        
        kh, kw = self.kernel_size
        dh, dw = self.dilation
        
        # Compute latitude for each row
        # Output feature map spatial dimensions
        out_h = (h + 2 * self.padding[0] - dh * (kh - 1) - 1) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - dw * (kw - 1) - 1) // self.stride[1] + 1
        
        # Center pixel positions in the input (accounting for stride and padding)
        cy = torch.arange(out_h, device=device).float() * self.stride[0] - self.padding[0]
        cx = torch.arange(out_w, device=device).float() * self.stride[1] - self.padding[1]
        
        # Latitude for each output row
        # Map from pixel y to latitude: lat = pi/2 - (y + 0.5) / h * pi
        lat = math.pi / 2.0 - (cy + 0.5) / h * math.pi
        cos_lat = torch.cos(lat).clamp(min=0.1)  # avoid extreme values near poles
        scale = 1.0 / cos_lat  # (out_h,)
        
        # Standard kernel offsets (relative to center)
        ky_offsets = torch.arange(kh, device=device).float() - (kh - 1) / 2.0
        kx_offsets = torch.arange(kw, device=device).float() - (kw - 1) / 2.0
        
        # Apply dilation
        ky_offsets = ky_offsets * dh
        kx_offsets = kx_offsets * dw
        
        # Scale horizontal offsets by latitude factor
        # shape: (out_h, kw) - different scale per row
        kx_scaled = kx_offsets.unsqueeze(0) * scale.unsqueeze(1)  # (out_h, kw)
        
        # Additional horizontal offset = scaled - original
        delta_x = kx_scaled - kx_offsets.unsqueeze(0)  # (out_h, kw)
        
        # Store offsets: for each output position, for each kernel element
        # We need (kh * kw) offset pairs (dy, dx) for each output position
        # But dy offsets remain unchanged; only dx changes
        offsets = delta_x  # (out_h, kw)
        
        self._cached_h = h
        self._cached_w = w
        self._cached_offsets = offsets.detach()
        
        return offsets
    
    def forward(self, x):
        """
        Forward pass with latitude-adaptive sampling.
        
        For efficiency, this implements the equirectangular convolution by:
        1. Computing latitude-dependent offsets
        2. Sampling the input at adjusted positions using grid_sample
        3. Applying pointwise convolution
        """
        B, C, H, W = x.shape
        kh, kw = self.kernel_size
        dh, dw = self.dilation
        
        out_h = (H + 2 * self.padding[0] - dh * (kh - 1) - 1) // self.stride[0] + 1
        out_w = (W + 2 * self.padding[1] - dw * (kw - 1) - 1) // self.stride[1] + 1
        
        # Compute latitude-dependent horizontal offsets
        delta_x = self._compute_offsets(H, W, x.device)  # (out_h, kw)
        
        # Standard kernel offsets
        ky_offsets = (torch.arange(kh, device=x.device).float() - (kh - 1) / 2.0) * dh
        kx_offsets = (torch.arange(kw, device=x.device).float() - (kw - 1) / 2.0) * dw
        
        # Pad input
        if self.padding[0] > 0 or self.padding[1] > 0:
            # Use circular padding for horizontal (wrap-around) and zero for vertical
            x_padded = F.pad(x, 
                           [self.padding[1], self.padding[1], 0, 0], 
                           mode='circular')
            x_padded = F.pad(x_padded, 
                           [0, 0, self.padding[0], self.padding[0]], 
                           mode='constant', value=0)
        else:
            x_padded = x
        
        _, _, padH, padW = x_padded.shape
        
        # For each kernel position, sample the input
        # Center positions
        cy = torch.arange(out_h, device=x.device).float() * self.stride[0] + self.padding[0]
        cx = torch.arange(out_w, device=x.device).float() * self.stride[1] + self.padding[1]
        
        grid_cy, grid_cx = torch.meshgrid(cy, cx, indexing='ij')  # (out_h, out_w)
        
        # Collect samples for all kernel positions
        samples = []
        for i in range(kh):
            for j in range(kw):
                # Standard offset
                dy = ky_offsets[i]
                dx = kx_offsets[j]
                
                # Latitude-adjusted additional horizontal offset
                # delta_x has shape (out_h, kw), index by j
                dx_adj = delta_x[:, j].unsqueeze(1).expand(-1, out_w)  # (out_h, out_w)
                
                # Sample positions
                sy = grid_cy + dy
                sx = grid_cx + dx + dx_adj
                
                # Handle horizontal wrap-around
                sx = sx % padW
                
                # Normalize to [-1, 1] for grid_sample
                norm_x = 2.0 * sx / (padW - 1) - 1.0
                norm_y = 2.0 * sy / (padH - 1) - 1.0
                
                grid = torch.stack([norm_x, norm_y], dim=-1)
                grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
                
                sampled = F.grid_sample(x_padded, grid, mode='bilinear',
                                       padding_mode='border', align_corners=True)
                samples.append(sampled)
        
        # Stack samples: (B, C, kh*kw, out_h, out_w)
        samples = torch.stack(samples, dim=2)
        
        # Reshape weight: (out_c, C/groups, kh, kw) -> (out_c, C/groups, kh*kw)
        weight = self.weight.view(self.out_channels, -1, kh * kw)
        
        # Apply convolution as matrix multiply
        # samples: (B, C, kh*kw, out_h, out_w) -> (B, out_h, out_w, C, kh*kw)
        samples = samples.permute(0, 3, 4, 1, 2)
        
        if self.groups == 1:
            # (B, out_h, out_w, C, kh*kw) x (out_c, C*kh*kw) -> (B, out_h, out_w, out_c)
            samples_flat = samples.reshape(B * out_h * out_w, C * kh * kw)
            weight_flat = weight.reshape(self.out_channels, -1).t()  # (C*kh*kw, out_c)
            output = torch.mm(samples_flat, weight_flat)
            output = output.reshape(B, out_h, out_w, self.out_channels)
        else:
            # Group convolution
            c_per_group = C // self.groups
            oc_per_group = self.out_channels // self.groups
            output = torch.zeros(B, out_h, out_w, self.out_channels, device=x.device)
            for g in range(self.groups):
                s = samples[..., g * c_per_group:(g + 1) * c_per_group, :]
                w = weight[g * oc_per_group:(g + 1) * oc_per_group]
                s_flat = s.reshape(B * out_h * out_w, c_per_group * kh * kw)
                w_flat = w.reshape(oc_per_group, -1).t()
                o = torch.mm(s_flat, w_flat)
                output[..., g * oc_per_group:(g + 1) * oc_per_group] = o.reshape(
                    B, out_h, out_w, oc_per_group)
        
        output = output.permute(0, 3, 1, 2)  # (B, out_c, out_h, out_w)
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output


def replace_conv_with_equirect(module, replace_first=False):
    """
    Recursively replace nn.Conv2d layers with EquirectConv2d.
    Only replaces 3x3 and larger convolutions (not 1x1 pointwise).
    
    Args:
        module: nn.Module to modify in-place
        replace_first: if True, also replace the first convolution layer
    """
    first_conv_found = False
    
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if not first_conv_found and not replace_first:
                first_conv_found = True
                continue
            
            # Only replace convolutions with kernel_size > 1
            ks = child.kernel_size
            if isinstance(ks, tuple):
                ks = ks[0]
            if ks <= 1:
                continue
            
            # Create replacement equirectangular convolution
            eq_conv = EquirectConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None
            )
            
            # Copy weights
            eq_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                eq_conv.bias.data.copy_(child.bias.data)
            
            setattr(module, name, eq_conv)
        else:
            replace_conv_with_equirect(child, replace_first)
    
    return module
