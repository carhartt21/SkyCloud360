"""
SkyCloudNet 360° adaptation wrappers.

This module implements four geometric adaptation strategies for processing
equirectangular images, as described in the SkyCloud360 paper:

- SkyCloudNet-CM:  Standard cubemap decomposition (6 x 90° faces)
- SkyCloudNet-ECM: Extended cubemap with overlapping faces for seamless blending
- SkyCloudNet-TPP: Tangent plane projection with multiple sampling points
- SkyCloudNet-EQC: Equirectangular convolutions (latitude-adaptive kernels)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from equirect_utils import (
    equirect_to_cubemap,
    cubemap_to_equirect,
    equirect_to_tangent,
    tangent_to_equirect,
    generate_tangent_points,
)
from equirect_conv import replace_conv_with_equirect


class SkyCloudNet360Base(nn.Module):
    """
    Base class for all SkyCloudNet 360° adaptations.
    
    Wraps the original MultiLabelModule and UnsupervisedSegmentationModule,
    applying geometric transformations before and after the standard forward pass.
    """
    
    def __init__(self, multilabel_module, cloud_seg_module):
        super(SkyCloudNet360Base, self).__init__()
        self.multilabel_module = multilabel_module
        self.cloud_seg_module = cloud_seg_module
    
    def forward_base(self, feed_dict, seg_size=None):
        """Run the standard SkyCloudNet forward pass."""
        if self.training:
            return self.multilabel_module(feed_dict, seg_size=seg_size)
        else:
            encoder_result, pred = self.multilabel_module(feed_dict, seg_size=seg_size)
            pred_seg = pred[1]
            pred_cloud = self.cloud_seg_module(feed_dict, encoder_result, pred_seg, seg_size=seg_size)
            return pred_seg, pred_cloud


class SkyCloudNetCubemap(SkyCloudNet360Base):
    """
    SkyCloudNet-CM: Standard Cubemap adaptation.
    
    Converts equirectangular input to 6 cubemap faces (90° FoV each),
    processes each face independently through SkyCloudNet, and reassembles
    the predictions back into equirectangular format.
    """
    
    def __init__(self, multilabel_module, cloud_seg_module, face_size=256):
        super().__init__(multilabel_module, cloud_seg_module)
        self.face_size = face_size
    
    def forward(self, feed_dict, seg_size=None):
        img_data = feed_dict['img_data']
        if isinstance(img_data, list):
            img_data = img_data[0]
        
        B, C, H, W = img_data.shape
        
        if seg_size is None:
            seg_size = (H, W)
        
        # Convert to cubemap
        faces, grids = equirect_to_cubemap(img_data, face_size=self.face_size, 
                                           extended=False)
        
        # Process each face
        seg_faces = []
        cloud_faces = []
        
        for face_idx in range(6):
            face_img = faces[:, face_idx]  # (B, C, face_size, face_size)
            
            face_feed = feed_dict.copy()
            face_feed['img_data'] = face_img
            
            face_seg_size = (self.face_size, self.face_size)
            
            if self.training:
                result = self.multilabel_module(face_feed, seg_size=face_seg_size)
                seg_faces.append(result[4])  # pred_seg
            else:
                encoder_result, pred = self.multilabel_module(face_feed, 
                                                              seg_size=face_seg_size)
                pred_seg = pred[1]
                seg_faces.append(pred_seg)
                
                pred_cloud = self.cloud_seg_module(face_feed, encoder_result, 
                                                   pred_seg, seg_size=face_seg_size)
                cloud_faces.append(pred_cloud)
        
        # Stack face predictions
        seg_stack = torch.stack(seg_faces, dim=1)  # (B, 6, num_class, face_size, face_size)
        
        # Reassemble to equirectangular
        pred_seg_eq = cubemap_to_equirect(seg_stack, seg_size[0], seg_size[1])
        
        if not self.training and cloud_faces:
            cloud_stack = torch.stack(cloud_faces, dim=1)
            pred_cloud_eq = cubemap_to_equirect(cloud_stack, seg_size[0], seg_size[1])
            return pred_seg_eq, pred_cloud_eq
        
        return pred_seg_eq


class SkyCloudNetExtendedCubemap(SkyCloudNet360Base):
    """
    SkyCloudNet-ECM: Extended Cubemap adaptation.
    
    Similar to the standard cubemap approach, but uses extended field-of-view
    faces (>90°) with overlap between adjacent faces. The overlapping regions
    are blended using distance-based weights to reduce seam artifacts.
    """
    
    def __init__(self, multilabel_module, cloud_seg_module, face_size=256, overlap=0.1):
        super().__init__(multilabel_module, cloud_seg_module)
        self.face_size = face_size
        self.overlap = overlap
    
    def forward(self, feed_dict, seg_size=None):
        img_data = feed_dict['img_data']
        if isinstance(img_data, list):
            img_data = img_data[0]
        
        B, C, H, W = img_data.shape
        
        if seg_size is None:
            seg_size = (H, W)
        
        # Convert to extended cubemap
        faces, grids = equirect_to_cubemap(img_data, face_size=self.face_size,
                                           extended=True, overlap=self.overlap)
        
        # Process each face
        seg_faces = []
        cloud_faces = []
        
        for face_idx in range(6):
            face_img = faces[:, face_idx]
            
            face_feed = feed_dict.copy()
            face_feed['img_data'] = face_img
            
            face_seg_size = (self.face_size, self.face_size)
            
            if self.training:
                result = self.multilabel_module(face_feed, seg_size=face_seg_size)
                seg_faces.append(result[4])
            else:
                encoder_result, pred = self.multilabel_module(face_feed, 
                                                              seg_size=face_seg_size)
                pred_seg = pred[1]
                seg_faces.append(pred_seg)
                
                pred_cloud = self.cloud_seg_module(face_feed, encoder_result, 
                                                   pred_seg, seg_size=face_seg_size)
                cloud_faces.append(pred_cloud)
        
        seg_stack = torch.stack(seg_faces, dim=1)
        
        # Reassemble with blending for extended cubemap
        pred_seg_eq = cubemap_to_equirect(seg_stack, seg_size[0], seg_size[1],
                                          extended=True, overlap=self.overlap)
        
        if not self.training and cloud_faces:
            cloud_stack = torch.stack(cloud_faces, dim=1)
            pred_cloud_eq = cubemap_to_equirect(cloud_stack, seg_size[0], seg_size[1],
                                                extended=True, overlap=self.overlap)
            return pred_seg_eq, pred_cloud_eq
        
        return pred_seg_eq


class SkyCloudNetTPP(SkyCloudNet360Base):
    """
    SkyCloudNet-TPP: Tangent Plane Projection adaptation.
    
    Samples multiple tangent planes on the sphere using gnomonic projection,
    processes each local perspective patch through SkyCloudNet, and blends
    the back-projected predictions with cosine-distance weighting.
    
    The tangent plane approach maintains consistent sampling density across
    the sphere while minimizing distortion, making it particularly effective
    for polar regions.
    """
    
    def __init__(self, multilabel_module, cloud_seg_module, 
                 patch_size=256, fov=math.pi/3, num_lat=4, num_lon=8):
        super().__init__(multilabel_module, cloud_seg_module)
        self.patch_size = patch_size
        self.fov = fov
        self.tangent_points = generate_tangent_points(num_lat=num_lat, num_lon=num_lon)
    
    def forward(self, feed_dict, seg_size=None):
        img_data = feed_dict['img_data']
        if isinstance(img_data, list):
            img_data = img_data[0]
        
        B, C, H, W = img_data.shape
        
        if seg_size is None:
            seg_size = (H, W)
        
        # Extract and process tangent plane patches
        seg_patches = []
        cloud_patches = []
        patch_centers = []
        patch_grids = []
        
        for lon, lat in self.tangent_points:
            # Extract tangent plane patch
            patch, grid = equirect_to_tangent(
                img_data, lon, lat, 
                patch_size=self.patch_size, fov=self.fov
            )
            
            patch_feed = feed_dict.copy()
            patch_feed['img_data'] = patch
            
            patch_seg_size = (self.patch_size, self.patch_size)
            
            if self.training:
                result = self.multilabel_module(patch_feed, seg_size=patch_seg_size)
                seg_patches.append(result[4])
            else:
                encoder_result, pred = self.multilabel_module(patch_feed, 
                                                              seg_size=patch_seg_size)
                pred_seg = pred[1]
                seg_patches.append(pred_seg)
                
                pred_cloud = self.cloud_seg_module(patch_feed, encoder_result,
                                                    pred_seg, seg_size=patch_seg_size)
                cloud_patches.append(pred_cloud)
            
            patch_centers.append((lon, lat))
            patch_grids.append(grid)
        
        # Back-project and blend
        pred_seg_eq = tangent_to_equirect(
            seg_patches, patch_centers, patch_grids,
            seg_size[0], seg_size[1], fov=self.fov
        )
        
        if not self.training and cloud_patches:
            pred_cloud_eq = tangent_to_equirect(
                cloud_patches, patch_centers, patch_grids,
                seg_size[0], seg_size[1], fov=self.fov
            )
            return pred_seg_eq, pred_cloud_eq
        
        return pred_seg_eq


class SkyCloudNetEQC(SkyCloudNet360Base):
    """
    SkyCloudNet-EQC: Equirectangular Convolution adaptation.
    
    Replaces standard 3x3 convolutions in the encoder and decoders with
    equirectangular-aware convolutions that adapt the horizontal kernel
    sampling based on latitude. This is the most lightweight adaptation
    as it modifies the network in-place without requiring multiple passes.
    """
    
    def __init__(self, multilabel_module, cloud_seg_module, replace_encoder=True,
                 replace_decoder=True):
        super().__init__(multilabel_module, cloud_seg_module)
        
        if replace_encoder:
            replace_conv_with_equirect(self.multilabel_module.encoder)
        
        if replace_decoder:
            replace_conv_with_equirect(self.multilabel_module.decoder_sky_seg)
            replace_conv_with_equirect(self.multilabel_module.decoder_attributes)
            replace_conv_with_equirect(self.cloud_seg_module.decoder_cloud_seg)
    
    def forward(self, feed_dict, seg_size=None):
        """Forward pass with equirectangular-aware convolutions."""
        return self.forward_base(feed_dict, seg_size=seg_size)


def build_360_model(multilabel_module, cloud_seg_module, mode='eqc', **kwargs):
    """
    Factory function to build a SkyCloudNet 360° adaptation model.
    
    Args:
        multilabel_module: base MultiLabelModule instance
        cloud_seg_module: base UnsupervisedSegmentationModule instance
        mode: adaptation type, one of:
            - 'cm': Standard cubemap (SkyCloudNet-CM)
            - 'ecm': Extended cubemap (SkyCloudNet-ECM)
            - 'tpp': Tangent plane projection (SkyCloudNet-TPP)
            - 'eqc': Equirectangular convolutions (SkyCloudNet-EQC)
            - 'none': No adaptation (baseline SkyCloudNet)
        **kwargs: additional arguments passed to the specific model constructor
        
    Returns:
        model: SkyCloudNet360 model instance
    """
    mode = mode.lower()
    
    if mode == 'cm':
        return SkyCloudNetCubemap(multilabel_module, cloud_seg_module, **kwargs)
    elif mode == 'ecm':
        return SkyCloudNetExtendedCubemap(multilabel_module, cloud_seg_module, **kwargs)
    elif mode == 'tpp':
        return SkyCloudNetTPP(multilabel_module, cloud_seg_module, **kwargs)
    elif mode == 'eqc':
        return SkyCloudNetEQC(multilabel_module, cloud_seg_module, **kwargs)
    elif mode == 'none':
        return SkyCloudNet360Base(multilabel_module, cloud_seg_module)
    else:
        raise ValueError(f"Unknown 360 adaptation mode: '{mode}'. "
                        f"Choose from: 'cm', 'ecm', 'tpp', 'eqc', 'none'")
