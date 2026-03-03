"""
Equirectangular image processing utilities.

Provides coordinate transformations and sampling grids for converting between
equirectangular, spherical, cubemap, and tangent-plane coordinate systems.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def create_equirect_grid(h, w, device='cpu'):
    """
    Create a grid of (longitude, latitude) for an equirectangular image.
    
    Args:
        h: image height
        w: image width
        device: torch device
        
    Returns:
        lon: (H, W) longitude in [-pi, pi]
        lat: (H, W) latitude in [-pi/2, pi/2]  (top = pi/2, bottom = -pi/2)
    """
    # Pixel centers: x in [0, w-1], y in [0, h-1]
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    x = torch.linspace(0.5, w - 0.5, w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Map to spherical coordinates
    lon = (grid_x / w) * 2 * math.pi - math.pi       # [-pi, pi]
    lat = math.pi / 2 - (grid_y / h) * math.pi        # [pi/2, -pi/2]
    
    return lon, lat


def lonlat_to_xyz(lon, lat):
    """Convert (longitude, latitude) to unit sphere (x, y, z) coordinates."""
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return x, y, z


def xyz_to_lonlat(x, y, z):
    """Convert unit sphere (x, y, z) to (longitude, latitude)."""
    lon = torch.atan2(y, x)
    lat = torch.asin(torch.clamp(z, -1.0, 1.0))
    return lon, lat


def lonlat_to_equirect_uv(lon, lat, h, w):
    """
    Convert (longitude, latitude) to equirectangular pixel coordinates.
    
    Returns:
        u: horizontal pixel coordinate in [0, w)
        v: vertical pixel coordinate in [0, h)
    """
    u = (lon + math.pi) / (2 * math.pi) * w
    v = (math.pi / 2 - lat) / math.pi * h
    return u, v


def equirect_uv_to_grid(u, v, h, w):
    """
    Convert equirectangular pixel coordinates (u, v) to normalized grid
    coordinates for F.grid_sample (range [-1, 1]).
    """
    grid_x = 2.0 * u / w - 1.0
    grid_y = 2.0 * v / h - 1.0
    return grid_x, grid_y


def compute_latitude_scale(h, w, device='cpu'):
    """
    Compute horizontal scale factor for each row of an equirectangular image.
    At latitude theta, the horizontal scale is 1/cos(theta).
    
    Args:
        h: image height
        w: image width
        device: torch device
        
    Returns:
        scale: (H,) tensor of horizontal scale factors per row
    """
    y = torch.linspace(0.5, h - 0.5, h, device=device)
    lat = math.pi / 2 - (y / h) * math.pi  # [pi/2, -pi/2]
    # Clamp to avoid division by zero at poles
    cos_lat = torch.cos(lat).clamp(min=1e-6)
    scale = 1.0 / cos_lat
    return scale


# ============================================================
# Cubemap utilities
# ============================================================

def equirect_to_cubemap(equirect, face_size=256, extended=False, overlap=0.1):
    """
    Convert an equirectangular image to cubemap faces.
    
    Args:
        equirect: (B, C, H, W) equirectangular image tensor
        face_size: output face resolution
        extended: if True, use extended FoV (>90°) to capture overlap
        overlap: fractional overlap for extended cubemap (e.g., 0.1 = 10%)
        
    Returns:
        faces: (B, 6, C, face_size, face_size) cubemap faces
               Order: front, right, back, left, top, bottom
        grids: list of 6 sampling grids for inverse mapping
    """
    B, C, H, W = equirect.shape
    device = equirect.device
    
    fov_half = math.pi / 4  # 90° FoV => half-angle = 45°
    if extended:
        fov_half = math.pi / 4 * (1.0 + overlap)
    
    # Face rotations: (forward_dir, up_dir) for each face
    # front(+x), right(+y), back(-x), left(-y), top(+z), bottom(-z)
    face_rotations = [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),   # front
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),   # right
        (torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),  # back
        (torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),  # left
        (torch.tensor([0.0, 0.0, 1.0]), torch.tensor([-1.0, 0.0, 0.0])),  # top
        (torch.tensor([0.0, 0.0, -1.0]), torch.tensor([1.0, 0.0, 0.0])),  # bottom
    ]
    
    faces = []
    grids = []
    
    for fwd, up in face_rotations:
        fwd = fwd.to(device).float()
        up = up.to(device).float()
        right = torch.cross(fwd, up)
        right = right / right.norm()
        up = torch.cross(right, fwd)
        up = up / up.norm()
        
        # Create sampling grid on the face
        tan_fov = math.tan(fov_half)
        u = torch.linspace(-tan_fov, tan_fov, face_size, device=device)
        v = torch.linspace(-tan_fov, tan_fov, face_size, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
        
        # 3D direction for each pixel on the face
        # direction = fwd + grid_u * right + grid_v * up  (but v is flipped)
        dirs = (fwd.unsqueeze(0).unsqueeze(0) 
                + grid_u.unsqueeze(-1) * right.unsqueeze(0).unsqueeze(0)
                - grid_v.unsqueeze(-1) * up.unsqueeze(0).unsqueeze(0))
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        
        # Convert to lon/lat
        lon = torch.atan2(dirs[..., 1], dirs[..., 0])
        lat = torch.asin(dirs[..., 2].clamp(-1, 1))
        
        # Convert to equirectangular pixel coordinates  
        sample_u, sample_v = lonlat_to_equirect_uv(lon, lat, H, W)
        grid_x, grid_y = equirect_uv_to_grid(sample_u, sample_v, H, W)
        
        # Handle wrap-around for equirectangular
        grid_x = grid_x.fmod(2.0)  # periodic wrapping
        grid_x = torch.where(grid_x > 1.0, grid_x - 2.0, grid_x)
        grid_x = torch.where(grid_x < -1.0, grid_x + 2.0, grid_x)
        
        sample_grid = torch.stack([grid_x, grid_y], dim=-1)  # (face_size, face_size, 2)
        sample_grid = sample_grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        face = F.grid_sample(equirect, sample_grid, mode='bilinear', 
                            padding_mode='border', align_corners=False)
        faces.append(face)
        grids.append(sample_grid[0])  # Store one copy of the grid
    
    faces = torch.stack(faces, dim=1)  # (B, 6, C, face_size, face_size)
    return faces, grids


def cubemap_to_equirect(faces, h, w, grids=None, extended=False, overlap=0.1):
    """
    Convert cubemap faces back to equirectangular format.
    
    Args:
        faces: (B, 6, C, face_size, face_size) cubemap face predictions
        h: output equirectangular height
        w: output equirectangular width
        grids: precomputed sampling grids (optional)
        extended: if True, blend overlapping regions
        overlap: overlap fraction for extended cubemap
        
    Returns:
        equirect: (B, C, H, W) equirectangular image
    """
    B, _, C, face_size, _ = faces.shape
    device = faces.device
    
    # Create equirect grid
    lon, lat = create_equirect_grid(h, w, device=device)
    x, y, z = lonlat_to_xyz(lon, lat)
    
    equirect = torch.zeros(B, C, h, w, device=device)
    weights = torch.zeros(B, 1, h, w, device=device)
    
    fov_half = math.pi / 4
    if extended:
        fov_half = math.pi / 4 * (1.0 + overlap)
    tan_fov = math.tan(fov_half)
    
    face_rotations = [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([0.0, 0.0, 1.0]), torch.tensor([-1.0, 0.0, 0.0])),
        (torch.tensor([0.0, 0.0, -1.0]), torch.tensor([1.0, 0.0, 0.0])),
    ]
    
    xyz = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    
    for face_idx, (fwd, up) in enumerate(face_rotations):
        fwd = fwd.to(device).float()
        up = up.to(device).float()
        right = torch.cross(fwd, up)
        right = right / right.norm()
        up = torch.cross(right, fwd)
        up = up / up.norm()
        
        # Project equirect points onto this face
        # dot with forward direction
        d = (xyz * fwd.unsqueeze(0).unsqueeze(0)).sum(-1)  # (H, W)
        
        # Only consider points in front of the face
        valid = d > 1e-6
        
        # Project onto face plane
        proj_u = (xyz * right.unsqueeze(0).unsqueeze(0)).sum(-1) / d.clamp(min=1e-6)
        proj_v = -(xyz * up.unsqueeze(0).unsqueeze(0)).sum(-1) / d.clamp(min=1e-6)
        
        # Check if within face bounds
        valid = valid & (proj_u.abs() <= tan_fov) & (proj_v.abs() <= tan_fov)
        
        # Convert to sampling coordinates for the face
        grid_x = proj_u / tan_fov  # [-1, 1]
        grid_y = proj_v / tan_fov  # [-1, 1]
        
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        face_data = faces[:, face_idx]  # (B, C, face_size, face_size)
        sampled = F.grid_sample(face_data, sample_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        
        # Compute blending weight (higher in center, lower at edges)
        if extended:
            w_u = 1.0 - (proj_u.abs() / tan_fov).clamp(0, 1)
            w_v = 1.0 - (proj_v.abs() / tan_fov).clamp(0, 1)
            blend_weight = (w_u * w_v).unsqueeze(0).unsqueeze(0)
        else:
            blend_weight = torch.ones(1, 1, h, w, device=device)
        
        valid_mask = valid.float().unsqueeze(0).unsqueeze(0)
        equirect += sampled * valid_mask * blend_weight
        weights += valid_mask * blend_weight
    
    equirect = equirect / weights.clamp(min=1e-6)
    return equirect


# ============================================================
# Tangent Plane Projection utilities
# ============================================================

def generate_tangent_points(num_lat=4, num_lon=8, include_poles=True):
    """
    Generate sampling points on the sphere for tangent plane projections.
    
    Args:
        num_lat: number of latitude bands
        num_lon: number of longitude samples per band
        include_poles: whether to include pole points
        
    Returns:
        points: list of (lon, lat) tuples in radians
    """
    points = []
    
    if include_poles:
        points.append((0.0, math.pi / 2))   # north pole
        points.append((0.0, -math.pi / 2))  # south pole
    
    for i in range(num_lat):
        lat = math.pi / 2 - (i + 1) * math.pi / (num_lat + 1)
        for j in range(num_lon):
            lon = -math.pi + j * 2 * math.pi / num_lon
            points.append((lon, lat))
    
    return points


def equirect_to_tangent(equirect, center_lon, center_lat, patch_size=256, fov=math.pi/3):
    """
    Extract a tangent plane (gnomonic) projection from an equirectangular image.
    
    Args:
        equirect: (B, C, H, W) equirectangular image
        center_lon: center longitude of tangent point in radians
        center_lat: center latitude of tangent point in radians
        patch_size: output patch size in pixels
        fov: field of view of the tangent plane in radians
        
    Returns:
        patch: (B, C, patch_size, patch_size) tangent plane image
        grid: sampling grid for back-projection
    """
    B, C, H, W = equirect.shape
    device = equirect.device
    
    # Create grid on tangent plane
    half_fov = fov / 2
    tan_half = math.tan(half_fov)
    
    u = torch.linspace(-tan_half, tan_half, patch_size, device=device)
    v = torch.linspace(-tan_half, tan_half, patch_size, device=device)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    
    # Convert tangent plane coordinates to 3D directions (gnomonic projection)
    # Using rotation around center_lon, center_lat
    cos_lat = math.cos(center_lat)
    sin_lat = math.sin(center_lat)
    cos_lon = math.cos(center_lon)
    sin_lon = math.sin(center_lon)
    
    # Local coordinate system at tangent point:
    # forward = radial direction at (center_lon, center_lat)
    # right = east direction
    # up = north direction
    
    # 3D direction in local frame: (1, u, -v) normalized
    rho = torch.sqrt(1.0 + grid_u ** 2 + grid_v ** 2)
    
    # Spherical coordinates relative to tangent point
    c = torch.atan(torch.sqrt(grid_u ** 2 + grid_v ** 2))
    cos_c = torch.cos(c)
    sin_c = torch.sin(c)
    
    # Convert to global lat/lon using gnomonic inverse
    lat = torch.asin(cos_c * sin_lat + (-grid_v * sin_c * cos_lat) / 
                     torch.sqrt(grid_u ** 2 + grid_v ** 2).clamp(min=1e-8))
    
    # Handle the case where grid_u and grid_v are both 0 (center point)
    center_mask = (grid_u.abs() < 1e-8) & (grid_v.abs() < 1e-8)
    
    lon = center_lon + torch.atan2(
        grid_u * sin_c,
        torch.sqrt(grid_u ** 2 + grid_v ** 2).clamp(min=1e-8) * cos_lat * cos_c 
        - (-grid_v) * sin_lat * sin_c
    )
    
    lat = torch.where(center_mask, torch.tensor(center_lat, device=device), lat)
    lon = torch.where(center_mask, torch.tensor(center_lon, device=device), lon)
    
    # Wrap longitude to [-pi, pi]
    lon = ((lon + math.pi) % (2 * math.pi)) - math.pi
    
    # Convert to equirectangular pixel coordinates
    sample_u, sample_v = lonlat_to_equirect_uv(lon, lat, H, W)
    grid_x, grid_y = equirect_uv_to_grid(sample_u, sample_v, H, W)
    
    # Clamp grid_y, wrap grid_x
    grid_y = grid_y.clamp(-1, 1)
    
    sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    
    patch = F.grid_sample(equirect, sample_grid, mode='bilinear',
                         padding_mode='border', align_corners=False)
    
    return patch, sample_grid[0]


def tangent_to_equirect(patches, centers, patch_grids, h, w, fov=math.pi/3):
    """
    Back-project tangent plane patches to equirectangular format and blend.
    
    Args:
        patches: list of (B, C, patch_size, patch_size) tangent plane predictions
        centers: list of (lon, lat) center points
        patch_grids: list of sampling grids from equirect_to_tangent
        h: output equirectangular height
        w: output equirectangular width
        fov: field of view used for the tangent planes
        
    Returns:
        equirect: (B, C, H, W) blended equirectangular prediction
    """
    B, C = patches[0].shape[0], patches[0].shape[1]
    device = patches[0].device
    patch_size = patches[0].shape[2]
    
    equirect = torch.zeros(B, C, h, w, device=device)
    weights = torch.zeros(B, 1, h, w, device=device)
    
    for patch, (center_lon, center_lat), grid in zip(patches, centers, patch_grids):
        # grid is (patch_size, patch_size, 2) in normalized [-1,1] coordinates
        # We need the inverse: for each equirect pixel, find its tangent plane coordinate
        
        # Create equirect grid
        lon_eq, lat_eq = create_equirect_grid(h, w, device=device)
        
        cos_lat0 = math.cos(center_lat)
        sin_lat0 = math.sin(center_lat)
        
        # Compute angular distance from tangent center
        delta_lon = lon_eq - center_lon
        # Wrap
        delta_lon = ((delta_lon + math.pi) % (2 * math.pi)) - math.pi
        
        cos_c = (sin_lat0 * torch.sin(lat_eq) + 
                 cos_lat0 * torch.cos(lat_eq) * torch.cos(delta_lon))
        
        # Only points in front of the tangent plane
        valid = cos_c > 0
        
        # Gnomonic forward projection
        kk = 1.0 / cos_c.clamp(min=1e-8)
        proj_u = kk * torch.cos(lat_eq) * torch.sin(delta_lon)
        proj_v = kk * (cos_lat0 * torch.sin(lat_eq) - 
                       sin_lat0 * torch.cos(lat_eq) * torch.cos(delta_lon))
        
        half_fov = fov / 2
        tan_half = math.tan(half_fov)
        
        # Check if within tangent plane bounds
        valid = valid & (proj_u.abs() <= tan_half) & (proj_v.abs() <= tan_half)
        
        # Normalize to [-1, 1] for grid_sample on the patch
        grid_x = proj_u / tan_half
        grid_y = -proj_v / tan_half  # flip v
        
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        sampled = F.grid_sample(patch, sample_grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False)
        
        # Distance-based blending weight (cosine weighting)
        blend_weight = cos_c.clamp(min=0).unsqueeze(0).unsqueeze(0)
        valid_mask = valid.float().unsqueeze(0).unsqueeze(0)
        
        equirect += sampled * valid_mask * blend_weight
        weights += valid_mask * blend_weight
    
    equirect = equirect / weights.clamp(min=1e-6)
    return equirect
