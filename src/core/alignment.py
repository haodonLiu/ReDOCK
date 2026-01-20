#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Alignment Module

Implements a simplified protein alignment algorithm for receptor-ligand docking.
"""

import torch
from typing import List
from src.models.structure import Structure
from src.core.coordinate_manager import CoordinateManager


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector to unit length.
    
    Args:
        vec (torch.Tensor): Input vector
        
    Returns:
        torch.Tensor: Normalized vector
    """
    norm = torch.norm(vec)
    if norm < 1e-6:
        raise ValueError("Vector has zero length.")
    return vec / norm

def calculate_rotation_parameters(current_vec: torch.Tensor, desired_vec: torch.Tensor, device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> tuple:
    """
    Calculate rotation axis and angle to align current_vec to desired_vec.
    
    Args:
        current_vec (torch.Tensor): Current vector
        desired_vec (torch.Tensor): Desired vector
        
    Returns:
        tuple: (rotation_axis, angle_degrees). Returns (None, 0) if vectors are colinear.
    """
    # Normalize vectors
    current_norm = normalize_vector(current_vec).to(device)
    desired_norm = normalize_vector(desired_vec).to(device)
    
    # Calculate cross product for rotation axis
    cross = torch.cross(current_norm, desired_norm, dim=0)
    cross_norm = torch.norm(cross)
    
    # Check if vectors are colinear
    if cross_norm < 1e-6:
        return None, 0.0
    
    # Normalize rotation axis
    rotation_axis = cross / cross_norm
    
    # Calculate angle in degrees
    dot = torch.dot(current_norm, desired_norm)
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp to avoid numerical issues
    angle_rad = torch.acos(dot)
    angle_deg = angle_rad * 180.0 / torch.pi
    
    return rotation_axis, angle_deg.item()


