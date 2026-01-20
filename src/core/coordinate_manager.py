#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Coordinate Manager

Handles coordinate manipulation for PDB structures.
"""

import math
import torch
from src.models.structure import Structure


class CoordinateManager:
    """
    Coordinate manager for handling atom coordinate operations.
    
    Attributes:
        structure (Structure): The structure to manage coordinates for
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
    """
    def __init__(self, structure: Structure, device: torch.device = torch.device("cuda")):
        self.structure = structure
        self.device = device
    
    def translate_coordinates(self, translation: torch.Tensor) -> None:
        """
        Translate all atoms by a given vector.
        
        Args:
            translation (torch.Tensor): 3-element tensor containing [dx, dy, dz]
        """
        # Ensure both coordinates and translation are on the correct device
        coords = self.structure.coordinates.to(self.device)
        translation = translation.to(self.device)
        # Directly update the tensor coordinates
        self.structure.coordinates = coords + translation
    
    def rotate_around_axis(self, axis: torch.Tensor, angle: float, center: torch.Tensor) -> None:
        """
        Rotate structure around specified axis by given angle using PyTorch.
        
        Args:
            axis (torch.Tensor): Rotation axis
            angle (float): Rotation angle in degrees
            center (torch.Tensor): Rotation center
        """
        # Ensure all tensors are on the correct device
        coords = self.structure.coordinates.to(self.device)
        axis = axis.to(self.device)
        center = center.to(self.device)
        
        # Normalize axis
        axis = axis / torch.norm(axis)
        
        # Convert angle to radians with higher precision
        angle_rad = torch.tensor(math.radians(angle), device=self.device, dtype=torch.float32)
        
        # Translate coordinates to origin
        coords_translated = coords - center
        
        # Calculate rotation matrix using PyTorch with higher precision
        ux, uy, uz = axis
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)
        
        # Rotation matrix components
        r11 = cos_theta + ux**2 * (1 - cos_theta)
        r12 = ux*uy*(1 - cos_theta) - uz*sin_theta
        r13 = ux*uz*(1 - cos_theta) + uy*sin_theta
        r21 = uy*ux*(1 - cos_theta) + uz*sin_theta
        r22 = cos_theta + uy**2 * (1 - cos_theta)
        r23 = uy*uz*(1 - cos_theta) - ux*sin_theta
        r31 = uz*ux*(1 - cos_theta) - uy*sin_theta
        r32 = uz*uy*(1 - cos_theta) + ux*sin_theta
        r33 = cos_theta + uz**2 * (1 - cos_theta)
        
        # Rotation matrix
        rotation_matrix = torch.stack([
            torch.stack([r11, r12, r13]),
            torch.stack([r21, r22, r23]),
            torch.stack([r31, r32, r33])
        ])
        
        # Apply rotation with higher precision
        coords_translated = coords_translated.to(torch.float32)
        rotation_matrix = rotation_matrix.to(torch.float32)
        coords_rotated = torch.matmul(coords_translated, rotation_matrix)
        
        # Translate back and convert to original dtype
        coords_final = coords_rotated + center
        coords_final = coords_final.to(self.structure.coordinates.dtype)
        
        # Directly update the tensor coordinates
        self.structure.coordinates = coords_final
