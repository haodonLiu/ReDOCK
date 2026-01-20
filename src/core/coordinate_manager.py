#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Coordinate Manager

Handles coordinate manipulation for PDB structures.
"""

import math
import torch
from typing import Tuple
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
        
        # Convert angle to radians directly from float
        angle_rad = torch.tensor(math.radians(angle), device=self.device, dtype=torch.float16)
        
        # Translate coordinates to origin
        coords_translated = coords - center
        
        # Calculate rotation matrix using PyTorch
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
        
        # Apply rotation
        coords_translated = coords_translated.to(torch.float16)
        rotation_matrix = rotation_matrix.to(torch.float16)
        coords_rotated = torch.matmul(coords_translated, rotation_matrix)
        
        # Translate back
        coords_final = coords_rotated + center
        
        # Directly update the tensor coordinates
        self.structure.coordinates = coords_final
    
    def apply_random_perturbation(self, translation_magnitude: float = 0.5, rotation_angle: float = 5.0, rotation_axis: torch.Tensor = None) -> None:
        """
        Apply random perturbation to the structure:
        1. Random translation in X and Y axes (Z axis remains 0)
        2. Random rotation around specified axis
        
        Args:
            translation_magnitude (float): Maximum translation magnitude in Å (default: 0.5)
            rotation_angle (float): Maximum rotation angle in degrees (default: 5.0)
            rotation_axis (torch.Tensor): Rotation axis (default: Z-axis)
        """
        # Set default rotation axis if not provided
        if rotation_axis is None:
            rotation_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        
        # Calculate rotation center (geometric center of structure)
        center = torch.mean(self.structure.coordinates, dim=0)
        
        # 1. Apply random translation in X and Y axes (Z remains 0)
        random_translation = torch.tensor([
            (torch.rand(1, device=self.device) - 0.5) * 2 * translation_magnitude,
            (torch.rand(1, device=self.device) - 0.5) * 2 * translation_magnitude,
            0
        ], device=self.device)
        self.translate_coordinates(random_translation)
        
        # 2. Apply random rotation
        random_angle = (torch.rand(1, device=self.device) - 0.5) * 2 * rotation_angle
        self.rotate_around_axis(rotation_axis, random_angle.item(), center)
