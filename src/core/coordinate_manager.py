#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Coordinate Manager

Handles coordinate manipulation for PDB structures.
"""

import torch
from ..models.coordinate import Coordinate
from ..utils.coordinate_utils import translate_coordinates as utils_translate_coords
from ..utils.coordinate_utils import rotate_around_axis as utils_rotate_around_axis


class CoordinateManager:
    """
    Coordinate manager for handling atom coordinate operations.
    
    Attributes:
        coordinate (Coordinate): The coordinate object to manage
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
    """
    def __init__(self, coordinate: Coordinate, device: torch.device = torch.device("cuda")):
        self.coordinate = coordinate
        self.device = device
    
    def translate_coordinates(self, translation: torch.Tensor) -> None:
        """
        Translate all atoms by a given vector.
        
        Args:
            translation (torch.Tensor): 3-element tensor containing [dx, dy, dz]
        """
        # Ensure both coordinates and translation are on the correct device
        coords = self.coordinate.coordinates.to(self.device)
        translation = translation.to(self.device)
        # Use utils function to translate coordinates
        self.coordinate.coordinates = utils_translate_coords(coords, translation)
    
    def rotate_around_axis(self, axis: torch.Tensor, angle: float, center: torch.Tensor) -> None:
        """
        Rotate structure around specified axis by given angle using PyTorch.
        
        Args:
            axis (torch.Tensor): Rotation axis
            angle (float): Rotation angle in degrees
            center (torch.Tensor): Rotation center
        """
        # Ensure all tensors are on the correct device
        coords = self.coordinate.coordinates.to(self.device)
        axis = axis.to(self.device)
        center = center.to(self.device)
        
        # Use utils function to rotate coordinates
        self.coordinate.coordinates = utils_rotate_around_axis(coords, axis, angle, center)
