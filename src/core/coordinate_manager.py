#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Coordinate Manager

Handles coordinate manipulation for PDB structures.
"""

import torch
from ..models.coordinate import Coordinate


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
        # Ensure translation is on the correct device
        translation = translation.to(self.device)
        # Directly call Coordinate class's translate method
        self.coordinate.translate(translation)
    
    def rotate_around_axis(self, axis: torch.Tensor, angle: float, center: torch.Tensor) -> None:
        """
        Rotate structure around specified axis by given angle using PyTorch.
        
        Args:
            axis (torch.Tensor): Rotation axis
            angle (float): Rotation angle in degrees
            center (torch.Tensor): Rotation center
        """
        # Ensure all tensors are on the correct device
        axis = axis.to(self.device)
        center = center.to(self.device)
        
        # Directly call Coordinate class's rotate_around_axis method
        self.coordinate.rotate_around_axis(axis, angle, center)
