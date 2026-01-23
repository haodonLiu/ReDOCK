#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate Data Model

Defines the Coordinate class for storing and manipulating 3D coordinates.
"""

import torch
from typing import List, Tuple, Optional
import math


class Coordinate:
    """
    Container class for storing and manipulating 3D coordinates.
    
    Attributes:
        coordinates (torch.Tensor): Tensor of 3D coordinates (shape: [N, 3])
    """
    def __init__(self):
        self.coordinates: torch.Tensor = torch.empty(0, 3, dtype=torch.float16)  # Store 3D coordinates as tensor

    def clear(self) -> None:
        """
        Clear all coordinates.
        """
        self.coordinates = torch.empty(0, 3, dtype=torch.float16)

    def get_point_count(self) -> int:
        """
        Get the total number of points in the coordinate set.
        
        Returns:
            int: Total number of points
        """
        return self.coordinates.shape[0]

    def get_coordinates_by_index(self, index: int) -> Tuple[float, float, float]:
        """
        Get coordinates by index.
        
        Args:
            index (int): Index of the point
            
        Returns:
            Tuple[float, float, float]: Coordinates of the point
        """
        coord = self.coordinates[index]
        return (float(coord[0]), float(coord[1]), float(coord[2]))

    def calculate_geometric_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the coordinate set.
        
        Returns:
            torch.Tensor: Geometric center coordinates as a tensor [x, y, z]
        """
        if self.coordinates.shape[0] == 0:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
        return torch.mean(self.coordinates, dim=0)

    def align_to_standard_coordinate_system(self) -> None:
        """
        Align the coordinate set to the standard coordinate system:
        1. Translate to center at origin
        """
        if self.coordinates.shape[0] == 0:
            return
        
        # Translate to origin
        center = self.calculate_geometric_center()
        self.translate(-center)

    def copy(self) -> 'Coordinate':
        """
        Create a deep copy of the coordinate set.
        
        Returns:
            Coordinate: Deep copy of the coordinate set
        """
        # Create a new coordinate set
        copied = Coordinate()
        
        # Copy coordinates
        copied.coordinates = self.coordinates.detach().clone()
        
        return copied

    def translate(self, translation: torch.Tensor) -> None:
        """
        Translate all coordinates by a given vector.
        
        Args:
            translation (torch.Tensor): Translation vector (shape: [3])
        """
        self.coordinates += translation

    def rotate(self, rotation_matrix: torch.Tensor, center: Optional[torch.Tensor] = None) -> None:
        """
        Rotate all coordinates by a given rotation matrix around a center point.
        
        Args:
            rotation_matrix (torch.Tensor): Rotation matrix (shape: [3, 3])
            center (torch.Tensor, optional): Center point for rotation (shape: [3])
        """
        if center is None:
            center = self.calculate_geometric_center()
        
        # Translate to origin
        self.coordinates -= center
        
        # Apply rotation
        self.coordinates = torch.matmul(self.coordinates, rotation_matrix.T)
        
        # Translate back
        self.coordinates += center

    def rotate_around_axis(self, axis: torch.Tensor, angle: float, center: Optional[torch.Tensor] = None) -> None:
        """
        Rotate coordinates around a specified axis by a given angle.
        
        Args:
            axis (torch.Tensor): Rotation axis (shape: [3])
            angle (float): Rotation angle in degrees
            center (torch.Tensor, optional): Rotation center (shape: [3])
        """
        if center is None:
            center = self.calculate_geometric_center()
        
        # Normalize axis
        axis = axis / torch.norm(axis)
        
        # Convert angle to radians
        angle_rad = torch.tensor(math.radians(angle), dtype=torch.float32, device=self.coordinates.device)
        
        # Translate coordinates to origin
        self.coordinates -= center
        
        # Calculate rotation matrix components
        ux, uy, uz = axis
        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)
        
        # Rotation matrix
        r11 = cos_theta + ux**2 * (1 - cos_theta)
        r12 = ux*uy*(1 - cos_theta) - uz*sin_theta
        r13 = ux*uz*(1 - cos_theta) + uy*sin_theta
        r21 = uy*ux*(1 - cos_theta) + uz*sin_theta
        r22 = cos_theta + uy**2 * (1 - cos_theta)
        r23 = uy*uz*(1 - cos_theta) - ux*sin_theta
        r31 = uz*ux*(1 - cos_theta) - uy*sin_theta
        r32 = uz*uy*(1 - cos_theta) + ux*sin_theta
        r33 = cos_theta + uz**2 * (1 - cos_theta)
        
        rotation_matrix = torch.stack([
            torch.stack([r11, r12, r13]),
            torch.stack([r21, r22, r23]),
            torch.stack([r31, r32, r33])
        ])
        
        # Apply rotation
        self.coordinates = self.coordinates.to(torch.float32)
        rotation_matrix = rotation_matrix.to(torch.float32)
        self.coordinates = torch.matmul(self.coordinates, rotation_matrix)
        
        # Translate back and convert to original dtype
        self.coordinates = self.coordinates + center
        self.coordinates = self.coordinates.to(self.coordinates.dtype)

    def rotate_euler(self, angle_x: float, angle_y: float, angle_z: float, center: Optional[torch.Tensor] = None) -> None:
        """
        Rotate coordinates using Euler angles around a center point.
        
        Args:
            angle_x (float): Rotation angle around X-axis in degrees
            angle_y (float): Rotation angle around Y-axis in degrees
            angle_z (float): Rotation angle around Z-axis in degrees
            center (torch.Tensor, optional): Center point for rotation (shape: [3])
        """
        if center is None:
            center = self.calculate_geometric_center()
        
        # Convert angles to radians
        angle_x_rad = math.radians(angle_x)
        angle_y_rad = math.radians(angle_y)
        angle_z_rad = math.radians(angle_z)
        
        # Create rotation matrices for each axis
        # Rotation around X-axis
        rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(angle_x_rad), -math.sin(angle_x_rad)],
            [0, math.sin(angle_x_rad), math.cos(angle_x_rad)]
        ], dtype=torch.float32, device=self.coordinates.device)
        
        # Rotation around Y-axis
        ry = torch.tensor([
            [math.cos(angle_y_rad), 0, math.sin(angle_y_rad)],
            [0, 1, 0],
            [-math.sin(angle_y_rad), 0, math.cos(angle_y_rad)]
        ], dtype=torch.float32, device=self.coordinates.device)
        
        # Rotation around Z-axis
        rz = torch.tensor([
            [math.cos(angle_z_rad), -math.sin(angle_z_rad), 0],
            [math.sin(angle_z_rad), math.cos(angle_z_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.coordinates.device)
        
        # Combined rotation matrix (Z * Y * X)
        rotation_matrix = rz @ ry @ rx
        
        # Apply rotation
        self.rotate(rotation_matrix, center)

    def __repr__(self) -> str:
        """String representation of the coordinate set"""
        return f"Coordinate(points={self.get_point_count()}, coordinates_shape={tuple(self.coordinates.shape)})"
