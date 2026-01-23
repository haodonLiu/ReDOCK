#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate Utilities Module

Handles coordinate-related calculations for protein structures, including:
- Center of mass calculations
- Coordinate transformations
- Rotation matrices
- Translation operations
"""

import torch
from typing import List, Tuple
from ..models.coordinate import Coordinate


def calculate_geometric_center(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Calculate geometric center (average position) for a set of coordinates.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        
    Returns:
        torch.Tensor: Geometric center (shape: [3])
    """
    return torch.mean(coordinates, dim=0)


def translate_coordinates(coordinates: torch.Tensor, translation_vector: torch.Tensor) -> torch.Tensor:
    """
    Translate coordinates by a given vector.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        translation_vector (torch.Tensor): Translation vector (shape: [3])
        
    Returns:
        torch.Tensor: Translated coordinates (shape: [num_atoms, 3])
    """
    coord_obj = Coordinate()
    coord_obj.coordinates = coordinates.clone()
    coord_obj.translate(translation_vector)
    return coord_obj.coordinates


def center_coordinates(coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center coordinates at the origin.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered coordinates and translation vector
    """
    center = calculate_geometric_center(coordinates)
    centered_coords = translate_coordinates(coordinates, -center)
    return centered_coords, center


def create_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> torch.Tensor:
    """
    Create a rotation matrix from Euler angles (X, Y, Z).
    
    Args:
        angle_x (float): Rotation angle around X-axis (radians)
        angle_y (float): Rotation angle around Y-axis (radians)
        angle_z (float): Rotation angle around Z-axis (radians)
        
    Returns:
        torch.Tensor: Rotation matrix (shape: [3, 3])
    """
    # Rotation around X-axis
    rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), -torch.sin(angle_x)],
        [0, torch.sin(angle_x), torch.cos(angle_x)]
    ], dtype=torch.float32)
    
    # Rotation around Y-axis
    ry = torch.tensor([
        [torch.cos(angle_y), 0, torch.sin(angle_y)],
        [0, 1, 0],
        [-torch.sin(angle_y), 0, torch.cos(angle_y)]
    ], dtype=torch.float32)
    
    # Rotation around Z-axis
    rz = torch.tensor([
        [torch.cos(angle_z), -torch.sin(angle_z), 0],
        [torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Combined rotation matrix (Z * Y * X)
    return rz @ ry @ rx


def rotate_coordinates(coordinates: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotate coordinates using a rotation matrix.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        rotation_matrix (torch.Tensor): Rotation matrix (shape: [3, 3])
        
    Returns:
        torch.Tensor: Rotated coordinates (shape: [num_atoms, 3])
    """
    coord_obj = Coordinate()
    coord_obj.coordinates = coordinates.clone()
    # 由于原函数没有center参数，我们将其设为None，让Coordinate类自己计算中心点
    coord_obj.rotate(rotation_matrix)
    return coord_obj.coordinates


def rotate_coordinates_euler(coordinates: torch.Tensor, angle_x: float, angle_y: float, angle_z: float) -> torch.Tensor:
    """
    Rotate coordinates using Euler angles.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        angle_x (float): Rotation angle around X-axis (radians)
        angle_y (float): Rotation angle around Y-axis (radians)
        angle_z (float): Rotation angle around Z-axis (radians)
        
    Returns:
        torch.Tensor: Rotated coordinates (shape: [num_atoms, 3])
    """
    coord_obj = Coordinate()
    coord_obj.coordinates = coordinates.clone()
    # Coordinate类的rotate_euler方法接受角度制，而原函数接受弧度制，需要转换
    angle_x_deg = torch.rad2deg(torch.tensor(angle_x)).item()
    angle_y_deg = torch.rad2deg(torch.tensor(angle_y)).item()
    angle_z_deg = torch.rad2deg(torch.tensor(angle_z)).item()
    coord_obj.rotate_euler(angle_x_deg, angle_y_deg, angle_z_deg)
    return coord_obj.coordinates


def calculate_rmsd(coordinates1: torch.Tensor, coordinates2: torch.Tensor) -> float:
    """
    Calculate Root Mean Square Deviation (RMSD) between two sets of coordinates.
    
    Args:
        coordinates1 (torch.Tensor): First set of coordinates (shape: [num_atoms, 3])
        coordinates2 (torch.Tensor): Second set of coordinates (shape: [num_atoms, 3])
        
    Returns:
        float: RMSD value
    """
    if coordinates1.shape != coordinates2.shape:
        raise ValueError("Coordinate sets must have the same shape")
    
    squared_diff = torch.sum((coordinates1 - coordinates2) ** 2, dim=1)
    mean_squared_diff = torch.mean(squared_diff)
    return torch.sqrt(mean_squared_diff).item()


def kabsch_algorithm(coordinates1: torch.Tensor, coordinates2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Kabsch algorithm for optimal alignment of two sets of coordinates.
    
    Args:
        coordinates1 (torch.Tensor): First set of coordinates (shape: [num_atoms, 3])
        coordinates2 (torch.Tensor): Second set of coordinates (shape: [num_atoms, 3])
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, float]: Rotation matrix, translation vector, and RMSD
    """
    # Center both coordinate sets
    coords1_centered, center1 = center_coordinates(coordinates1)
    coords2_centered, center2 = center_coordinates(coordinates2)
    
    # Calculate covariance matrix
    covariance = torch.matmul(coords1_centered.T, coords2_centered)
    
    # Singular Value Decomposition
    u, s, vh = torch.linalg.svd(covariance)
    
    # Calculate rotation matrix
    rotation_matrix = torch.matmul(vh.T, u.T)
    
    # Ensure right-handed coordinate system
    if torch.linalg.det(rotation_matrix) < 0:
        vh[-1, :] *= -1
        rotation_matrix = torch.matmul(vh.T, u.T)
    
    # Calculate optimal translation
    translation = center2 - torch.matmul(center1, rotation_matrix.T)
    
    # Calculate RMSD
    aligned_coords1 = torch.matmul(coords1_centered, rotation_matrix.T) + center2
    rmsd = calculate_rmsd(aligned_coords1, coordinates2)
    
    return rotation_matrix, translation, rmsd


def align_coordinates(coordinates1: torch.Tensor, coordinates2: torch.Tensor) -> torch.Tensor:
    """
    Align coordinates1 to coordinates2 using Kabsch algorithm.
    
    Args:
        coordinates1 (torch.Tensor): Coordinates to align (shape: [num_atoms, 3])
        coordinates2 (torch.Tensor): Target coordinates (shape: [num_atoms, 3])
        
    Returns:
        torch.Tensor: Aligned coordinates1 (shape: [num_atoms, 3])
    """
    rotation_matrix, translation, _ = kabsch_algorithm(coordinates1, coordinates2)
    coords1_centered, _ = center_coordinates(coordinates1)
    aligned_coords = torch.matmul(coords1_centered, rotation_matrix.T) + translation
    return aligned_coords


def generate_random_rotation() -> torch.Tensor:
    """
    Generate a random rotation matrix.
    
    Returns:
        torch.Tensor: Random rotation matrix (shape: [3, 3])
    """
    # Generate random Euler angles
    angle_x = torch.rand(1).item() * 2 * torch.pi
    angle_y = torch.rand(1).item() * 2 * torch.pi
    angle_z = torch.rand(1).item() * 2 * torch.pi
    
    return create_rotation_matrix(angle_x, angle_y, angle_z)


def generate_random_translation(max_distance: float = 10.0) -> torch.Tensor:
    """
    Generate a random translation vector.
    
    Args:
        max_distance (float, optional): Maximum translation distance in each direction
        
    Returns:
        torch.Tensor: Random translation vector (shape: [3])
    """
    return (torch.rand(3) - 0.5) * 2 * max_distance


def rotate_around_axis(coordinates: torch.Tensor, axis: torch.Tensor, angle: float, center: torch.Tensor) -> torch.Tensor:
    """
    Rotate coordinates around a specified axis by a given angle.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        axis (torch.Tensor): Rotation axis (shape: [3])
        angle (float): Rotation angle in degrees
        center (torch.Tensor): Rotation center (shape: [3])
        
    Returns:
        torch.Tensor: Rotated coordinates (shape: [num_atoms, 3])
    """
    coord_obj = Coordinate()
    coord_obj.coordinates = coordinates.clone()
    coord_obj.rotate_around_axis(axis, angle, center)
    return coord_obj.coordinates
