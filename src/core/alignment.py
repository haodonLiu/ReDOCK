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


def calculate_protein_center(structure: Structure) -> torch.Tensor:
    """
    Calculate the geometric center of a protein based on atomic positions.
    
    Args:
        structure (Structure): Protein structure
        
    Returns:
        torch.Tensor: Center coordinates as a tensor [x, y, z]
    """
    if structure.coordinates.shape[0] == 0:
        raise ValueError("Structure has no atoms.")
    return torch.mean(structure.coordinates, dim=0)


def calculate_group_center(structure: Structure, atom_indices: List[int]) -> torch.Tensor:
    """
    Calculate the center of a specific target group based on atomic positions.
    
    Args:
        structure (Structure): Protein structure
        atom_indices (List[int]): Indices of atoms in the target group
        
    Returns:
        torch.Tensor: Group center coordinates as a tensor [x, y, z]
    """
    if not atom_indices:
        raise ValueError("Atom indices list is empty.")
    coords = structure.coordinates[atom_indices]
    return torch.mean(coords, dim=0)


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


def calculate_rotation_parameters(current_vec: torch.Tensor, desired_vec: torch.Tensor) -> tuple:
    """
    Calculate rotation axis and angle to align current_vec to desired_vec.
    
    Args:
        current_vec (torch.Tensor): Current vector
        desired_vec (torch.Tensor): Desired vector
        
    Returns:
        tuple: (rotation_axis, angle_degrees). Returns (None, 0) if vectors are colinear.
    """
    # Normalize vectors
    current_norm = normalize_vector(current_vec)
    desired_norm = normalize_vector(desired_vec)
    
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


def _align_protein(structure: Structure, target_indices: List[int], direction: str) -> None:
    """
    Generic protein alignment function.
    
    Args:
        structure (Structure): Protein structure
        target_indices (List[int]): Indices of atoms in the target group
        direction (str): 'positive' for positive z-axis, 'negative' for negative z-axis
    """
    coord_manager = CoordinateManager(structure)
    
    # Step 1: Translate center to origin
    center = calculate_protein_center(structure)
    translation = -center
    coord_manager.translate_coordinates(translation)
    
    # Step 2: Calculate vector from center to target group
    target = calculate_group_center(structure, target_indices)
    current_vec = target  # Since center is now at origin
    
    # Step 3: Rotate to align with desired z-axis direction
    if direction == 'positive':
        desired_vec = torch.tensor([0.0, 0.0, 1.0], device=current_vec.device, dtype=current_vec.dtype)
    else:
        desired_vec = torch.tensor([0.0, 0.0, -1.0], device=current_vec.device, dtype=current_vec.dtype)
    
    rotation_axis, angle = calculate_rotation_parameters(current_vec, desired_vec)
    
    if rotation_axis is not None and angle > 1e-6:
        coord_manager.rotate_around_axis(rotation_axis, angle, torch.zeros(3, device=rotation_axis.device, dtype=current_vec.dtype))

def align_proteins(receptor: Structure, receptor_target_indices: List[int], 
                   ligand: Structure, ligand_target_indices: List[int]) -> None:
    """
    Align both receptor and ligand proteins according to specifications.
    
    Args:
        receptor (Structure): Receptor protein structure
        receptor_target_indices (List[int]): Indices of atoms in receptor target group
        ligand (Structure): Ligand protein structure
        ligand_target_indices (List[int]): Indices of atoms in ligand target group
    """
    # Align receptor first
    _align_protein(receptor, receptor_target_indices, 'positive')
    # Then align ligand
    _align_protein(ligand, ligand_target_indices, 'negative')
    
    # After alignment, both proteins are at (0, 0, 0)
    # We need to separate them along the z-axis to avoid immediate atom collisions


