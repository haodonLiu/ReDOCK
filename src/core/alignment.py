#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Alignment Module

Implements a simplified protein alignment algorithm for receptor-ligand docking.
"""

import torch
from typing import List, Dict
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
    cross = torch.cross(current_norm, desired_norm)
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


def align_receptor(receptor: Structure, receptor_target_indices: List[int]) -> None:
    """
    Align receptor protein:
    1. Translate to place center at origin
    2. Rotate to align target group center with positive z-axis
    
    Args:
        receptor (Structure): Receptor protein structure
        receptor_target_indices (List[int]): Indices of atoms in receptor target group
    """
    # Step 1: Translate receptor center to origin
    rec_center = calculate_protein_center(receptor)
    translation = -rec_center  # Vector to move center to origin
    coord_manager = CoordinateManager(receptor)
    coord_manager.translate_coordinates(translation)
    
    # Step 2: Calculate target group center
    rec_target = calculate_group_center(receptor, receptor_target_indices)
    
    # Step 3: Rotate to align target group with positive z-axis
    current_vec = rec_target
    desired_vec = torch.tensor([0.0, 0.0, 1.0], device=current_vec.device)
    
    rotation_axis, angle = calculate_rotation_parameters(current_vec, desired_vec)
    
    if rotation_axis is not None and angle > 1e-6:
        coord_manager.rotate_around_axis(rotation_axis, angle, torch.zeros(3, device=rotation_axis.device))


def align_ligand(ligand: Structure, ligand_target_indices: List[int]) -> None:
    """
    Align ligand protein:
    1. Rotate to align center-to-target-group vector with negative z-axis
    2. This ensures ligand target group faces toward receptor target group
    
    Args:
        ligand (Structure): Ligand protein structure
        ligand_target_indices (List[int]): Indices of atoms in ligand target group
    """
    # Step 1: Calculate ligand center and target group center
    lig_center = calculate_protein_center(ligand)
    lig_target = calculate_group_center(ligand, ligand_target_indices)
    
    # Step 2: Calculate vector from center to target group
    current_vec = lig_target - lig_center
    
    # Step 3: Rotate to align vector with negative z-axis
    desired_vec = torch.tensor([0.0, 0.0, -1.0], device=current_vec.device)
    
    rotation_axis, angle = calculate_rotation_parameters(current_vec, desired_vec)
    
    if rotation_axis is not None and angle > 1e-6:
        coord_manager = CoordinateManager(ligand)
        coord_manager.rotate_around_axis(rotation_axis, angle, lig_center)


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
    align_receptor(receptor, receptor_target_indices)
    
    # Then align ligand
    align_ligand(ligand, ligand_target_indices)


def validate_alignment(receptor: Structure, receptor_target_indices: List[int], 
                       ligand: Structure, ligand_target_indices: List[int]) -> Dict[str, bool]:
    """
    Validate alignment accuracy against success criteria.
    
    Args:
        receptor (Structure): Aligned receptor protein
        receptor_target_indices (List[int]): Receptor target group indices
        ligand (Structure): Aligned ligand protein
        ligand_target_indices (List[int]): Ligand target group indices
        
    Returns:
        Dict[str, bool]: Validation results for each success criterion
    """
    # Check receptor center at origin
    rec_center = calculate_protein_center(receptor)
    rec_center_at_origin = torch.allclose(rec_center, torch.zeros(3), atol=1e-3)
    
    # Check receptor target group on positive z-axis
    rec_target = calculate_group_center(receptor, receptor_target_indices)
    rec_target_on_z = torch.allclose(rec_target[:2], torch.zeros(2), atol=1e-3) and rec_target[2] > 0
    
    # Check ligand vector colinear with z-axis
    lig_center = calculate_protein_center(ligand)
    lig_target = calculate_group_center(ligand, ligand_target_indices)
    lig_vec = lig_target - lig_center
    lig_vec_colinear = torch.allclose(lig_vec[:2], torch.zeros(2), atol=1e-3)
    
    # Check ligand target faces receptor target
    # Negative z-component means vector points toward positive z (receptor target direction)
    lig_target_faces_receptor = lig_vec[2] < 0
    
    return {
        "receptor_center_at_origin": rec_center_at_origin,
        "receptor_target_on_positive_z": rec_target_on_z,
        "ligand_vector_colinear_with_z": lig_vec_colinear,
        "ligand_target_faces_receptor": lig_target_faces_receptor,
        "all_criteria_met": all([rec_center_at_origin, rec_target_on_z, 
                               lig_vec_colinear, lig_target_faces_receptor])
    }
