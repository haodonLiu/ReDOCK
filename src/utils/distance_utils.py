#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance Utilities Module

Handles distance-related calculations for protein structures, including:
- Atom pair distance calculations
- Distance matrix calculations
- Minimum distance detection
- Distance-based filtering
"""

import torch
from typing import List, Tuple, Optional


def calculate_atom_pair_distances(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """
    Calculate distances between all atom pairs from two sets of coordinates.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        
    Returns:
        torch.Tensor: Distance matrix between atom pairs (shape: [num_atoms1, num_atoms2])
    """
    expanded1 = coords1.unsqueeze(1)  # (N, 1, 3)
    expanded2 = coords2.unsqueeze(0)  # (1, M, 3)
    return torch.norm(expanded1 - expanded2, dim=2)  # (N, M)


def calculate_distance_vectors(coords1: torch.Tensor, coords2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate distance vectors and distances between all atom pairs.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Distance vectors (shape: [num_atoms1, num_atoms2, 3]) and distances (shape: [num_atoms1, num_atoms2])
    """
    expanded1 = coords1.unsqueeze(1)  # (N, 1, 3)
    expanded2 = coords2.unsqueeze(0)  # (1, M, 3)
    distance_vectors = expanded2 - expanded1  # (N, M, 3)
    distances = torch.norm(distance_vectors, dim=2)  # (N, M)
    return distance_vectors, distances


def find_minimum_distance(coords1: torch.Tensor, coords2: torch.Tensor) -> float:
    """
    Find the minimum distance between any pair of atoms from two sets.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        
    Returns:
        float: Minimum distance between any atom pair
    """
    distances = calculate_atom_pair_distances(coords1, coords2)
    return torch.min(distances).item()


def find_all_distances_below_cutoff(coords1: torch.Tensor, coords2: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Find all atom pairs with distances below a specified cutoff.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        cutoff (float): Distance cutoff threshold
        
    Returns:
        torch.Tensor: Boolean mask indicating atom pairs below cutoff (shape: [num_atoms1, num_atoms2])
    """
    distances = calculate_atom_pair_distances(coords1, coords2)
    return distances < cutoff


def count_clashes(coords1: torch.Tensor, coords2: torch.Tensor, clash_distance: float = 1.0) -> int:
    """
    Count the number of atom clashes (distances below clash threshold).
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        clash_distance (float, optional): Distance threshold for clashes (default: 1.0 Å)
        
    Returns:
        int: Number of clashing atom pairs
    """
    clash_mask = find_all_distances_below_cutoff(coords1, coords2, clash_distance)
    return torch.sum(clash_mask).item()


def calculate_closest_atoms(coords1: torch.Tensor, coords2: torch.Tensor, top_n: int = 1) -> List[Tuple[int, int, float]]:
    """
    Find the closest atom pairs between two sets of coordinates.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        top_n (int, optional): Number of closest pairs to return
        
    Returns:
        List[Tuple[int, int, float]]: List of (index1, index2, distance) for closest pairs
    """
    distances = calculate_atom_pair_distances(coords1, coords2)
    flat_distances = distances.flatten()
    flat_indices = torch.argsort(flat_distances)
    
    num_atoms2 = coords2.shape[0]
    closest_pairs = []
    
    for idx in flat_indices[:top_n]:
        i = idx // num_atoms2
        j = idx % num_atoms2
        closest_pairs.append((i.item(), j.item(), flat_distances[idx].item()))
    
    return closest_pairs


def calculate_radius_of_gyration(coordinates: torch.Tensor) -> float:
    """
    Calculate radius of gyration for a set of coordinates.
    
    Args:
        coordinates (torch.Tensor): Atom coordinates (shape: [num_atoms, 3])
        
    Returns:
        float: Radius of gyration
    """
    center = torch.mean(coordinates, dim=0)
    centered_coords = coordinates - center
    sq_distances = torch.sum(centered_coords ** 2, dim=1)
    return torch.sqrt(torch.mean(sq_distances)).item()


def is_within_distance(coords1: torch.Tensor, coords2: torch.Tensor, distance: float) -> bool:
    """
    Check if any atom pair is within a specified distance.
    
    Args:
        coords1 (torch.Tensor): First set of atom coordinates (shape: [num_atoms1, 3])
        coords2 (torch.Tensor): Second set of atom coordinates (shape: [num_atoms2, 3])
        distance (float): Maximum allowed distance
        
    Returns:
        bool: True if any atom pair is within the specified distance
    """
    min_dist = find_minimum_distance(coords1, coords2)
    return min_dist < distance


def batch_calculate_distances(coords_list1: List[torch.Tensor], coords_list2: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculate distances for multiple coordinate set pairs in batch.
    
    Args:
        coords_list1 (List[torch.Tensor]): List of first coordinate sets
        coords_list2 (List[torch.Tensor]): List of second coordinate sets
        
    Returns:
        List[torch.Tensor]: List of distance matrices
    """
    if len(coords_list1) != len(coords_list2):
        raise ValueError("Coordinate list lengths must match")
    
    distance_matrices = []
    for coords1, coords2 in zip(coords_list1, coords_list2):
        distances = calculate_atom_pair_distances(coords1, coords2)
        distance_matrices.append(distances)
    
    return distance_matrices
