#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conformation Utilities Module

Handles conformation-related operations for protein structures, including:
- Conformation generation
- Conformation filtering
- Conformation analysis
- RMSD calculations
- Clustering of conformations
"""

import torch
from typing import List, Tuple, Dict, Optional
from ..models.topology import Topology
from ..models.coordinate import Coordinate
from .coordinate_utils import calculate_rmsd, kabsch_algorithm, align_coordinates
from .distance_utils import calculate_atom_pair_distances, count_clashes


def generate_random_conformation(original_top: Topology, original_coord: Coordinate, max_rotation: float = 180.0, max_translation: float = 10.0) -> Tuple[Topology, Coordinate]:
    """
    Generate a random conformation by rotating and translating the original structure.
    
    Args:
        original_top (Topology): Original protein topology
        original_coord (Coordinate): Original protein coordinates
        max_rotation (float, optional): Maximum rotation angle in degrees
        max_translation (float, optional): Maximum translation distance in Å
        
    Returns:
        Tuple[Topology, Coordinate]: Randomly generated conformation as (topology, coordinate) tuple
    """
    from .coordinate_utils import generate_random_translation
    
    # Generate random rotation angles (in degrees, since Coordinate.rotate_euler uses degrees)
    angle_x = (torch.rand(1).item() - 0.5) * 2 * max_rotation
    angle_y = (torch.rand(1).item() - 0.5) * 2 * max_rotation
    angle_z = (torch.rand(1).item() - 0.5) * 2 * max_rotation
    
    # Generate random translation
    translation = generate_random_translation(max_translation)
    
    # Create a copy of the original coordinates and apply transformations
    new_coord = original_coord.copy()
    new_coord.rotate_euler(angle_x, angle_y, angle_z)
    new_coord.translate(translation)
    
    return (original_top, new_coord)


def filter_clashing_conformations(conformations: List[Tuple[Topology, Coordinate]], receptor_top: Topology, receptor_coord: Coordinate, clash_threshold: float = 1.0) -> List[Tuple[Topology, Coordinate]]:
    """
    Filter out conformations that clash with the receptor.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations to filter
        receptor_top (Topology): Receptor protein topology
        receptor_coord (Coordinate): Receptor protein coordinates
        clash_threshold (float, optional): Distance threshold for clashes (default: 1.0 Å)
        
    Returns:
        List[Tuple[Topology, Coordinate]]: Filtered list of non-clashing conformations
    """
    filtered_conformations = []
    
    for conformation in conformations:
        _, conf_coord = conformation
        clash_count = count_clashes(receptor_coord.coordinates, conf_coord.coordinates, clash_threshold)
        if clash_count == 0:
            filtered_conformations.append(conformation)
    
    return filtered_conformations


def calculate_conformation_rmsd(conformation1: Tuple[Topology, Coordinate], conformation2: Tuple[Topology, Coordinate]) -> float:
    """
    Calculate RMSD between two conformations of the same protein.
    
    Args:
        conformation1 (Tuple[Topology, Coordinate]): First conformation
        conformation2 (Tuple[Topology, Coordinate]): Second conformation
        
    Returns:
        float: RMSD value in Å
    """
    _, conf1_coord = conformation1
    _, conf2_coord = conformation2
    return calculate_rmsd(conf1_coord.coordinates, conf2_coord.coordinates)


def align_conformation_to_reference(conformation: Tuple[Topology, Coordinate], reference: Tuple[Topology, Coordinate]) -> Tuple[Topology, Coordinate]:
    """
    Align a conformation to a reference structure using Kabsch algorithm.
    
    Args:
        conformation (Tuple[Topology, Coordinate]): Conformation to align
        reference (Tuple[Topology, Coordinate]): Reference structure
        
    Returns:
        Tuple[Topology, Coordinate]: Aligned conformation as (topology, coordinate) tuple
    """
    conf_top, conf_coord = conformation
    _, ref_coord = reference
    aligned_coords = align_coordinates(conf_coord.coordinates, ref_coord.coordinates)
    aligned_coord = Coordinate(aligned_coords)
    return (conf_top, aligned_coord)


def cluster_conformations(conformations: List[Tuple[Topology, Coordinate]], n_clusters: int, max_iter: int = 100) -> Tuple[List[Tuple[Topology, Coordinate]], List[int]]:
    """
    Cluster conformations using K-means algorithm based on RMSD.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations to cluster
        n_clusters (int): Number of clusters
        max_iter (int, optional): Maximum number of iterations
        
    Returns:
        Tuple[List[Tuple[Topology, Coordinate]], List[int]]: Cluster centroids and cluster assignments
    """
    if not conformations:
        return [], []
    
    # Convert conformations to coordinate tensors
    coordinates = torch.stack([conf[1].coordinates for conf in conformations])
    num_conformations = len(conformations)
    
    # Initialize centroids randomly
    centroid_indices = torch.randperm(num_conformations)[:n_clusters]
    centroids = coordinates[centroid_indices]
    
    # K-means clustering
    for _ in range(max_iter):
        # Calculate RMSD from each conformation to each centroid
        rmsd_matrix = torch.zeros((num_conformations, n_clusters), dtype=torch.float32)
        for i in range(num_conformations):
            for j in range(n_clusters):
                rmsd_matrix[i, j] = calculate_rmsd(coordinates[i], centroids[j])
        
        # Assign each conformation to closest centroid
        assignments = torch.argmin(rmsd_matrix, dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for j in range(n_clusters):
            cluster_members = coordinates[assignments == j]
            if len(cluster_members) > 0:
                # For centroid, use the conformation closest to the mean
                mean_coords = torch.mean(cluster_members, dim=0)
                distances = torch.tensor([calculate_rmsd(mean_coords, member) for member in cluster_members])
                closest_idx = torch.argmin(distances)
                new_centroids[j] = cluster_members[closest_idx]
            else:
                # If cluster is empty, keep original centroid
                new_centroids[j] = centroids[j]
        
        # Check convergence
        if torch.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    # Create centroid structures
    centroid_structures = []
    centroid_top = conformations[0][0]  # All conformations share the same topology
    for centroid_coords in centroids:
        centroid_coord = Coordinate(centroid_coords)
        centroid_structures.append((centroid_top, centroid_coord))
    
    return centroid_structures, assignments.tolist()


def select_representative_conformations(conformations: List[Tuple[Topology, Coordinate]], n_representatives: int) -> List[Tuple[Topology, Coordinate]]:
    """
    Select representative conformations using clustering.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations
        n_representatives (int): Number of representative conformations to select
        
    Returns:
        List[Tuple[Topology, Coordinate]]: List of representative conformations
    """
    if len(conformations) <= n_representatives:
        return conformations
    
    centroids, _ = cluster_conformations(conformations, n_representatives)
    return centroids


def calculate_conformation_energies(conformations: List[Tuple[Topology, Coordinate]], receptor_top: Topology, receptor_coord: Coordinate, energy_calculator) -> List[Dict[str, float]]:
    """
    Calculate energies for a list of conformations.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations
        receptor_top (Topology): Receptor protein topology
        receptor_coord (Coordinate): Receptor protein coordinates
        energy_calculator: Energy calculator object with calculate_batch_energy method
        
    Returns:
        List[Dict[str, float]]: List of energy dictionaries for each conformation
    """
    return energy_calculator.calculate_batch_energy(receptor_top, receptor_coord, conformations)


def rank_conformations_by_energy(conformations: List[Tuple[Topology, Coordinate]], energies: List[Dict[str, float]]) -> List[Tuple[Tuple[Topology, Coordinate], Dict[str, float]]]:
    """
    Rank conformations by their total energy.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations
        energies (List[Dict[str, float]]): List of energy dictionaries
        
    Returns:
        List[Tuple[Tuple[Topology, Coordinate], Dict[str, float]]]: List of (conformation, energy) pairs sorted by total energy
    """
    # Pair conformations with their energies
    paired = list(zip(conformations, energies))
    
    # Sort by total energy
    sorted_paired = sorted(paired, key=lambda x: x[1]['total'])
    
    return sorted_paired


def analyze_conformation_diversity(conformations: List[Tuple[Topology, Coordinate]]) -> Dict[str, float]:
    """
    Analyze the diversity of a set of conformations.
    
    Args:
        conformations (List[Tuple[Topology, Coordinate]]): List of conformations
        
    Returns:
        Dict[str, float]: Diversity metrics including average RMSD, min RMSD, and max RMSD
    """
    if len(conformations) < 2:
        return {
            'average_rmsd': 0.0,
            'min_rmsd': 0.0,
            'max_rmsd': 0.0
        }
    
    # Calculate all pairwise RMSDs
    rmsds = []
    for i in range(len(conformations)):
        for j in range(i + 1, len(conformations)):
            rmsd = calculate_conformation_rmsd(conformations[i], conformations[j])
            rmsds.append(rmsd)
    
    # Calculate statistics
    rmsds_tensor = torch.tensor(rmsds)
    stats = {
        'average_rmsd': torch.mean(rmsds_tensor).item(),
        'min_rmsd': torch.min(rmsds_tensor).item(),
        'max_rmsd': torch.max(rmsds_tensor).item()
    }
    
    return stats


def generate_conformation_pool(original_top: Topology, original_coord: Coordinate, pool_size: int, receptor_top: Optional[Topology] = None, receptor_coord: Optional[Coordinate] = None, clash_threshold: float = 1.0) -> List[Tuple[Topology, Coordinate]]:
    """
    Generate a pool of diverse, non-clashing conformations.
    
    Args:
        original_top (Topology): Original protein topology
        original_coord (Coordinate): Original protein coordinates
        pool_size (int): Size of conformation pool to generate
        receptor_top (Optional[Topology], optional): Receptor topology for clash filtering
        receptor_coord (Optional[Coordinate], optional): Receptor coordinates for clash filtering
        clash_threshold (float, optional): Distance threshold for clashes
        
    Returns:
        List[Tuple[Topology, Coordinate]]: Generated conformation pool as list of (topology, coordinate) tuples
    """
    conformations = []
    
    while len(conformations) < pool_size:
        # Generate random conformation
        random_conformation = generate_random_conformation(original_top, original_coord)
        
        # Check for clashes if receptor is provided
        if receptor_top is None or receptor_coord is None or count_clashes(receptor_coord.coordinates, random_conformation[1].coordinates, clash_threshold) == 0:
            conformations.append(random_conformation)
    
    return conformations


def validate_conformation(receptor_coord: Coordinate, 
                         ligand_coord: Coordinate, 
                         ligand_group: List[int], 
                         receptor_group: List[int], 
                         min_residue_distance: float, 
                         max_residue_distance: float, 
                         clash_distance: float = 1.0) -> bool:
    """
    Validate a conformation by checking distance constraints, orientation, and clashes.
    
    Args:
        receptor_coord (Coordinate): Receptor coordinates
        ligand_coord (Coordinate): Ligand coordinates
        ligand_group (List[int]): Ligand group indices
        receptor_group (List[int]): Receptor group indices
        min_residue_distance (float): Minimum allowed residue distance
        max_residue_distance (float): Maximum allowed residue distance
        clash_distance (float, optional): Distance threshold for clashes
        
    Returns:
        bool: True if conformation is valid, False otherwise
    """
    from .coordinate_utils import calculate_geometric_center
    
    # Calculate residue group centers
    ligand_group_coords = ligand_coord.coordinates[ligand_group]
    ligand_group_center = calculate_geometric_center(ligand_group_coords)
    ligand_center = ligand_coord.calculate_geometric_center().to(ligand_group_center.device)
    
    receptor_group_coords = receptor_coord.coordinates[receptor_group]
    receptor_group_center = calculate_geometric_center(receptor_group_coords)
    
    # Check residue group distance
    residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
    if residue_group_distance < min_residue_distance or residue_group_distance > max_residue_distance:
        return False
    
    # Check orientation constraint
    ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
    if residue_group_distance > ligand_to_receptor_group:
        return False
    
    # Check for atom clashes
    clash_count = count_clashes(receptor_coord.coordinates, ligand_coord.coordinates, clash_distance)
    if clash_count > 0:
        return False
    
    return True
