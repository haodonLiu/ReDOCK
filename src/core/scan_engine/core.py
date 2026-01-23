#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan Engine Core Module

This module provides the core scanning functionality for protein-protein docking,
including the main scan engine class and its primary methods.
"""

from typing import List, Tuple, Dict
import torch
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

from ..coordinate_manager import CoordinateManager
from ..energy_calculator import EnergyCalculator
from ...models.topology import Topology
from ...models.coordinate import Coordinate
from ...utils.logger import Logger
from ...utils.io_utils import PDBIO as PDBWriter
from ...utils.clustering import kmeans_clustering_conformations, hierarchical_clustering, dbscan_clustering
from ...utils.conformation_utils import generate_random_conformation, filter_clashing_conformations, calculate_conformation_rmsd, align_conformation_to_reference, cluster_conformations, select_representative_conformations, calculate_conformation_energies, rank_conformations_by_energy, analyze_conformation_diversity, generate_conformation_pool
from ...utils.coordinate_utils import calculate_geometric_center, translate_coordinates, rotate_coordinates_euler, generate_random_translation, generate_random_rotation
from ...utils.distance_utils import calculate_atom_pair_distances, find_minimum_distance, find_all_distances_below_cutoff, count_clashes
from .optimization import ConformationOptimizer


class ScanEngine:
    """
    Dedicated scan engine for protein-protein docking.
    
    This engine generates conformations by simultaneously scanning:
    1. Translation in xyz directions
    2. Rotation around xyz axes
    3. Validating and filtering conformations based on specified criteria
    """
    
    def __init__(self, logger: Logger, energy_calculator: EnergyCalculator, device: torch.device):
        """
        Initialize the scan engine.
        
        Args:
            logger (Logger): Logger instance for logging
            energy_calculator (EnergyCalculator): Energy calculator for VDW energy calculation
            device (torch.device): Device to use for calculations
        """
        self.logger = logger
        self.energy_calculator = energy_calculator
        self.device = device
        self.optimizer = ConformationOptimizer(logger, energy_calculator, device)
    
    def dedicated_scan(self, 
                      receptor_top: Topology, 
                      receptor_coord: Coordinate, 
                      ligand_top: Topology, 
                      ligand_coord: Coordinate, 
                      receptor_group: List[int], 
                      ligand_group: List[int],
                      initial_distance: float = 5.0,
                      coarse_translation_step: float = 1.0,
                      fine_translation_step: float = 0.4,
                      rotation_step: float = 60.0,
                      max_translation_range: float = 5.0,
                      min_residue_distance: float = 5.0,
                      max_residue_distance: float = 8.0) -> List[Tuple[Topology, Coordinate]]:
        """
        Perform dedicated scan to generate valid conformations using two-step strategy:
        1. Coarse scan with lower VDW requirements
        2. Fine scan on promising regions with higher VDW requirements
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            receptor_group (List[int]): Indices of receptor residue group atoms
            ligand_group (List[int]): Indices of ligand residue group atoms
            initial_distance (float): Initial distance between residue groups
            coarse_translation_step (float): Step size for coarse translation scan
            fine_translation_step (float): Step size for fine translation scan
            rotation_step (float): Step size for rotation scan
            max_translation_range (float): Maximum translation range in each direction
            min_residue_distance (float): Minimum residue group distance allowed
            max_residue_distance (float): Maximum residue group distance allowed
            
        Returns:
            List[Tuple[Topology, Coordinate]]: List of valid conformations as (topology, coordinate) tuples
        """
        self.logger.info("=== Dedicated Scan Engine ===")
        self.logger.info(f"Initial parameters:")
        self.logger.info(f"  Initial distance: {initial_distance:.2f} Å")
        self.logger.info(f"  Coarse translation step: {coarse_translation_step:.2f} Å")
        self.logger.info(f"  Fine translation step: {fine_translation_step:.2f} Å")
        self.logger.info(f"  Rotation step: {rotation_step:.2f} degrees")
        self.logger.info(f"  Max translation range: {max_translation_range:.2f} Å")
        self.logger.info(f"  Min residue distance: {min_residue_distance:.2f} Å")
        self.logger.info(f"  Max residue distance: {max_residue_distance:.2f} Å")
        
        # Calculate receptor residue group center
        receptor_group_coords = receptor_coord.coordinates[receptor_group]
        receptor_group_center = calculate_geometric_center(receptor_group_coords)
        self.logger.info(f"Receptor residue group center: ({receptor_group_center[0]:.4f}, {receptor_group_center[1]:.4f}, {receptor_group_center[2]:.4f})")
        
        # Calculate ligand residue group offset from ligand geometric center
        ligand_geometric_center = ligand_coord.calculate_geometric_center().to(self.device)
        ligand_group_coords = ligand_coord.coordinates[ligand_group]
        ligand_group_center = calculate_geometric_center(ligand_group_coords).to(self.device)
        ligand_group_offset = ligand_group_center - ligand_geometric_center
        self.logger.info(f"Ligand group offset: ({ligand_group_offset[0]:.4f}, {ligand_group_offset[1]:.4f}, {ligand_group_offset[2]:.4f})")
        
        # Save original ligand position
        original_position = ligand_coord.coordinates.clone()
        
        # Ensure all coordinates are on the correct device before calculations
        receptor_coord.coordinates = receptor_coord.coordinates.to(self.device)
        ligand_coord.coordinates = ligand_coord.coordinates.to(self.device)
        receptor_group_center = receptor_group_center.to(self.device)
        ligand_group_offset = ligand_group_offset.to(self.device)
        
        all_conformations = []
        
        # Step 1: Coarse scan
        self.logger.info("=== Step 1: Coarse Scan ===")
        self.logger.info(f"Using translation step: {coarse_translation_step:.2f} Å")
        
        # Perform coarse scan (translation + rotation simultaneously)
        coarse_conformations = self._simultaneous_scan(
            receptor_top=receptor_top,
            receptor_coord=receptor_coord,
            ligand_top=ligand_top,
            ligand_coord=ligand_coord,
            receptor_group_center=receptor_group_center,
            ligand_group_offset=ligand_group_offset,
            ligand_group=ligand_group,
            receptor_group=receptor_group,
            initial_distance=initial_distance,
            translation_step=coarse_translation_step,
            max_translation_range=max_translation_range,
            rotation_step=rotation_step,
            min_residue_distance=min_residue_distance,
            max_residue_distance=max_residue_distance,
            original_position=original_position.to(self.device),
            scan_type="coarse"
        )
        
        self.logger.info(f"Generated {len(coarse_conformations)} valid conformations from coarse scan")
        
        # If no coarse conformations found, exit directly
        if not coarse_conformations:
            self.logger.error("No valid conformations from coarse scan, exiting program")
            ligand_coord.coordinates = original_position.clone()
            return []
        
        # Step 1.5: Cluster coarse conformations
        # Create a copy of the original ligand for reference
        original_ligand_top = ligand_top
        original_ligand_coord = Coordinate(ligand_coord.coordinates.clone())
        
        # Cluster coarse conformations using K-means based on RMSD
        centroids, assignments = cluster_conformations(coarse_conformations, n_clusters=10, max_iter=100)
        
        # Create clusters dictionary
        coarse_clusters = {}
        for i in range(10):
            cluster_members = [conf for conf, idx in zip(coarse_conformations, assignments) if idx == i]
            if cluster_members:
                coarse_clusters[i] = cluster_members
        
        # Save clustered conformations
        self.save_clusters(
            clusters=coarse_clusters,
            receptor_top=receptor_top,
            receptor_coord=receptor_coord,
            output_prefix="coarse_cluster"
        )
        
        # Update coarse_conformations to use only cluster representatives (centroids)
        coarse_conformations = centroids
        self.logger.info(f"Using {len(coarse_conformations)} cluster representatives for fine scan")
        
        # Step 2: Fine scan on promising regions from coarse scan
        self.logger.info("=== Step 2: Fine Scan on Promising Regions ===")
        self.logger.info(f"Using translation step: {fine_translation_step:.2f} Å")
        
        # For each coarse conformation, perform fine scan around it
        for i, (coarse_top, coarse_coord) in enumerate(coarse_conformations):
            self.logger.info(f"Processing coarse conformation {i+1}/{len(coarse_conformations)}")
            
            # Calculate the position of this coarse conformation
            coarse_ligand_center = coarse_coord.calculate_geometric_center().to(self.device)
            coarse_ligand_group_center = self.utils.calculate_center(coarse_coord, ligand_group).to(self.device)
            
            # Calculate the offset from the initial position
            coarse_offset = coarse_ligand_group_center - (receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=self.device))
            
            # Perform fine scan around this position with smaller step size
            fine_conformations = self._fine_scan_around_position(
                receptor_top=receptor_top,
                receptor_coord=receptor_coord,
                ligand_top=ligand_top,
                ligand_coord=ligand_coord,
                receptor_group_center=receptor_group_center,
                ligand_group_offset=ligand_group_offset,
                ligand_group=ligand_group,
                receptor_group=receptor_group,
                center_offset=coarse_offset,
                initial_distance=initial_distance,
                fine_translation_step=fine_translation_step,
                min_residue_distance=min_residue_distance,
                max_residue_distance=max_residue_distance,
                original_position=original_position.to(self.device),
                rotation_step=rotation_step
            )
            
            self.logger.info(f"Generated {len(fine_conformations)} valid conformations from fine scan around coarse conformation {i+1}")
            
            # Add fine conformations to the list
            all_conformations.extend(fine_conformations)
        
        # If no fine conformations found, use coarse conformations
        if not all_conformations:
            self.logger.warning("No valid conformations from fine scan, using coarse conformations")
            all_conformations = coarse_conformations
        
        # Step 2.5: Cluster fine scan conformations into 5 clusters
        if all_conformations:
            self.logger.info("=== Step 2.5: Clustering Fine Scan Conformations ===")
            # Create a copy of the original ligand for reference
            original_ligand_top = ligand_top
            original_ligand_coord = Coordinate(ligand_coord.coordinates.clone())
            
            # Cluster fine scan conformations into 5 clusters
            centroids, assignments = cluster_conformations(all_conformations, n_clusters=5, max_iter=100)
            
            # Create clusters dictionary
            fine_clusters = {}
            for i in range(5):
                cluster_members = [conf for conf, idx in zip(all_conformations, assignments) if idx == i]
                if cluster_members:
                    fine_clusters[i] = cluster_members
            
            # Save clustered conformations
            self.save_clusters(
                clusters=fine_clusters,
                receptor_top=receptor_top,
                receptor_coord=receptor_coord,
                output_prefix="fine_cluster"
            )
            
            # Update all_conformations to use only cluster representatives (centroids)
            all_conformations = centroids
            self.logger.info(f"Using {len(all_conformations)} cluster representatives for optimization")
        
        # Step 3: Optimize conformations by applying random jitter to search for lowest energy states
        if all_conformations:
            self.logger.info("=== Step 3: Optimizing Conformations ===")
            optimized_conformations = []
            
            for i, (conf_top, conf_coord) in enumerate(tqdm(all_conformations, desc="Optimizing conformations")):
                # Optimize conformation with 1000 steps as requested
                optimized_top, optimized_coord = self.optimizer.optimize_conformation(
                    conformation_top=conf_top,
                    conformation_coord=conf_coord,
                    receptor_top=receptor_top,
                    receptor_coord=receptor_coord,
                    ligand_group=ligand_group,
                    receptor_group=receptor_group,
                    max_residue_distance=max_residue_distance,
                    max_steps=1000  # As requested
                )
                optimized_conformations.append((optimized_top, optimized_coord))
            
            all_conformations = optimized_conformations
            self.logger.info(f"Optimized {len(all_conformations)} conformations")
        
        # Restore original ligand position
        ligand_coord.coordinates = original_position.clone()
        
        self.logger.info(f"Total valid conformations generated: {len(all_conformations)}")
        return all_conformations
    
    def _simultaneous_scan(self, 
                          receptor_top: Topology,
                          receptor_coord: Coordinate,
                          ligand_top: Topology,
                          ligand_coord: Coordinate,
                          receptor_group_center: torch.Tensor,
                          ligand_group_offset: torch.Tensor,
                          ligand_group: List[int],
                          receptor_group: List[int],
                          initial_distance: float,
                          translation_step: float,
                          max_translation_range: float,
                          rotation_step: float,
                          min_residue_distance: float,
                          max_residue_distance: float,
                          original_position: torch.Tensor,
                          scan_type: str = "coarse") -> List[Tuple[Topology, Coordinate]]:
        """
        Perform simultaneous translation and rotation scan to generate conformations.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            ligand_group (List[int]): Indices of ligand residue group atoms
            receptor_group (List[int]): Indices of receptor residue group atoms
            initial_distance (float): Initial distance between residue groups
            translation_step (float): Step size for translation scan
            max_translation_range (float): Maximum translation range in each direction
            rotation_step (float): Step size for rotation scan
            min_residue_distance (float): Minimum residue group distance allowed
            max_residue_distance (float): Maximum residue group distance allowed
            original_position (torch.Tensor): Original ligand position
            scan_type (str): Type of scan, either "coarse" or "fine"
            
        Returns:
            List[Tuple[Topology, Coordinate]]: List of valid conformations as (topology, coordinate) tuples
        """
        conformations = []
        
        # Generate translation grid
        num_steps = int(max_translation_range / translation_step)
        translation_values = torch.linspace(-max_translation_range, max_translation_range, num_steps * 2 + 1, device=self.device)
        total_translation_points = len(translation_values) ** 3
        
        # Generate rotation parameters based on scan type
        if scan_type == "coarse":
            # Coarse scan: 60 degrees step, 0-360 range
            rotation_angles = torch.linspace(0.0, 360.0, int(360.0 / rotation_step), device=self.device)
        else:  # fine scan
            # Fine scan: 10 degrees step, -30 to 30 range
            rotation_angles = torch.linspace(-30.0, 30.0, int(60.0 / 10.0) + 1, device=self.device)
        
        rotation_axes = [
            torch.tensor([1.0, 0.0, 0.0], device=self.device),  # x-axis
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # y-axis  
            torch.tensor([0.0, 0.0, 1.0], device=self.device)   # z-axis
        ]
        
        total_rotation_combinations = len(rotation_axes) * len(rotation_angles)
        total_points = total_translation_points * total_rotation_combinations
        
        self.logger.info(f"{scan_type.capitalize()} scan: {len(translation_values)} translation points in each dimension, {len(rotation_axes)} axes, {len(rotation_angles)} angles")
        self.logger.info(f"Total scan points: {total_points}")
        
        # Calculate initial ligand geometric center with validation
        initial_ligand_group_center, initial_ligand_center = self._calculate_valid_initial_position(
            receptor_top=receptor_top,
            receptor_coord=receptor_coord,
            ligand_top=ligand_top,
            ligand_coord=ligand_coord,
            receptor_group_center=receptor_group_center,
            ligand_group_offset=ligand_group_offset,
            initial_distance=initial_distance,
            min_residue_distance=min_residue_distance,
            max_residue_distance=max_residue_distance
        )
        
        # Calculate initial translation
        ligand_coord.coordinates = original_position.clone()
        initial_translation = initial_ligand_center - ligand_coord.calculate_geometric_center().to(self.device)
        
        processed_points = 0
        valid_confs = 0
        
        # Create coordinate manager once
        coord_manager = CoordinateManager(ligand_coord, device=self.device)
        
        # Pre-calculate receptor group center for validation
        receptor_group_coords = receptor_coord.coordinates[receptor_group]
        receptor_group_center = calculate_geometric_center(receptor_group_coords)
        
        # Create a generator for all translation and rotation combinations
        def scan_generator():
            for dx in translation_values:
                for dy in translation_values:
                    for dz in translation_values:
                        for axis in rotation_axes:
                            for angle in rotation_angles:
                                yield dx, dy, dz, axis, angle
        
        # Process all combinations in a single loop
        filtered_points = 0
        for dx, dy, dz, axis, angle in tqdm(scan_generator(), total=total_points, desc=f"{scan_type.capitalize()} scan points"):
            processed_points += 1
            
            # Skip some combinations to reduce redundant calculations
            if scan_type == "coarse":
                # Skip 0 degrees rotation for coarse scan (already checked as base conformation)
                if angle == 0.0:
                    continue
            
            # 从数学角度过滤无效参数
            # 1. 计算平移后的残基组中心
            translation = torch.tensor([dx, dy, dz], device=self.device)
            current_ligand_group_center = initial_ligand_group_center + translation
            
            # 2. 计算残基组距离
            residue_group_distance = torch.norm(receptor_group_center - current_ligand_group_center).item()
            
            # 3. 过滤掉残基组距离超出范围的组合
            if residue_group_distance < min_residue_distance or residue_group_distance > max_residue_distance:
                filtered_points += 1
                continue
            
            # 4. 计算配体中心到受体残基组的距离
            current_ligand_center = current_ligand_group_center - ligand_group_offset
            ligand_to_receptor_group = torch.norm(current_ligand_center - receptor_group_center).item()
            
            # 5. 过滤掉残基组距离大于配体中心到受体残基组距离的组合
            if residue_group_distance > ligand_to_receptor_group:
                filtered_points += 1
                continue
            
            # 6. 从数学角度估计最小原子距离
            # 简单估算：如果残基组距离小于一定值，可能存在原子碰撞
            if residue_group_distance < 3.0:  # 3Å是一个保守的阈值
                filtered_points += 1
                continue
            
            # Reset ligand to original position
            ligand_coord.coordinates = original_position.clone()
            
            # Apply translation
            final_translation = initial_translation + translation
            coord_manager.translate_coordinates(final_translation)
            
            # Calculate current ligand geometric center for rotation (using direct translation)
            current_ligand_center = initial_ligand_center + translation
            
            # Apply rotation
            coord_manager.rotate_around_axis(axis, angle.item(), current_ligand_center)
            
            # Validate conformation
            if self._validate_conformation(
                receptor_coord=receptor_coord,
                ligand_coord=ligand_coord,
                ligand_group=ligand_group,
                receptor_group=receptor_group,
                min_residue_distance=min_residue_distance,
                max_residue_distance=max_residue_distance
            ):
                # Valid conformation found
                valid_ligand_top = ligand_top
                valid_ligand_coord = Coordinate(ligand_coord.coordinates.clone())
                conformations.append((valid_ligand_top, valid_ligand_coord))
                valid_confs += 1
        
        self.logger.info(f"Filtered out {filtered_points} invalid parameter combinations based on mathematical analysis")
        
        self.logger.info(f"{scan_type.capitalize()} scan completed: {len(conformations)} valid conformations from {processed_points} processed points")
        return conformations
    
    def _calculate_valid_initial_position(self, 
                                        receptor_top: Topology,
                                        receptor_coord: Coordinate,
                                        ligand_top: Topology,
                                        ligand_coord: Coordinate,
                                        receptor_group_center: torch.Tensor,
                                        ligand_group_offset: torch.Tensor,
                                        initial_distance: float,
                                        min_residue_distance: float,
                                        max_residue_distance: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate a valid initial position for the ligand that satisfies:
        1. Distance constraint (within min and max residue distance)
        2. Orientation constraint (residue group distance <= ligand center to receptor group distance)
        3. No collision (minimum atom distance >= 1A)
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            initial_distance (float): Initial distance between residue groups
            min_residue_distance (float): Minimum residue group distance allowed
            max_residue_distance (float): Maximum residue group distance allowed
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Valid initial ligand group center and ligand center
        """
        device = receptor_group_center.device
        
        # Start with the original initial position
        initial_ligand_group_center = receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=device)
        initial_ligand_center = initial_ligand_group_center - ligand_group_offset
        
        # Validate the initial position
        if self._validate_initial_position(receptor_top, receptor_coord, ligand_top, ligand_coord, initial_ligand_center, 
                                         receptor_group_center, ligand_group_offset, 
                                         min_residue_distance, max_residue_distance):
            self.logger.info(f"Initial position is valid: ligand group center = ({initial_ligand_group_center[0]:.4f}, {initial_ligand_group_center[1]:.4f}, {initial_ligand_group_center[2]:.4f})")
            return initial_ligand_group_center, initial_ligand_center
        
        # If initial position is invalid, search for a valid one
        self.logger.warning("Initial position is invalid, searching for a valid position...")
        
        # Search in a sphere around the initial position
        search_radius = 5.0
        num_points = 100
        
        # Generate points on a sphere
        from ...utils.algorithms import generate_uniform_sphere_points
        sphere_points = generate_uniform_sphere_points(num_points)
        
        for i in range(num_points):
            # Generate a point at different distances
            for distance in torch.linspace(min_residue_distance, max_residue_distance, steps=5):
                # Calculate direction vector
                direction = sphere_points[i] * distance
                test_ligand_group_center = receptor_group_center + direction
                test_ligand_center = test_ligand_group_center - ligand_group_offset
                
                # Validate this position
                if self._validate_initial_position(receptor_top, receptor_coord, ligand_top, ligand_coord, test_ligand_center, 
                                                 receptor_group_center, ligand_group_offset, 
                                                 min_residue_distance, max_residue_distance):
                    self.logger.info(f"Found valid initial position: ligand group center = ({test_ligand_group_center[0]:.4f}, {test_ligand_group_center[1]:.4f}, {test_ligand_group_center[2]:.4f})")
                    return test_ligand_group_center, test_ligand_center
        
        # If no valid position found, return the original position with a warning
        self.logger.warning("Could not find a fully valid initial position, using original position")
        return initial_ligand_group_center, initial_ligand_center
    
    def _validate_initial_position(self, 
                                  receptor_top: Topology,
                                  receptor_coord: Coordinate,
                                  ligand_top: Topology,
                                  ligand_coord: Coordinate,
                                  ligand_center: torch.Tensor,
                                  receptor_group_center: torch.Tensor,
                                  ligand_group_offset: torch.Tensor,
                                  min_residue_distance: float,
                                  max_residue_distance: float) -> bool:
        """
        Validate an initial position for the ligand.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            ligand_center (torch.Tensor): Ligand geometric center
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            min_residue_distance (float): Minimum residue group distance allowed
            max_residue_distance (float): Maximum residue group distance allowed
            
        Returns:
            bool: True if position is valid, False otherwise
        """
        # Calculate ligand group center
        ligand_group_center = ligand_center + ligand_group_offset
        
        # 1. Check distance constraint
        residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
        if residue_group_distance < min_residue_distance or residue_group_distance > max_residue_distance:
            return False
        
        # 2. Check orientation constraint
        ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
        if residue_group_distance > ligand_to_receptor_group:
            return False
        
        # 3. Check collision constraint
        # Calculate the minimum possible atom distance based on residue group distance
        # This is a conservative estimate
        if residue_group_distance < 3.0:
            return False
        
        return True
    
    def _validate_conformation(self, 
                              receptor_coord: Coordinate, 
                              ligand_coord: Coordinate, 
                              ligand_group: List[int], 
                              receptor_group: List[int], 
                              min_residue_distance: float, 
                              max_residue_distance: float) -> bool:
        """
        Validate a conformation by checking distance constraints, orientation, and clashes.
        
        Args:
            receptor_coord (Coordinate): Receptor coordinates
            ligand_coord (Coordinate): Ligand coordinates
            ligand_group (List[int]): Ligand group indices
            receptor_group (List[int]): Receptor group indices
            min_residue_distance (float): Minimum allowed residue distance
            max_residue_distance (float): Maximum allowed residue distance
            
        Returns:
            bool: True if conformation is valid, False otherwise
        """
        from ...utils.conformation_utils import validate_conformation
        return validate_conformation(
            receptor_coord=receptor_coord,
            ligand_coord=ligand_coord,
            ligand_group=ligand_group,
            receptor_group=receptor_group,
            min_residue_distance=min_residue_distance,
            max_residue_distance=max_residue_distance,
            clash_distance=1.0
        )
    
    def _fine_scan_around_position(self, 
                                 receptor_top: Topology,
                                 receptor_coord: Coordinate,
                                 ligand_top: Topology,
                                 ligand_coord: Coordinate,
                                 receptor_group_center: torch.Tensor,
                                 ligand_group_offset: torch.Tensor,
                                 ligand_group: List[int],
                                 receptor_group: List[int],
                                 center_offset: torch.Tensor,
                                 initial_distance: float,
                                 fine_translation_step: float,
                                 min_residue_distance: float,
                                 max_residue_distance: float,
                                 original_position: torch.Tensor,
                                 rotation_step: float) -> List[Tuple[Topology, Coordinate]]:
        """
        Perform fine scan around a specific position from coarse scan.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            ligand_group (List[int]): Indices of ligand residue group atoms
            receptor_group (List[int]): Indices of receptor residue group atoms
            center_offset (torch.Tensor): Offset from initial position to scan around
            initial_distance (float): Initial distance between residue groups
            fine_translation_step (float): Step size for fine translation scan
            min_residue_distance (float): Minimum residue group distance allowed
            max_residue_distance (float): Maximum residue group distance allowed
            original_position (torch.Tensor): Original ligand position
            rotation_step (float): Step size for rotation scan
            
        Returns:
            List[Tuple[Topology, Coordinate]]: List of valid fine scan conformations as (topology, coordinate) tuples
        """
        conformations = []
        
        # Define fine scan range according to requirements: ±1A
        fine_scan_range = 1.0  # Å around the coarse position
        
        # Generate fine translation grid with specified step size
        num_steps = int(fine_scan_range / fine_translation_step)
        translation_values = torch.linspace(-fine_scan_range, fine_scan_range, num_steps * 2 + 1, device=self.device)
        
        # Generate rotation parameters for fine scan: ±30 degrees with 15 degree step
        rotation_step = 15.0
        rotation_angles = torch.linspace(-30.0, 30.0, int(60.0 / rotation_step) + 1, device=self.device)
        rotation_axes = [
            torch.tensor([1.0, 0.0, 0.0], device=self.device),  # x-axis
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # y-axis  
            torch.tensor([0.0, 0.0, 1.0], device=self.device)   # z-axis
        ]
        
        total_points = len(translation_values) ** 3 * len(rotation_axes) * len(rotation_angles)
        self.logger.info(f"Fine scan grid: {len(translation_values)} points in each dimension, total {total_points} points")
        
        # Calculate initial ligand geometric center for this coarse position
        initial_ligand_group_center = receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=self.device) + center_offset
        initial_ligand_center = initial_ligand_group_center - ligand_group_offset
        
        # Calculate initial translation
        ligand_coord.coordinates = original_position.clone()
        initial_translation = initial_ligand_center - ligand_coord.calculate_geometric_center().to(self.device)
        
        processed_points = 0
        valid_confs = 0
        filtered_points = 0
        
        # Create coordinate manager once
        coord_manager = CoordinateManager(ligand_coord, device=self.device)
        
        # Pre-calculate receptor group center for validation
        receptor_group_coords = receptor_coord.coordinates[receptor_group]
        receptor_group_center = calculate_geometric_center(receptor_group_coords)
        
        # Create a generator for all fine scan combinations
        def fine_scan_generator():
            for dx in translation_values:
                for dy in translation_values:
                    for dz in translation_values:
                        for axis in rotation_axes:
                            for angle in rotation_angles:
                                yield dx, dy, dz, axis, angle
        
        # Process all combinations in a single loop
        for dx, dy, dz, axis, angle in tqdm(fine_scan_generator(), total=total_points, desc="Fine scan points"):
            processed_points += 1
            
            # 从数学角度过滤无效参数
            # 1. 计算平移后的残基组中心
            fine_translation = torch.tensor([dx, dy, dz], device=self.device)
            current_ligand_group_center = initial_ligand_group_center + fine_translation
            
            # 2. 计算残基组距离
            residue_group_distance = torch.norm(receptor_group_center - current_ligand_group_center).item()
            
            # 3. 过滤掉残基组距离超出范围的组合
            if residue_group_distance < min_residue_distance or residue_group_distance > max_residue_distance:
                filtered_points += 1
                continue
            
            # 4. 计算配体中心到受体残基组的距离
            current_ligand_center = current_ligand_group_center - ligand_group_offset
            ligand_to_receptor_group = torch.norm(current_ligand_center - receptor_group_center).item()
            
            # 5. 过滤掉残基组距离大于配体中心到受体残基组距离的组合
            if residue_group_distance > ligand_to_receptor_group:
                filtered_points += 1
                continue
            
            # 6. 从数学角度估计最小原子距离
            if residue_group_distance < 3.0:
                filtered_points += 1
                continue
            
            # Reset ligand to original position
            ligand_coord.coordinates = original_position.clone()
            
            # Apply translation
            final_translation = initial_translation + fine_translation
            coord_manager.translate_coordinates(final_translation)
            
            # Calculate current ligand geometric center for rotation (using direct translation)
            current_ligand_center = initial_ligand_center + fine_translation
            
            # Apply rotation
            coord_manager.rotate_around_axis(axis, angle.item(), current_ligand_center)
            
            # Validate conformation
            if self._validate_conformation(
                receptor_coord=receptor_coord,
                ligand_coord=ligand_coord,
                ligand_group=ligand_group,
                receptor_group=receptor_group,
                min_residue_distance=min_residue_distance,
                max_residue_distance=max_residue_distance
            ):
                # Valid conformation found
                valid_ligand_top = ligand_top
                valid_ligand_coord = Coordinate(ligand_coord.coordinates.clone())
                conformations.append((valid_ligand_top, valid_ligand_coord))
                valid_confs += 1
        
        self.logger.info(f"Filtered out {filtered_points} invalid parameter combinations based on mathematical analysis")
        self.logger.info(f"Fine scan completed: {len(conformations)} valid conformations from {processed_points} processed points")
        
        return conformations
    
    def save_clusters(self, clusters: Dict[int, List[Tuple[Topology, Coordinate]]], receptor_top: Topology, receptor_coord: Coordinate, output_prefix: str = "cluster"):
        """
        Save clustered conformations to PDB files in a run-specific directory.
        
        Args:
            clusters (Dict[int, List[Tuple[Topology, Coordinate]]]): Dictionary of clusters
            receptor_top (Topology): Receptor topology
            receptor_coord (Coordinate): Receptor coordinates
            output_prefix (str): Prefix for output PDB filenames
        """
        import os
        import datetime
        
        # Create a run-specific directory based on timestamp
        timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join("output", timestamp)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
            self.logger.info(f"Created run-specific output directory: {run_dir}")
        
        # Create output directory based on prefix within the run directory
        output_dir = os.path.join(run_dir, f"{output_prefix}_files")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
        
        # Create PDB writer instance
        writer = PDBWriter(logger=self.logger)
        
        for cluster_id, conformations in clusters.items():
            # Save best conformation from each cluster (first one)
            best_top, best_coord = conformations[0]
            
            # Save to PDB file using enhanced IO module
            output_file = os.path.join(output_dir, f"{output_prefix}_{cluster_id}.pdb")
            if writer.write_combined_structure(receptor_top, receptor_coord, best_top, best_coord, output_file):
                self.logger.info(f"Saved best conformation from cluster {cluster_id} to {output_file}")
            else:
                self.logger.error(f"Failed to save conformation from cluster {cluster_id}")
