#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan Engine Optimization Module

This module provides conformation optimization functionality for protein-protein docking,
including random jitter optimization to find lowest energy states.
"""

from typing import List
import torch
from ...models.topology import Topology
from ...models.coordinate import Coordinate
from ...core.energy_calculator import EnergyCalculator
from ...core.coordinate_manager import CoordinateManager
from ...utils.logger import Logger


class ConformationOptimizer:
    """
    Conformation optimizer for protein-protein docking.
    
    This class provides methods for optimizing conformations to find lowest energy states.
    """
    
    def __init__(self, logger: Logger, energy_calculator: EnergyCalculator, device: torch.device):
        """
        Initialize the conformation optimizer.
        
        Args:
            logger (Logger): Logger instance for logging
            energy_calculator (EnergyCalculator): Energy calculator for collision score calculation
            device (torch.device): Device to use for calculations
        """
        self.logger = logger
        self.energy_calculator = energy_calculator
        self.device = device
    
    def optimize_conformation(self, 
                              ligand_topology: Topology, 
                              conformation: Coordinate, 
                              receptor_topology: Topology, 
                              receptor_coordinate: Coordinate, 
                              ligand_group: List[int], 
                              receptor_group: List[int], 
                              max_residue_distance: float, 
                              max_steps: int = 1000, 
                              batch_size: int = 10) -> Coordinate:
        """
        Optimize a conformation by applying random jitter and selecting the lowest energy state.
        Considers both collision score and electrostatic energy.
        
        Args:
            ligand_topology (Topology): Ligand topology
            conformation (Coordinate): Original conformation
            receptor_topology (Topology): Receptor topology
            receptor_coordinate (Coordinate): Receptor coordinate
            ligand_group (List[int]): Indices of ligand residue group atoms
            receptor_group (List[int]): Indices of receptor residue group atoms
            max_residue_distance (float): Maximum residue group distance allowed
            max_steps (int): Number of optimization steps (jitters) to apply
            batch_size (int): Number of jittered conformations to generate per batch
            
        Returns:
            Coordinate: Optimized conformation with lowest energy
        """
        # Start with the original conformation
        best_conformation = self.create_ligand_copy(conformation)
        best_energy = float('inf')
        
        # Import utilities for validation
        from ...utils.coordinate_utils import calculate_geometric_center
        from ...utils.distance_utils import count_clashes
        
        # Perform optimization in batches
        import tqdm
        for _ in tqdm.tqdm(range(max_steps)):
            # Generate batch_size jittered conformations
            jittered_confs = self.random_jitter_batch(best_conformation, batch_size)
            
            # Validate the jittered conformations
            valid_mask = []
            for conf in jittered_confs:
                # Calculate residue group centers
                ligand_group_coords = conf.coordinates[ligand_group]
                ligand_group_center = calculate_geometric_center(ligand_group_coords)
                ligand_center = conf.calculate_geometric_center().to(self.device)
                
                receptor_group_coords = receptor_coordinate.coordinates[receptor_group]
                receptor_group_center = calculate_geometric_center(receptor_group_coords)
                
                # Check residue group distance
                residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
                if residue_group_distance < 4.0 or residue_group_distance > max_residue_distance:
                    valid_mask.append(False)
                    continue
                
                # Check orientation constraint
                ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
                if residue_group_distance > ligand_to_receptor_group:
                    valid_mask.append(False)
                    continue
                
                # Check for atom clashes
                clash_count = count_clashes(receptor_coordinate.coordinates, conf.coordinates, clash_distance=1.0)
                if clash_count > 0:
                    valid_mask.append(False)
                    continue
                
                valid_mask.append(True)
            
            # Process only valid conformations
            valid_confs = [conf for conf, valid in zip(jittered_confs, valid_mask) if valid]
            
            if not valid_confs:
                continue
            
            # Calculate energy for valid conformations
            for conf in valid_confs:
                # Calculate collision score (van der Waals energy)
                collision_score = self.energy_calculator.calculate_collision_score(receptor_coordinate, conf)
                # Calculate electrostatic energy
                electrostatic_energy = self.energy_calculator.electrostatic_calculator.cal_ele_energy(receptor_coordinate, conf)
                # Calculate total energy
                total_energy = collision_score + electrostatic_energy
                
                # Update best conformation if current is better
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_conformation = conf
        
        return best_conformation
    
    def random_jitter(self, 
                      conformation: Coordinate, 
                      translation_std: float = 0.2, 
                      rotation_std: float = 2.0) -> Coordinate:
        """
        Apply random jitter to a conformation to search for lower energy states.
        Uses normal distribution sampling for translation and small angle rotation.
        
        Args:
            conformation (Coordinate): Original conformation
            translation_std (float): Standard deviation for translation jitter (normal distribution) in Å
            rotation_std (float): Standard deviation for rotation jitter (normal distribution) in degrees
            
        Returns:
            Coordinate: Jittered conformation
        """
        # Create a copy of the conformation
        jittered_conf = Coordinate()
        jittered_conf.coordinates = conformation.coordinates.detach().clone()
        
        # Create coordinate manager
        coord_manager = CoordinateManager(jittered_conf, device=self.device)
        
        # Apply random translation jitter using normal distribution
        translation_jitter = torch.randn(3, device=self.device) * translation_std
        coord_manager.translate_coordinates(translation_jitter)
        
        # Apply random rotation jitter around a random axis
        random_axis = torch.randn(3, device=self.device)
        random_axis = random_axis / torch.norm(random_axis)
        # Use normal distribution for rotation angle
        rotation_angle = torch.randn(1, device=self.device).item() * rotation_std
        
        # Calculate center for rotation
        center = jittered_conf.calculate_geometric_center().to(self.device)
        
        # Apply rotation
        coord_manager.rotate_around_axis(random_axis, rotation_angle, center)
        
        return jittered_conf
    
    def random_jitter_batch(self, 
                           conformation: Coordinate, 
                           batch_size: int = 10, 
                           translation_std: float = 0.2, 
                           rotation_std: float = 2.0) -> List[Coordinate]:
        """
        Generate multiple jittered conformations in batch.
        
        Args:
            conformation (Coordinate): Original conformation
            batch_size (int): Number of jittered conformations to generate
            translation_std (float): Standard deviation for translation jitter (normal distribution) in Å
            rotation_std (float): Standard deviation for rotation jitter (normal distribution) in degrees
            
        Returns:
            List[Coordinate]: List of jittered conformations
        """
        jittered_confs = []
        
        for _ in range(batch_size):
            jittered_conf = self.random_jitter(conformation, translation_std, rotation_std)
            jittered_confs.append(jittered_conf)
        
        return jittered_confs
    
    def create_ligand_copy(self, ligand_coordinate: Coordinate) -> Coordinate:
        """
        Create a copy of the ligand coordinate.
        
        Args:
            ligand_coordinate (Coordinate): Original ligand coordinate
            
        Returns:
            Coordinate: Copy of the ligand coordinate
        """
        ligand_copy = Coordinate()
        ligand_copy.coordinates = ligand_coordinate.coordinates.detach().clone()
        return ligand_copy
    

    
    def calculate_center(self, coordinate: Coordinate, group_indices: List[int]) -> torch.Tensor:
        """
        Calculate the center of a residue group.
        
        Args:
            coordinate (Coordinate): Coordinate set
            group_indices (List[int]): Indices of atoms in the residue group
            
        Returns:
            torch.Tensor: Center coordinates of the residue group
        """
        if not group_indices:
            raise ValueError("Group indices list is empty")
        
        # Get coordinates of the residue group atoms
        group_coords = coordinate.coordinates[group_indices]
        
        # Calculate center
        return torch.mean(group_coords, dim=0)
