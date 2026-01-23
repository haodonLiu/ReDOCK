#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Calculator Core Module

Main energy calculator logic for protein-protein docking, including:
- Batch energy calculations
- Conformation scoring
- Distance penalty calculation
"""

from typing import List, Tuple, Dict
import torch
from ...models.topology import Topology
from ...models.coordinate import Coordinate
from ...models.force_field import ForceField
from ...utils.logger import Logger
from .vdw import VDWCalculator
from .electrostatic import ElectrostaticCalculator
from .utils import EnergyUtils


class EnergyCalculator:
    """
    Energy calculator for protein-protein docking conformations.
    
    Attributes:
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
        force_field (ForceField): Force field parameters
        distance_penalty_coeff (float): Distance penalty coefficient
        max_batch_size (int): Maximum batch size for atom pair calculations
        min_batch_size (int): Minimum batch size for atom pair calculations
        logger (Logger): Logger instance for debug logging
        vdw_calculator (VDWCalculator): Van der Waals energy calculator
        electrostatic_calculator (ElectrostaticCalculator): Electrostatic energy calculator
        force_calculator (ForceCalculator): Force calculator
        utils (EnergyUtils): Utility functions
    """
    def __init__(self, force_field: ForceField, device: torch.device = torch.device("cpu"),
                 distance_penalty_coeff: float = 0.5):
        """
        Initialize the energy calculator.
        
        Args:
            force_field (ForceField): Force field parameters
            device (torch.device, optional): Device for calculations
            distance_penalty_coeff (float, optional): Distance penalty coefficient
        """
        self.device = device
        self.force_field = force_field
        # Scoring parameters
        self.distance_penalty_coeff = distance_penalty_coeff  # kcal/mol per Å
        # Batch processing parameters
        self.max_batch_size = 10000  # Maximum number of atoms per batch
        self.min_batch_size = 100  # Minimum number of atoms per batch
        # Initialize logger
        self.logger = Logger()
        # Get available GPU memory if using GPU
        self.available_memory = None
        if self.device.type == 'cuda':
            self.available_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.logger.info(f"Available GPU memory: {self.available_memory / 1024**3:.2f} GB")
        
        # Initialize calculators
        self.vdw_calculator = VDWCalculator(force_field, device)
        self.electrostatic_calculator = ElectrostaticCalculator(force_field, device)
        self.utils = EnergyUtils(device)
    
    def calculate_batch_energy(self, receptor_top: Topology, receptor_coord: Coordinate, ligands: List[Tuple[Topology, Coordinate]], include_vdw: bool = True, include_electrostatic: bool = True) -> List[Dict[str, float]]:
        """
        Calculate energy for multiple ligand conformations in batch.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligands (List[Tuple[Topology, Coordinate]]): List of ligand protein structures as (topology, coordinate) tuples
            include_vdw (bool, optional): Whether to include van der Waals energy
            include_electrostatic (bool, optional): Whether to include electrostatic energy
            
        Returns:
            List[Dict[str, float]]: List of energy dictionaries for each ligand conformation
        """
        results = []
        
        for ligand_top, ligand_coord in ligands:
            energies = {
                'vdw': 0.0,
                'electrostatic': 0.0,
                'total': 0.0
            }
            
            if include_vdw:
                vdw_energy = self.vdw_calculator.calculate_vdw_energy(receptor_top, receptor_coord, ligand_top, ligand_coord)
                energies['vdw'] = vdw_energy
                energies['total'] += vdw_energy
            
            if include_electrostatic:
                # Extract ligand coordinates for batch processing
                ligand_coordinates = ligand_coord.coordinates.unsqueeze(0)  # Add batch dimension
                electrostatic_energy = self.electrostatic_calculator.cal_ele_energy(receptor_top, receptor_coord, ligand_coordinates, ligand_top, ligand_coord)
                energies['electrostatic'] = electrostatic_energy.item()
                energies['total'] += electrostatic_energy.item()
            
            results.append(energies)
        
        return results
    
    def calculate_distance_penalty(self, rec_group_center: torch.Tensor, 
                                  lig_group_center: torch.Tensor) -> float:
        """
        Calculate distance penalty based on residue group center distance.
        Aggressively rewards closer distances (more negative = better) to prioritize proximity over energy.
        
        Args:
            rec_group_center (torch.Tensor): Receptor residue group center
            lig_group_center (torch.Tensor): Ligand residue group center
            
        Returns:
            float: Distance penalty in kcal/mol (negative values are rewards)
        """
        import math
        group_distance = torch.norm(rec_group_center - lig_group_center).item()
        
        # Aggressively reward closer distances with stronger weight for proximity
        # Use a stronger exponential decay to make closer distances have much lower scores
        # This ensures proximity is prioritized over energy terms
        if group_distance < 2.0:
            # Very strong reward for extremely close distances
            reward = -100000.0 * (2.0 - group_distance)
        elif group_distance < 5.0:
            # Strong reward for close distances
            reward = -10000.0 * math.exp(-group_distance / 2.0)
        elif group_distance < 10.0:
            # Moderate reward for medium distances
            reward = -1000.0 * math.exp(-group_distance / 3.0)
        else:
            # Small reward for longer distances
            reward = -100.0 * math.exp(-group_distance / 5.0)
        
        return float(reward)
    
    def score_conformation(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_top: Topology, ligand_coord: Coordinate, 
                          rec_group_indices: List[int], lig_group_indices: List[int],
                          include_vdw: bool = False, include_distance: bool = False,
                          distance_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Score a docking conformation based on specified energy terms.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            rec_group_indices (List[int]): Indices of receptor residue group atoms
            lig_group_indices (List[int]): Indices of ligand residue group atoms
            include_vdw (bool, optional): Whether to include van der Waals energy
            include_distance (bool, optional): Whether to include distance penalty/reward
            distance_weight (float, optional): Weight for distance term (only used if include_distance is True)
            
        Returns:
            Tuple[float, Dict[str, float]]: Total score and detailed scores
        """
        # Calculate electrostatic energy
        ligand_coordinates = ligand_coord.coordinates.unsqueeze(0)  # Add batch dimension
        electrostatic_energy = self.electrostatic_calculator.cal_ele_energy(receptor_top, receptor_coord, ligand_coordinates, ligand_top, ligand_coord).item()
        
        # Calculate van der Waals energy if requested
        vdw_energy = 0.0
        if include_vdw:
            vdw_energy = self.vdw_calculator.calculate_vdw_energy(receptor_top, receptor_coord, ligand_top, ligand_coord)
        
        # Calculate residue group centers
        rec_coords = receptor_coord.coordinates.to(self.device)
        lig_coords = ligand_coord.coordinates.to(self.device)
        
        rec_group_coords = rec_coords[rec_group_indices]
        lig_group_coords = lig_coords[lig_group_indices]
        
        rec_group_center = torch.mean(rec_group_coords, dim=0)
        lig_group_center = torch.mean(lig_group_coords, dim=0)
        
        # Calculate distance penalty if requested
        distance_penalty = 0.0
        if include_distance:
            distance_penalty = self.calculate_distance_penalty(rec_group_center, lig_group_center) * distance_weight
        
        # Calculate total score
        total_score = electrostatic_energy + vdw_energy + distance_penalty
        
        # Create detailed score dictionary
        detailed_scores = {
            'electrostatic': electrostatic_energy,
            'vdw': vdw_energy,
            'distance': distance_penalty
        }
        
        return total_score, detailed_scores
    
    def calculate_collision_score(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_top: Topology, ligand_coord: Coordinate) -> float:
        """
        Calculate collision score based on van der Waals energy.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            
        Returns:
            float: Collision score (lower is better)
        """
        # Use van der Waals energy as collision score
        # High positive VDW energy indicates collisions
        return self.vdw_calculator.calculate_vdw_energy(receptor_top, receptor_coord, ligand_top, ligand_coord)
