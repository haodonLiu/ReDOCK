#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Van der Waals Energy Calculator Module

Handles van der Waals energy calculations for protein-protein docking conformations.
"""

from typing import List
import math
import torch
from ...models.topology import Topology
from ...models.coordinate import Coordinate
from ...models.force_field import ForceField
from .utils import EnergyUtils

class VDWCalculator(EnergyUtils):
    """
    Van der Waals energy calculator for protein-protein docking.
    """
    def __init__(self, force_field: ForceField, device: torch.device = torch.device("cpu")):
        super().__init__(force_field, device)
    
    def _get_atom_types(self, atoms: List) -> List[str]:
        """
        Get atom types for atoms.
        
        Args:
            atoms (List): List of Atom objects
            
        Returns:
            List[str]: List of atom types
        """
        atom_types = []
        for atom in atoms:
            residue_name = atom.res_name
            atom_name = atom.atom_name
            atom_type = self.force_field.get_atom_type_for_residue(residue_name, atom_name)
            atom_types.append(atom_type)
        return atom_types
    
    def _get_vdw_parameters(self, atom_types: List[str]) -> tuple[List[float], List[float]]:
        """
        Get VDW parameters for a list of atom types.
        
        Args:
            atom_types (List[str]): List of atom types
            
        Returns:
            Tuple[List[float], List[float]]: Lists of sigma and epsilon parameters
        """
        sigmas = []
        epsilons = []
        for atom_type in atom_types:
            sigma, epsilon = self.force_field.get_vdw_params(atom_type)
            sigmas.append(sigma)
            epsilons.append(epsilon)
        return sigmas, epsilons
    
    def calculate_collision_score(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_top: Topology, ligand_coord: Coordinate) -> float:
        """
        Calculate custom collision score between receptor and ligand atoms.
        This score is designed to avoid nan and inf values that can occur with VDW calculations.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            
        Returns:
            float: Collision score (lower is better)
        """
        # Calculate distances
        distances = self.calculate_atom_pair_distances(receptor_coord.coordinates, ligand_coord.coordinates)
        
        # Apply distance cutoff (1.2 Å) as requested
        cutoff = 1.2  # 1.2 Å
        cutoff_mask = distances < cutoff
        
        # Clamp distances to avoid division by zero
        min_distance = 0.1  # 0.1 Å - avoid extreme values
        clamped_distances = torch.clamp(distances, min=min_distance)
        
        # Calculate collision score using a simple inverse distance function
        # This function increases as atoms get closer, indicating higher collision
        collision_score = 1.0 / clamped_distances
        
        # Apply cutoff mask
        collision_score = collision_score * cutoff_mask.float()
        
        # Clip extreme values to prevent overflow
        collision_score = torch.clamp(collision_score, 0.0, 10000.0)
        
        # Ensure no inf or nan values
        collision_score = torch.nan_to_num(collision_score, nan=10000.0, posinf=10000.0, neginf=0.0)
        
        total_score = torch.sum(collision_score).item()
        
        # Ensure total score is finite
        if not math.isfinite(total_score):
            total_score = 10000.0
        
        return float(total_score)
    
    def calculate_vdw_energy(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_top: Topology, ligand_coord: Coordinate) -> float:
        """
        Calculate van der Waals energy between receptor and ligand atoms using PyTorch.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            
        Returns:
            float: Van der Waals energy in kcal/mol
        """
        # Calculate distances
        distances = self.calculate_atom_pair_distances(receptor_coord.coordinates, ligand_coord.coordinates)
        distances_nm = distances * 0.1  # Convert to nanometers
        
        # Get atom types from pre-stored lists in Topology objects
        rec_atom_types = receptor_top.atom_types
        lig_atom_types = ligand_top.atom_types
        
        rec_sigmas, rec_epsilons = self._get_vdw_parameters(rec_atom_types)
        lig_sigmas, lig_epsilons = self._get_vdw_parameters(lig_atom_types)
        
        # Convert to tensors and reshape for broadcasting
        # Use float32 to avoid overflow issues with float16
        rec_sigmas = torch.tensor(rec_sigmas, dtype=torch.float32, device=self.device).unsqueeze(1)
        rec_epsilons = torch.tensor(rec_epsilons, dtype=torch.float32, device=self.device).unsqueeze(1)
        lig_sigmas = torch.tensor(lig_sigmas, dtype=torch.float32, device=self.device).unsqueeze(0)
        lig_epsilons = torch.tensor(lig_epsilons, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply mixing rules
        sigma_ij = 0.5 * (rec_sigmas + lig_sigmas)  # Lorentz mixing rule
        epsilon_ij = torch.sqrt(rec_epsilons * lig_epsilons)  # Berthelot mixing rule
        
        # Apply distance cutoff (1.2 Å = 0.12 nm) as requested
        cutoff_nm = 0.12  # 1.2 Å
        cutoff_mask = distances_nm < cutoff_nm
        
        # Clamp distances to avoid division by zero and overflow
        min_distance_nm = 0.05  # 0.5 Å - avoid extreme values
        clamped_distances = torch.clamp(distances_nm, min=min_distance_nm)
        
        # Calculate LJ potential only for atoms within cutoff
        sigma_over_r = sigma_ij / clamped_distances
        vdw_energy = 4 * epsilon_ij * (sigma_over_r**12 - sigma_over_r**6)
        
        # Apply cutoff mask
        vdw_energy = vdw_energy * cutoff_mask.float()
        
        # Clip extreme values to prevent overflow
        vdw_energy = torch.clamp(vdw_energy, -10000.0, 10000.0)
        
        # Convert from kJ/mol to kcal/mol
        vdw_energy = vdw_energy * 0.239
        
        # Ensure no inf or nan values
        vdw_energy = torch.nan_to_num(vdw_energy, nan=10000.0, posinf=10000.0, neginf=-10000.0)
        
        total_energy = torch.sum(vdw_energy).item()
        
        # Ensure total energy is finite
        if not math.isfinite(total_energy):
            total_energy = 10000.0
        
        return float(total_energy)
