#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Calculator Module

Handles all energy calculation logic for protein-protein docking, including:
- Van der Waals energy
- Electrostatic energy
- Distance penalty
- Conformation scoring
"""

from typing import List, Tuple, Dict
import torch
from src.models.structure import Structure
from src.models.force_field import ForceField


class EnergyCalculator:
    """
    Energy calculator for protein-protein docking conformations.
    
    Attributes:
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
        force_field (ForceField): Force field parameters
        distance_penalty_coeff (float): Distance penalty coefficient
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
    
    def _calculate_atom_pair_distances(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances between all atom pairs from two sets of coordinates.
        
        Args:
            coords1 (torch.Tensor): First set of atom coordinates
            coords2 (torch.Tensor): Second set of atom coordinates
            
        Returns:
            torch.Tensor: Distance matrix between atom pairs
        """
        expanded1 = coords1.unsqueeze(1)  # (N, 1, 3)
        expanded2 = coords2.unsqueeze(0)  # (1, M, 3)
        return torch.norm(expanded1 - expanded2, dim=2)  # (N, M)
    
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
    
    def _get_vdw_parameters(self, atom_types: List[str]) -> Tuple[List[float], List[float]]:
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
    
    def calculate_vdw_energy(self, receptor: Structure, ligand: Structure) -> float:
        """
        Calculate van der Waals energy between receptor and ligand atoms using PyTorch.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            
        Returns:
            float: Van der Waals energy in kcal/mol
        """
        # Calculate distances
        distances = self._calculate_atom_pair_distances(receptor.coordinates, ligand.coordinates)
        distances_nm = distances * 0.1  # Convert to nanometers
        
        # Get atom types and VDW parameters
        rec_atom_types = self._get_atom_types(receptor.atoms)
        lig_atom_types = self._get_atom_types(ligand.atoms)
        
        rec_sigmas, rec_epsilons = self._get_vdw_parameters(rec_atom_types)
        lig_sigmas, lig_epsilons = self._get_vdw_parameters(lig_atom_types)
        
        # Convert to tensors and reshape for broadcasting
        rec_sigmas = torch.tensor(rec_sigmas, dtype=torch.float16, device=self.device).unsqueeze(1)
        rec_epsilons = torch.tensor(rec_epsilons, dtype=torch.float16, device=self.device).unsqueeze(1)
        lig_sigmas = torch.tensor(lig_sigmas, dtype=torch.float16, device=self.device).unsqueeze(0)
        lig_epsilons = torch.tensor(lig_epsilons, dtype=torch.float16, device=self.device).unsqueeze(0)
        
        # Apply mixing rules
        sigma_ij = 0.5 * (rec_sigmas + lig_sigmas)  # Lorentz mixing rule
        epsilon_ij = torch.sqrt(rec_epsilons * lig_epsilons)  # Berthelot mixing rule
        
        # Calculate LJ potential
        sigma_over_r = sigma_ij / distances_nm
        vdw_energy = 4 * epsilon_ij * (sigma_over_r**12 - sigma_over_r**6)
        
        # Convert from kJ/mol to kcal/mol
        vdw_energy = vdw_energy * 0.239
        
        return float(torch.sum(vdw_energy).item())
    
    def calculate_electrostatic_energy(self, receptor: Structure, ligand: Structure, cutoff: float = 1.2) -> float:
        """
        Calculate electrostatic energy between receptor and ligand atoms using PyTorch.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            cutoff (float, optional): Minimum distance cutoff for electrostatic interactions (Å)
            
        Returns:
            float: Electrostatic energy in kcal/mol
        """
        # Calculate distances
        distances = self._calculate_atom_pair_distances(receptor.coordinates, ligand.coordinates)
        
        # Add epsilon to prevent division by zero
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        distances = torch.max(distances, epsilon)
        
        # Apply cutoff
        cutoff_mask = distances > cutoff
        
        # Get charges for atoms
        def get_charges(atoms):
            charges = []
            for atom in atoms:
                charge = self.force_field.get_atom_charge(atom.res_name, atom.atom_name)
                charges.append(charge)
            return charges
        
        rec_charges = torch.tensor(get_charges(receptor.atoms), 
                                  device=self.device, dtype=torch.float32).unsqueeze(1)
        lig_charges = torch.tensor(get_charges(ligand.atoms), 
                                  device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Calculate electrostatic energy
        k = 332.0  # Conversion factor (kcal·Å/(mol·e²))
        electrostatic_energy = k * rec_charges * lig_charges / distances
        
        # Apply cutoff mask and clip values
        electrostatic_energy = electrostatic_energy * cutoff_mask.float()
        electrostatic_energy = torch.clamp(electrostatic_energy, -10000.0, 10000.0)
        
        return float(torch.sum(electrostatic_energy).item())
    
    def calculate_distance_penalty(self, rec_group_center: torch.Tensor, 
                                  lig_group_center: torch.Tensor) -> float:
        """
        Calculate distance penalty based on residue group center distance.
        Rewards closer distances (more negative = better) to promote better docking results.
        
        Args:
            rec_group_center (torch.Tensor): Receptor residue group center
            lig_group_center (torch.Tensor): Ligand residue group center
            
        Returns:
            float: Distance penalty in kcal/mol (negative values are rewards)
        """
        import math
        group_distance = torch.norm(rec_group_center - lig_group_center).item()
        
        # Reward closer distances - the closer, the higher the reward (more negative)
        reward = -math.exp(-group_distance / 5.0) * self.distance_penalty_coeff * 10.0
        
        return float(reward)
    
    def score_conformation(self, receptor: Structure, ligand: Structure, 
                          rec_group_indices: List[int], lig_group_indices: List[int],
                          include_vdw: bool = False, include_distance: bool = False,
                          distance_weight: float = 1.0, charge_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Score a docking conformation based on specified energy terms.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            rec_group_indices (List[int]): Indices of receptor residue group atoms
            lig_group_indices (List[int]): Indices of ligand residue group atoms
            include_vdw (bool, optional): Whether to include van der Waals energy
            include_distance (bool, optional): Whether to include distance penalty/reward
            distance_weight (float, optional): Weight for distance term (only used if include_distance is True)
            charge_weight (float, optional): Weight for electrostatic term
            
        Returns:
            Tuple[float, Dict[str, float]]: Total score and detailed scores
        """
        # Calculate electrostatic energy
        electrostatic_energy = self.calculate_electrostatic_energy(receptor, ligand) * charge_weight
        
        # Calculate van der Waals energy if requested
        vdw_energy = 0.0
        if include_vdw:
            vdw_energy = self.calculate_vdw_energy(receptor, ligand)
        
        # Calculate residue group centers
        rec_coords = receptor.coordinates.to(self.device)
        lig_coords = ligand.coordinates.to(self.device)
        
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
    

