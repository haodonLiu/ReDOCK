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
from src.utils.logger import Logger


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
        self.max_batch_size = 1000  # Maximum number of atoms per batch
        self.min_batch_size = 100  # Minimum number of atoms per batch
        # Initialize logger
        self.logger = Logger()
        # Get available GPU memory if using GPU
        self.available_memory = None
        if self.device.type == 'cuda':
            self.available_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.logger.info(f"Available GPU memory: {self.available_memory / 1024**3:.2f} GB")
    
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
    
    def _adjust_batch_size(self, num_atoms1: int, num_atoms2: int) -> int:
        """
        Adjust batch size dynamically based on available GPU memory and calculation requirements.
        
        Args:
            num_atoms1 (int): Number of atoms in the first structure
            num_atoms2 (int): Number of atoms in the second structure
            
        Returns:
            int: Adjusted batch size
        """
        if self.device.type != 'cuda' or self.available_memory is None:
            # If not using GPU or memory info not available, use default batch size
            return self.max_batch_size
        
        # Calculate estimated memory usage per batch
        # Each atom pair distance calculation requires (batch_size * num_atoms2) floats
        # For VDW energy, we also need storage for sigmas, epsilons, and energy values
        estimated_memory_per_batch = lambda batch_size: batch_size * num_atoms2 * 4 * 5  # 5 tensors per calculation
        
        # Start with max batch size and adjust down until it fits in memory
        batch_size = self.max_batch_size
        while batch_size >= self.min_batch_size:
            estimated_memory = estimated_memory_per_batch(batch_size)
            if estimated_memory < self.available_memory * 0.8:  # Leave 20% memory for other operations
                break
            batch_size = int(batch_size * 0.7)  # Reduce batch size by 30%
        
        # Ensure batch size is at least min_batch_size
        batch_size = max(batch_size, self.min_batch_size)
        
        return batch_size
    
    def calculate_vdw_energy(self, receptor: Structure, ligand: Structure) -> float:
        """
        Calculate van der Waals energy between receptor and ligand atoms using PyTorch with batch processing.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            
        Returns:
            float: Van der Waals energy in kcal/mol
        """
        # Get atom types and VDW parameters first
        rec_atom_types = self._get_atom_types(receptor.atoms)
        lig_atom_types = self._get_atom_types(ligand.atoms)
        
        rec_sigmas, rec_epsilons = self._get_vdw_parameters(rec_atom_types)
        lig_sigmas, lig_epsilons = self._get_vdw_parameters(lig_atom_types)
        
        # Convert to tensors and reshape for broadcasting
        # Use float32 to avoid overflow issues with float16
        rec_sigmas = torch.tensor(rec_sigmas, dtype=torch.float32, device=self.device)
        rec_epsilons = torch.tensor(rec_epsilons, dtype=torch.float32, device=self.device)
        lig_sigmas = torch.tensor(lig_sigmas, dtype=torch.float32, device=self.device)
        lig_epsilons = torch.tensor(lig_epsilons, dtype=torch.float32, device=self.device)
        
        # Calculate mixing rules for all atom pairs
        # We'll do this once upfront since it's not memory-intensive
        sigma_ij = 0.5 * (rec_sigmas.unsqueeze(1) + lig_sigmas.unsqueeze(0))  # Lorentz mixing rule
        epsilon_ij = torch.sqrt(rec_epsilons.unsqueeze(1) * lig_epsilons.unsqueeze(0))  # Berthelot mixing rule
        
        # Apply distance cutoff (10 Å = 1.0 nm) - beyond this, VDW interactions are negligible
        cutoff_nm = 1.0  # 10 Å
        min_distance_nm = 0.1  # 1 Å - avoid extreme values
        
        # Adjust batch size dynamically based on available memory
        num_rec_atoms = len(receptor.atoms)
        num_lig_atoms = len(ligand.atoms)
        batch_size = self._adjust_batch_size(num_rec_atoms, num_lig_atoms)
        self.logger.debug(f"Using batch size {batch_size} for VDW energy calculation")
        
        total_energy = 0.0
        
        # Process receptor atoms in batches
        for start_idx in range(0, num_rec_atoms, batch_size):
            # Calculate end index for this batch
            end_idx = min(start_idx + batch_size, num_rec_atoms)
            
            # Get batch of receptor coordinates
            rec_batch_coords = receptor.coordinates[start_idx:end_idx]
            
            # Calculate distances for this batch
            distances = self._calculate_atom_pair_distances(rec_batch_coords, ligand.coordinates)
            distances_nm = distances * 0.1  # Convert to nanometers
            
            # Apply cutoff mask
            cutoff_mask = distances_nm < cutoff_nm
            
            # Clamp distances to avoid division by zero and overflow
            clamped_distances = torch.clamp(distances_nm, min=min_distance_nm)
            
            # Get VDW parameters for this batch
            batch_sigma_ij = sigma_ij[start_idx:end_idx]
            batch_epsilon_ij = epsilon_ij[start_idx:end_idx]
            
            # Calculate LJ potential only for atoms within cutoff
            sigma_over_r = batch_sigma_ij / clamped_distances
            vdw_energy = 4 * batch_epsilon_ij * (sigma_over_r**12 - sigma_over_r**6)
            
            # Apply cutoff mask
            vdw_energy = vdw_energy * cutoff_mask.float()
            
            # Clip extreme values to prevent overflow
            vdw_energy = torch.clamp(vdw_energy, -10000.0, 10000.0)
            
            # Convert from kJ/mol to kcal/mol
            vdw_energy = vdw_energy * 0.239
            
            # Add to total energy
            batch_energy = torch.sum(vdw_energy).item()
            total_energy += batch_energy
            
            # Clear cache to free up memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return float(total_energy)
    
    def calculate_electrostatic_energy(self, receptor: Structure, ligand: Structure, cutoff: float = 1.2) -> float:
        """
        Calculate electrostatic energy between receptor and ligand atoms using PyTorch with batch processing.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            cutoff (float, optional): Minimum distance cutoff for electrostatic interactions (Å)
            
        Returns:
            float: Electrostatic energy in kcal/mol
        """
        # Get charges for atoms first
        def get_charges(atoms):
            charges = []
            for atom in atoms:
                charge = self.force_field.get_atom_charge(atom.res_name, atom.atom_name)
                charges.append(charge)
            return charges
        
        rec_charges = torch.tensor(get_charges(receptor.atoms), 
                                  device=self.device, dtype=torch.float32)
        lig_charges = torch.tensor(get_charges(ligand.atoms), 
                                  device=self.device, dtype=torch.float32)
        
        # Add epsilon to prevent division by zero
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        
        # Conversion factor (kcal·Å/(mol·e²))
        k = 332.0
        
        # Adjust batch size dynamically based on available memory
        num_rec_atoms = len(receptor.atoms)
        num_lig_atoms = len(ligand.atoms)
        batch_size = self._adjust_batch_size(num_rec_atoms, num_lig_atoms)
        self.logger.debug(f"Using batch size {batch_size} for electrostatic energy calculation")
        
        total_energy = 0.0
        
        # Process receptor atoms in batches
        for start_idx in range(0, num_rec_atoms, batch_size):
            # Calculate end index for this batch
            end_idx = min(start_idx + batch_size, num_rec_atoms)
            
            # Get batch of receptor coordinates and charges
            rec_batch_coords = receptor.coordinates[start_idx:end_idx]
            rec_batch_charges = rec_charges[start_idx:end_idx]
            
            # Calculate distances for this batch
            distances = self._calculate_atom_pair_distances(rec_batch_coords, ligand.coordinates)
            
            # Prevent division by zero
            distances = torch.max(distances, epsilon)
            
            # Apply cutoff
            cutoff_mask = distances > cutoff
            
            # Calculate electrostatic energy for this batch
            # Reshape charges for broadcasting
            rec_batch_charges_reshaped = rec_batch_charges.unsqueeze(1)
            lig_charges_reshaped = lig_charges.unsqueeze(0)
            
            electrostatic_energy = k * rec_batch_charges_reshaped * lig_charges_reshaped / distances
            
            # Apply cutoff mask and clip values
            electrostatic_energy = electrostatic_energy * cutoff_mask.float()
            electrostatic_energy = torch.clamp(electrostatic_energy, -10000.0, 10000.0)
            
            # Add to total energy
            batch_energy = torch.sum(electrostatic_energy).item()
            total_energy += batch_energy
            
            # Clear cache to free up memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return float(total_energy)
    
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
    

