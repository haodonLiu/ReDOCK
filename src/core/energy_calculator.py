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
        self.max_batch_size = 10000  # Maximum number of atoms per batch
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
    
    def calculate_batch_energy(self, receptor: Structure, ligands: List[Structure], include_vdw: bool = True, include_electrostatic: bool = True) -> List[Dict[str, float]]:
        """
        Calculate energy for multiple ligand conformations in batch.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligands (List[Structure]): List of ligand protein structures
            include_vdw (bool, optional): Whether to include van der Waals energy
            include_electrostatic (bool, optional): Whether to include electrostatic energy
            
        Returns:
            List[Dict[str, float]]: List of energy dictionaries for each ligand conformation
        """
        results = []
        
        for ligand in ligands:
            energies = {
                'vdw': 0.0,
                'electrostatic': 0.0,
                'total': 0.0
            }
            
            if include_vdw:
                vdw_energy = self.calculate_vdw_energy(receptor, ligand)
                energies['vdw'] = vdw_energy
                energies['total'] += vdw_energy
            
            if include_electrostatic:
                electrostatic_energy = self.calculate_electrostatic_energy(receptor, ligand)
                energies['electrostatic'] = electrostatic_energy
                energies['total'] += electrostatic_energy
            
            results.append(energies)
        
        return results
    
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
        
        total_energy = torch.sum(vdw_energy).item()
        
        return float(total_energy)
    
    def calculate_vdw_forces(self, receptor: Structure, ligand: Structure) -> torch.Tensor:
        """
        Calculate van der Waals forces between receptor and ligand atoms.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            
        Returns:
            torch.Tensor: Forces on ligand atoms (shape: [num_ligand_atoms, 3])
        """
        # Calculate distances and distance vectors
        rec_coords = receptor.coordinates
        lig_coords = ligand.coordinates
        
        # Expand dimensions for broadcasting
        rec_coords_expanded = rec_coords.unsqueeze(1)  # [num_rec_atoms, 1, 3]
        lig_coords_expanded = lig_coords.unsqueeze(0)  # [1, num_lig_atoms, 3]
        
        # Calculate distance vectors (receptor -> ligand)
        distance_vectors = lig_coords_expanded - rec_coords_expanded  # [num_rec_atoms, num_lig_atoms, 3]
        distances = torch.norm(distance_vectors, dim=2)  # [num_rec_atoms, num_lig_atoms]
        distances_nm = distances * 0.1  # Convert to nanometers
        
        # Apply cutoff (1.2 Å = 0.12 nm)
        cutoff_nm = 0.12
        cutoff_mask = distances_nm < cutoff_nm
        
        # Get VDW parameters
        rec_atom_types = self._get_atom_types(receptor.atoms)
        lig_atom_types = self._get_atom_types(ligand.atoms)
        
        rec_sigmas, rec_epsilons = self._get_vdw_parameters(rec_atom_types)
        lig_sigmas, lig_epsilons = self._get_vdw_parameters(lig_atom_types)
        
        # Convert to tensors and reshape for broadcasting
        rec_sigmas = torch.tensor(rec_sigmas, dtype=torch.float32, device=self.device).unsqueeze(1)
        rec_epsilons = torch.tensor(rec_epsilons, dtype=torch.float32, device=self.device).unsqueeze(1)
        lig_sigmas = torch.tensor(lig_sigmas, dtype=torch.float32, device=self.device).unsqueeze(0)
        lig_epsilons = torch.tensor(lig_epsilons, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply mixing rules
        sigma_ij = 0.5 * (rec_sigmas + lig_sigmas)
        epsilon_ij = torch.sqrt(rec_epsilons * lig_epsilons)
        
        # Clamp distances to avoid division by zero
        min_distance_nm = 0.05
        clamped_distances = torch.clamp(distances_nm, min=min_distance_nm)
        
        # Calculate LJ force magnitude
        # F = -dU/dr
        # U = 4ε[(σ/r)^12 - (σ/r)^6]
        # F = 4ε[12(σ^12/r^13) - 6(σ^6/r^7)]
        sigma_over_r = sigma_ij / clamped_distances
        force_magnitude = 4 * epsilon_ij * (12 * (sigma_over_r ** 12) / clamped_distances - 6 * (sigma_over_r ** 6) / clamped_distances)
        
        # Apply cutoff mask
        force_magnitude = force_magnitude * cutoff_mask.float()
        
        # Convert to kcal/(mol·Å) from kJ/(mol·nm)
        force_magnitude = force_magnitude * 0.239 / 0.1  # kJ to kcal, nm to Å
        
        # Calculate force vectors
        # Normalize distance vectors (avoid division by zero)
        distances_safe = torch.clamp(distances, min=1e-6)
        normalized_distance_vectors = distance_vectors / distances_safe.unsqueeze(2)
        
        # Calculate forces on ligand atoms
        # Force on ligand atom = sum over all receptor atoms of (force_magnitude * normalized_distance_vector)
        # Note: Force direction is repulsive when atoms are too close, attractive when at optimal distance
        ligand_forces = torch.sum(force_magnitude.unsqueeze(2) * normalized_distance_vectors, dim=0)
        
        return ligand_forces
    
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
    
    def calculate_electrostatic_forces(self, receptor: Structure, ligand: Structure, cutoff: float = 1.2) -> torch.Tensor:
        """
        Calculate electrostatic forces between receptor and ligand atoms.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            cutoff (float, optional): Minimum distance cutoff (Å)
            
        Returns:
            torch.Tensor: Forces on ligand atoms (shape: [num_ligand_atoms, 3])
        """
        rec_coords = receptor.coordinates
        lig_coords = ligand.coordinates
        
        # Expand dimensions for broadcasting
        rec_coords_expanded = rec_coords.unsqueeze(1)  # [num_rec_atoms, 1, 3]
        lig_coords_expanded = lig_coords.unsqueeze(0)  # [1, num_lig_atoms, 3]
        
        # Calculate distance vectors and distances
        distance_vectors = lig_coords_expanded - rec_coords_expanded  # [num_rec_atoms, num_lig_atoms, 3]
        distances = torch.norm(distance_vectors, dim=2)  # [num_rec_atoms, num_lig_atoms]
        
        # Apply cutoff (only include distances > cutoff)
        cutoff_mask = distances > cutoff
        
        # Avoid division by zero
        epsilon = torch.tensor(0.001, device=self.device)
        distances = torch.max(distances, epsilon)
        
        # Get charges
        def get_charges(atoms):
            charges = []
            for atom in atoms:
                charge = self.force_field.get_atom_charge(atom.res_name, atom.atom_name)
                charges.append(charge)
            return charges
        
        rec_charges = torch.tensor(get_charges(receptor.atoms), device=self.device, dtype=torch.float32).unsqueeze(1)
        lig_charges = torch.tensor(get_charges(ligand.atoms), device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Calculate electrostatic force magnitude
        # F = -dU/dr
        # U = k * q1 * q2 / r
        # F = k * q1 * q2 / r^2
        k = 332.0  # kcal·Å/(mol·e²)
        force_magnitude = k * rec_charges * lig_charges / (distances ** 2)
        
        # Apply cutoff mask
        force_magnitude = force_magnitude * cutoff_mask.float()
        
        # Calculate force vectors
        normalized_distance_vectors = distance_vectors / distances.unsqueeze(2)
        electrostatic_forces = force_magnitude.unsqueeze(2) * normalized_distance_vectors
        
        # Sum forces from all receptor atoms for each ligand atom
        total_electrostatic_forces = torch.sum(electrostatic_forces, dim=0)
        
        return total_electrostatic_forces
    
    def calculate_total_forces(self, receptor: Structure, ligand: Structure, 
                              rec_group_indices: List[int], lig_group_indices: List[int],
                              bias_strength: float = 100.0) -> torch.Tensor:
        """
        Calculate total forces on ligand atoms, including only VDW and bias forces.
        Bias force increases with distance and is always stronger than VDW forces.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            rec_group_indices (List[int]): Receptor target residue group indices
            lig_group_indices (List[int]): Ligand target residue group indices
            bias_strength (float, optional): Base strength of bias force towards target residue group
            
        Returns:
            torch.Tensor: Total forces on ligand atoms (shape: [num_ligand_atoms, 3])
        """
        # Calculate VDW forces
        vdw_forces = self.calculate_vdw_forces(receptor, ligand)
        
        # Calculate bias force towards receptor target group
        # Get centers of target groups
        rec_group_center = torch.mean(receptor.coordinates[rec_group_indices], dim=0)
        lig_group_center = torch.mean(ligand.coordinates[lig_group_indices], dim=0)
        
        # Calculate current distance between residue groups
        current_distance = torch.norm(rec_group_center - lig_group_center).item()
        
        # Calculate bias direction (ligand group -> receptor group)
        bias_direction = rec_group_center - lig_group_center
        bias_direction = bias_direction / torch.norm(bias_direction)

        # Make bias force increase with distance - the farther away, the stronger the bias
        # Distance-based bias: bias strength = base_strength * (1 + current_distance / 10.0)
        distance_based_bias = bias_strength * current_distance * 4
        
        # Ensure bias force is always stronger than VDW forces
        effective_bias_strength = distance_based_bias**1.5 
        
        # Apply bias force to all ligand atoms (pointing towards receptor target group)
        bias_force = bias_direction * effective_bias_strength
        bias_forces = torch.ones(ligand.coordinates.shape[0], 3, device=self.device) * bias_force
        
        # Calculate total force - only VDW + bias (no electrostatic)
        total_forces = vdw_forces + bias_forces
        
        return total_forces
    
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
    

