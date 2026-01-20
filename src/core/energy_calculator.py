#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Calculator Module

Handles all energy calculation logic for protein-protein docking, including:
- Van der Waals energy
- Electrostatic energy
- Solvent penalty
- Distance penalty
- Conformation scoring
"""

from typing import List, Tuple, Dict
import math
import random
import torch
from src.models.structure import Structure
from src.models.force_field import ForceField


class EnergyCalculator:
    """
    Energy calculator for protein-protein docking conformations.
    
    Attributes:
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
        force_field (ForceField): Force field parameters
        solvent_penalty_coeff (float): Solvent favorable coefficient
        distance_penalty_coeff (float): Distance penalty coefficient
        optimal_distance (float): Optimal center-center distance
    """
    def __init__(self, force_field: ForceField, device: torch.device = torch.device("cpu"),
                 solvent_penalty_coeff: float = 0.1,
                 distance_penalty_coeff: float = 0.5):
        """
        Initialize the energy calculator.
        
        Args:
            force_field (ForceField): Force field parameters
            device (torch.device, optional): Device for calculations
            solvent_penalty_coeff (float, optional): Solvent penalty coefficient
            distance_penalty_coeff (float, optional): Distance penalty coefficient
        """
        self.device = device
        self.force_field = force_field
        # Scoring parameters
        self.solvent_penalty_coeff = solvent_penalty_coeff  # kcal/mol per contact
        self.distance_penalty_coeff = distance_penalty_coeff  # kcal/mol per Å
    
    def _is_hydrogen(self, atom_name: str) -> bool:
        """
        Check if an atom is a hydrogen atom.
        
        Args:
            atom_name (str): Atom name
            
        Returns:
            bool: True if hydrogen atom, False otherwise
        """
        return atom_name.startswith('H')
    
    def _get_non_hydrogen_atoms(self, structure: Structure) -> Tuple[torch.Tensor, List[bool]]:
        """
        Get non-hydrogen atom coordinates and mask from a structure.
        
        Args:
            structure (Structure): Protein structure
            
        Returns:
            Tuple[torch.Tensor, List[bool]]: Non-hydrogen atom coordinates and mask
        """
        non_h_mask = [not self._is_hydrogen(atom.atom_name) for atom in structure.atoms]
        coords = structure.coordinates.to(self.device)
        non_h_coords = coords[non_h_mask]
        return non_h_coords, non_h_mask
    
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
    
    def _get_atom_types(self, atoms: List, mask: List[bool]) -> List[str]:
        """
        Get atom types for atoms that pass the mask.
        
        Args:
            atoms (List): List of Atom objects
            mask (List[bool]): Mask indicating which atoms to include
            
        Returns:
            List[str]: List of atom types
        """
        atom_types = []
        for atom, include in zip(atoms, mask):
            if include:
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
    
    def _get_atom_charges(self, atoms: List, indices: List[int]) -> torch.Tensor:
        """
        Get charges for atoms at specified indices.
        
        Args:
            atoms (List): List of Atom objects
            indices (List[int]): Indices of atoms to get charges for
            
        Returns:
            torch.Tensor: Tensor of atom charges
        """
        charges = []
        for i in indices:
            atom = atoms[i]
            # Try to get charge from atom object if available, otherwise use force field charge
            try:
                charge = float(atom.charge) if atom.charge else 0.0
            except (ValueError, AttributeError):
                charge = self.force_field.get_atom_charge(atom.res_name, atom.atom_name)
            charges.append(charge)
        return torch.tensor(charges, device=self.device, dtype=torch.float32)
    
    def calculate_vdw_energy(self, receptor: Structure, ligand: Structure) -> float:
        """
        Calculate van der Waals energy between receptor and ligand atoms using PyTorch.
        Skips hydrogen atoms to improve performance.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            
        Returns:
            float: Van der Waals energy in kcal/mol
        """
        # Get non-hydrogen atoms
        rec_coords_non_h, rec_non_h_mask = self._get_non_hydrogen_atoms(receptor)
        lig_coords_non_h, lig_non_h_mask = self._get_non_hydrogen_atoms(ligand)
        
        # If no non-hydrogen atoms, return 0 energy
        if len(rec_coords_non_h) == 0 or len(lig_coords_non_h) == 0:
            return 0.0
        
        # Calculate distances
        distances = self._calculate_atom_pair_distances(rec_coords_non_h, lig_coords_non_h)
        distances_nm = distances * 0.1  # Convert to nanometers
        
        # Get atom types and VDW parameters
        rec_atom_types = self._get_atom_types(receptor.atoms, rec_non_h_mask)
        lig_atom_types = self._get_atom_types(ligand.atoms, lig_non_h_mask)
        
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
        Skips hydrogen atoms to improve performance.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            cutoff (float, optional): Minimum distance cutoff for electrostatic interactions (Å)
            
        Returns:
            float: Electrostatic energy in kcal/mol
        """
        # Get non-hydrogen atoms
        rec_coords_non_h, rec_non_h_mask = self._get_non_hydrogen_atoms(receptor)
        lig_coords_non_h, lig_non_h_mask = self._get_non_hydrogen_atoms(ligand)
        
        # If no non-hydrogen atoms, return 0 energy
        if len(rec_coords_non_h) == 0 or len(lig_coords_non_h) == 0:
            return 0.0
        
        # Calculate distances
        distances = self._calculate_atom_pair_distances(rec_coords_non_h, lig_coords_non_h)
        
        # Add epsilon to prevent division by zero
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        distances = torch.max(distances, epsilon)
        
        # Apply cutoff
        cutoff_mask = distances > cutoff
        
        # Get charges for non-hydrogen atoms
        def get_charges(atoms, mask):
            charges = []
            for i, (atom, include) in enumerate(zip(atoms, mask)):
                if include:
                    charge = self.force_field.get_atom_charge(atom.res_name, atom.atom_name)
                    charges.append(charge)
            return charges
        
        rec_charges = torch.tensor(get_charges(receptor.atoms, rec_non_h_mask), 
                                  device=self.device, dtype=torch.float32).unsqueeze(1)
        lig_charges = torch.tensor(get_charges(ligand.atoms, lig_non_h_mask), 
                                  device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Calculate electrostatic energy
        k = 332.0  # Conversion factor (kcal·Å/(mol·e²))
        electrostatic_energy = k * rec_charges * lig_charges / distances
        
        # Apply cutoff mask and clip values
        electrostatic_energy = electrostatic_energy * cutoff_mask.float()
        electrostatic_energy = torch.clamp(electrostatic_energy, -10000.0, 10000.0)
        
        return float(torch.sum(electrostatic_energy).item())
    
    def calculate_solvent_penalty(self, receptor: Structure, ligand: Structure, 
                                 rec_group_coords: torch.Tensor, lig_group_coords: torch.Tensor) -> float:
        """
        Calculate solvent penalty based on atom contact surface area.
        The lower the solvent accessibility (more atom contacts), the higher the reward.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            rec_group_coords (torch.Tensor): Receptor residue group coordinates
            lig_group_coords (torch.Tensor): Ligand residue group coordinates
            
        Returns:
            float: Solvent penalty in kcal/mol (negative value indicates reward)
        """
        # Get non-hydrogen atoms and calculate distances
        rec_coords_non_h, _ = self._get_non_hydrogen_atoms(receptor)
        lig_coords_non_h, _ = self._get_non_hydrogen_atoms(ligand)
        
        if len(rec_coords_non_h) == 0 or len(lig_coords_non_h) == 0:
            return 0.0
        
        distances = self._calculate_atom_pair_distances(rec_coords_non_h, lig_coords_non_h)
        
        # Calculate residue group centers and distance
        rec_group_center = torch.mean(rec_group_coords, dim=0)
        lig_group_center = torch.mean(lig_group_coords, dim=0)
        group_distance = torch.norm(rec_group_center - lig_group_center).item()
        
        # Calculate contacts at different distance ranges
        thresholds = [5.0, 8.0, 12.0]  # Close, medium, far contacts
        weights = [2.0, 1.0, 0.5]  # Corresponding weights
        
        solvent_reward = 0.0
        for i, threshold in enumerate(thresholds):
            if i == 0:
                contacts = torch.sum(distances <= threshold).item()
            else:
                contacts = torch.sum((distances > thresholds[i-1]) & (distances <= threshold)).item()
            solvent_reward += contacts * self.solvent_penalty_coeff * weights[i]
        
        # Calculate distance-dependent solvent term
        distance_diff = group_distance - 15.0
        distance_dependent_coeff = 0.0 if distance_diff > 700 else 1.0 / (1.0 + math.exp(0.2 * distance_diff))
        distance_solvent_reward = distance_dependent_coeff * self.solvent_penalty_coeff * 20.0
        
        # Total solvent reward (negative value for favorable in score calculation)
        total_solvent_reward = solvent_reward + distance_solvent_reward
        
        return -float(total_solvent_reward)
    
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
        group_distance = torch.norm(rec_group_center - lig_group_center).item()
        
        # Reward closer distances - the closer, the higher the reward (more negative)
        # Use exponential function to create stronger rewards for very close distances
        # but still maintain a smooth gradient
        reward = -math.exp(-group_distance / 5.0) * self.distance_penalty_coeff * 10.0
        
        return float(reward)
    
    def calculate_attraction_potential(self, rec_group_center: torch.Tensor, 
                                      lig_group_center: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Calculate continuous attraction potential between residue groups and the corresponding force vector.
        Now always attracts ligand towards receptor regardless of distance.
        
        Args:
            rec_group_center (torch.Tensor): Receptor residue group center
            lig_group_center (torch.Tensor): Ligand residue group center
            
        Returns:
            Tuple[float, torch.Tensor]: 
                - Attraction potential energy in kcal/mol
                - Force vector acting on ligand (towards receptor) in kcal/(mol·Å)
        """
        # Parameters
        k_attraction = 1000.0  # Spring constant (kcal/mol·Å²)
        
        # Calculate distance and direction
        delta = rec_group_center - lig_group_center
        distance = torch.norm(delta).item()
        
        if distance < 1e-6:
            return 0.0, torch.zeros(3, device=self.device)
        
        direction = delta / distance
        
        # Calculate potential and force - always attract ligand towards receptor
        # Potential increases with distance
        attraction_potential = k_attraction * distance**2
        
        # Force always points towards receptor
        force_magnitude = -2 * k_attraction * distance
        
        force_vector = direction * force_magnitude
        
        return float(attraction_potential), force_vector
    
    def calculate_collision_force(self, rec_coords: torch.Tensor, lig_coords: torch.Tensor, 
                                 rec_atoms: List, lig_atoms: List) -> torch.Tensor:
        """
        Calculate collision reaction forces between atoms based on Lennard-Jones potential gradient.
        
        Args:
            rec_coords (torch.Tensor): Receptor atom coordinates
            lig_coords (torch.Tensor): Ligand atom coordinates
            rec_atoms (List): List of receptor Atom objects
            lig_atoms (List): List of ligand Atom objects
            
        Returns:
            torch.Tensor: Total collision force vector acting on ligand
        """
        # Create masks for non-hydrogen atoms
        rec_non_h_mask = torch.tensor([not self._is_hydrogen(atom.atom_name) for atom in rec_atoms], 
                                     device=self.device, dtype=torch.bool)
        lig_non_h_mask = torch.tensor([not self._is_hydrogen(atom.atom_name) for atom in lig_atoms], 
                                     device=self.device, dtype=torch.bool)
        
        # Apply masks
        rec_coords_non_h = rec_coords[rec_non_h_mask]
        lig_coords_non_h = lig_coords[lig_non_h_mask]
        
        if len(rec_coords_non_h) == 0 or len(lig_coords_non_h) == 0:
            return torch.zeros(3, device=self.device)
        
        # Calculate distances and direction vectors
        expanded_rec = rec_coords_non_h.unsqueeze(1)
        expanded_lig = lig_coords_non_h.unsqueeze(0)
        delta_vectors = expanded_rec - expanded_lig
        distances = torch.norm(delta_vectors, dim=2)
        
        # Add epsilon to prevent division by zero
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        distances_safe = torch.max(distances, epsilon)
        
        direction_vectors = delta_vectors / distances_safe.unsqueeze(2)
        
        # Get atom types and VDW parameters
        rec_atom_types = self._get_atom_types(rec_atoms, rec_non_h_mask.cpu().numpy())
        lig_atom_types = self._get_atom_types(lig_atoms, lig_non_h_mask.cpu().numpy())
        
        rec_sigmas, rec_epsilons = self._get_vdw_parameters(rec_atom_types)
        lig_sigmas, lig_epsilons = self._get_vdw_parameters(lig_atom_types)
        
        # Convert to tensors
        rec_sigmas = torch.tensor(rec_sigmas, dtype=torch.float32, device=self.device).unsqueeze(1)
        rec_epsilons = torch.tensor(rec_epsilons, dtype=torch.float32, device=self.device).unsqueeze(1)
        lig_sigmas = torch.tensor(lig_sigmas, dtype=torch.float32, device=self.device).unsqueeze(0)
        lig_epsilons = torch.tensor(lig_epsilons, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply mixing rules and convert units
        sigma_ij = 0.5 * (rec_sigmas + lig_sigmas)  # nm
        epsilon_ij = torch.sqrt(rec_epsilons * lig_epsilons)  # kJ/mol
        sigma_ij_angstrom = sigma_ij * 10.0  # Convert to Å
        
        # Calculate force
        sigma_over_r = sigma_ij_angstrom / distances_safe
        lj_force_magnitude = 24 * epsilon_ij * (2 * sigma_over_r**12 - sigma_over_r**6)  # kJ/(mol·Å)
        lj_force_magnitude = lj_force_magnitude * 0.239  # Convert to kcal/(mol·Å)
        
        lj_force_vectors = lj_force_magnitude.unsqueeze(2) * direction_vectors
        total_force = torch.sum(lj_force_vectors, dim=(0, 1))
        
        return total_force
    
    def calculate_total_force(self, receptor: Structure, ligand: Structure, 
                             rec_group_indices: List[int], lig_group_indices: List[int]) -> torch.Tensor:
        """
        Calculate total force acting on the ligand, including collision reaction forces
        and attraction potential forces.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            rec_group_indices (List[int]): Indices of receptor residue group atoms
            lig_group_indices (List[int]): Indices of ligand residue group atoms
            
        Returns:
            torch.Tensor: Total force vector acting on ligand
        """
        # Calculate residue group centers
        rec_coords = receptor.coordinates.to(self.device)
        lig_coords = ligand.coordinates.to(self.device)
        
        rec_group_coords = rec_coords[rec_group_indices]
        lig_group_coords = lig_coords[lig_group_indices]
        
        rec_group_center = torch.mean(rec_group_coords, dim=0)
        lig_group_center = torch.mean(lig_group_coords, dim=0)
        
        # Calculate forces
        collision_force = self.calculate_collision_force(rec_coords, lig_coords, 
                                                        receptor.atoms, ligand.atoms)
        
        _, attraction_force = self.calculate_attraction_potential(rec_group_center, lig_group_center)
        
        return collision_force + attraction_force
    
    def _verify_electrostatic_principles(self, receptor: Structure, ligand: Structure, 
                                        rec_group_indices: List[int], lig_group_indices: List[int]) -> None:
        """
        Verify that electrostatic energy calculations follow fundamental principles.
        """
        # Get coordinates and charges
        rec_coords = receptor.coordinates.to(self.device)[rec_group_indices]
        lig_coords = ligand.coordinates.to(self.device)[lig_group_indices]
        
        rec_charges = self._get_atom_charges(receptor.atoms, rec_group_indices)
        lig_charges = self._get_atom_charges(ligand.atoms, lig_group_indices)
        
        # Calculate distances
        distances = self._calculate_atom_pair_distances(rec_coords, lig_coords)
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        distances = torch.max(distances, epsilon)
        
        # Check sample pairs
        k = 332.0  # Coulomb constant
        sample_size = min(50, len(rec_group_indices) * len(lig_group_indices))
        
        for _ in range(sample_size):
            rec_idx = random.randint(0, len(rec_group_indices) - 1)
            lig_idx = random.randint(0, len(lig_group_indices) - 1)
            
            rec_charge = rec_charges[rec_idx].item()
            lig_charge = lig_charges[lig_idx].item()
            
            if rec_charge == 0.0 or lig_charge == 0.0:
                continue
            
            dist = distances[rec_idx, lig_idx].item()
            actual_energy = k * rec_charge * lig_charge / dist
            
            # Verify principles
            if (rec_charge * lig_charge) < 0:
                assert actual_energy < 0, f"Electrostatic principle violated: Opposite charges should attract"
            elif (rec_charge * lig_charge) > 0:
                assert actual_energy > 0, f"Electrostatic principle violated: Same charges should repel"
    
    def score_conformation(self, receptor: Structure, ligand: Structure, 
                          rec_group_indices: List[int], lig_group_indices: List[int]) -> Tuple[float, Dict[str, float]]:
        """
        Score a docking conformation based on electrostatic energy and distance.
        Rewards closer distances to promote better docking results.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            rec_group_indices (List[int]): Indices of receptor residue group atoms
            lig_group_indices (List[int]): Indices of ligand residue group atoms
            
        Returns:
            Tuple[float, Dict[str, float]]: Total score and detailed scores
        """
        # Calculate electrostatic energy
        electrostatic_energy = self.calculate_electrostatic_energy(receptor, ligand)
        
        # Calculate residue group centers
        rec_coords = receptor.coordinates.to(self.device)
        lig_coords = ligand.coordinates.to(self.device)
        
        rec_group_coords = rec_coords[rec_group_indices]
        lig_group_coords = lig_coords[lig_group_indices]
        
        rec_group_center = torch.mean(rec_group_coords, dim=0)
        lig_group_center = torch.mean(lig_group_coords, dim=0)
        
        # Calculate distance penalty (now includes rewards for closer distances)
        distance_penalty = self.calculate_distance_penalty(rec_group_center, lig_group_center)
        
        # Verify electrostatic principles
        self._verify_electrostatic_principles(receptor, ligand, rec_group_indices, lig_group_indices)
        
        # Calculate total score (electrostatic energy + distance penalty/reward)
        total_score = electrostatic_energy + distance_penalty
        
        # Create detailed score dictionary
        detailed_scores = {
            'electrostatic': electrostatic_energy,
            'distance': distance_penalty
        }
        
        return total_score, detailed_scores
    

