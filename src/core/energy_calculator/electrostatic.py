#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Electrostatic Energy Calculator Module

Handles electrostatic energy calculations for protein-protein docking conformations.
"""

import torch
from ...models.topology import Topology
from ...models.coordinate import Coordinate
from ...models.force_field import ForceField
from .utils import EnergyUtils


class ElectrostaticCalculator(EnergyUtils):
    """
    Electrostatic energy calculator for protein-protein docking.
    
    Attributes:
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
        force_field (ForceField): Force field parameters
    """
    def __init__(self, force_field: ForceField, device: torch.device = torch.device("cuda")):
        super().__init__(force_field, device)
    
    def cal_ele_energy(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_coordinates_batch: torch.Tensor, ligand_top: Topology, ligand_coord: Coordinate, cutoff: float = 1.2) -> torch.Tensor:
        """
        Calculate electrostatic energy for multiple ligand conformations in batch.
        Supports [batch, length, 3] input format for ligand coordinates.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_coordinates_batch (torch.Tensor): Batch of ligand coordinates with shape [batch, length, 3]
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates with pre-calculated charges
            cutoff (float, optional): Minimum distance cutoff for electrostatic interactions (Å)
            
        Returns:
            torch.Tensor: Electrostatic energies for each conformation in batch, shape [batch]
        """
        # Get receptor coordinates and charges
        rec_coords = receptor_coord.coordinates.to(self.device)
        rec_charges = receptor_coord.charges.to(self.device, dtype=torch.float32).unsqueeze(1)
        
        # Get ligand charges (same for all conformations in batch)
        lig_charges = ligand_coord.charges.to(self.device, dtype=torch.float32)
        
        # Calculate distances for batch
        # Shape: [batch, rec_atoms, lig_atoms]
        distances = self.calculate_atom_pair_distances(rec_coords, ligand_coordinates_batch)
        
        # Add epsilon to prevent division by zero
        epsilon = torch.tensor(0.001, device=self.device, dtype=torch.float32)
        distances = torch.max(distances, epsilon)
        
        # Apply cutoff
        cutoff_mask = distances > cutoff
        
        # Reshape charges for broadcasting
        # rec_charges: [rec_atoms, 1] -> [1, rec_atoms, 1]
        # lig_charges: [lig_atoms] -> [1, 1, lig_atoms]
        rec_charges = rec_charges.unsqueeze(0)
        lig_charges = lig_charges.unsqueeze(0).unsqueeze(0)
        
        # Calculate electrostatic energy
        k = 332.0  # Conversion factor (kcal·Å/(mol·e²))
        electrostatic_energy = k * rec_charges * lig_charges / distances
        
        # Apply cutoff mask and clip values
        electrostatic_energy = electrostatic_energy * cutoff_mask.float()
        electrostatic_energy = torch.clamp(electrostatic_energy, -10000.0, 10000.0)
        
        # Ensure no inf or nan values
        electrostatic_energy = torch.nan_to_num(electrostatic_energy, nan=0.0, posinf=10000.0, neginf=-10000.0)
        
        # Sum over atom pairs and return energies for each conformation
        total_energies = torch.sum(electrostatic_energy, dim=(1, 2))
        
        return total_energies
    
    def calculate_electrostatic_forces(self, receptor_top: Topology, receptor_coord: Coordinate, ligand_top: Topology, ligand_coord: Coordinate, cutoff: float = 1.2) -> torch.Tensor:
        """
        Calculate electrostatic forces between receptor and ligand atoms.
        
        Args:
            receptor_top (Topology): Receptor protein topology
            receptor_coord (Coordinate): Receptor protein coordinates
            ligand_top (Topology): Ligand protein topology
            ligand_coord (Coordinate): Ligand protein coordinates
            cutoff (float, optional): Minimum distance cutoff (Å)
            
        Returns:
            torch.Tensor: Forces on ligand atoms (shape: [num_ligand_atoms, 3])
        """
        rec_coords = receptor_coord.coordinates
        lig_coords = ligand_coord.coordinates
        
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
        
        # Get charges from pre-stored tensors
        rec_charges = receptor_coord.charges.to(self.device, dtype=torch.float32).unsqueeze(1)
        lig_charges = ligand_coord.charges.to(self.device, dtype=torch.float32).unsqueeze(0)
        
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
