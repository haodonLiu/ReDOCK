#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Energy Calculator Utility Module

Contains utility functions for energy calculations, including:
- Distance calculations
- Atom type handling
- Force field parameter utilities
- Numerical stability helpers
"""

import torch
from ...models.force_field import ForceField


class EnergyUtils:
    """
    Utility class for energy calculations.
    
    Attributes:
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
    """
    def __init__(self, device: torch.device = torch.device("cuda"), force_field: ForceField = None):
        self.device = device
        self.force_field = force_field
    
    def calculate_atom_pair_distances(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances between all atom pairs from receptor and batch of ligand coordinates.
        
        Args:
            coords1 (torch.Tensor): Receptor atom coordinates with shape [rec_atoms, 3]
            coords2_batch (torch.Tensor): Batch of ligand atom coordinates with shape [batch, lig_atoms, 3]
            
        Returns:
            torch.Tensor: Distance matrix with shape [batch, rec_atoms, lig_atoms]
        """
        # Expand receptor coordinates to match batch size
        # coords1: [rec_atoms, 3] -> [1, rec_atoms, 1, 3]
        expanded1 = coords1.unsqueeze(0).unsqueeze(2)
        # coords2_batch: [batch, lig_atoms, 3] -> [batch, 1, lig_atoms, 3]
        expanded2 = coords2.unsqueeze(1)
        
        # Calculate distances
        # Shape after subtraction: [batch, rec_atoms, lig_atoms, 3]
        # Shape after norm: [batch, rec_atoms, lig_atoms]
        return torch.norm(expanded1 - expanded2, dim=3)

    
