#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Chain Data Model

Defines the Chain class for storing protein chain information.
"""

from typing import List, Optional
from .residue import Residue
from .atom import Atom
import torch


class Chain:
    """
    Chain data class for storing protein chain information.
    
    Attributes:
        chain_id (str): Chain identifier
        residues (dict): Dictionary of residues in the chain, keyed by residue sequence number
        atoms (list): List of all atoms in the chain
        coordinates (torch.Tensor): Tensor of all atom coordinates in the chain
        total_charge (float): Total charge of the chain
    """
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.residues = {}  # Key: residue sequence number, Value: Residue object
        self.atoms = []     # List of all atoms in the chain
        self.coordinates = torch.empty(0, 3, dtype=torch.float16)
        self.total_charge = 0.0

    def add_residue(self, residue: Residue) -> None:
        """
        Add a residue to the chain.
        
        Args:
            residue (Residue): Residue to add to the chain
        """
        self.residues[residue.res_seq] = residue
        
        # Update chain atoms and coordinates
        for atom in residue.atoms:
            self.atoms.append(atom)
        
        if residue.coordinates.shape[0] > 0:
            self.coordinates = torch.cat([self.coordinates, residue.coordinates], dim=0)
        
        # Update total charge
        self.total_charge += residue.total_charge
    

    
    def get_residue(self, res_seq: int) -> Optional[Residue]:
        """
        Get a residue by its sequence number.
        
        Args:
            res_seq (int): Residue sequence number
            
        Returns:
            Optional[Residue]: Residue object if found, None otherwise
        """
        return self.residues.get(res_seq)
    
    def get_residues(self) -> List[Residue]:
        """
        Get all residues in the chain, sorted by sequence number.
        
        Returns:
            List[Residue]: List of residues sorted by sequence number
        """
        return [self.residues[res_seq] for res_seq in sorted(self.residues.keys())]
    
    def get_atoms(self) -> List[Atom]:
        """
        Get all atoms in the chain.
        
        Returns:
            List[PDBAtom]: List of atoms in the chain
        """
        return self.atoms
    
    def calculate_geometric_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the chain.
        
        Returns:
            torch.Tensor: Geometric center coordinates as a tensor [x, y, z]
        """
        if self.coordinates.shape[0] == 0:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
        return torch.mean(self.coordinates, dim=0)
    
    def __repr__(self) -> str:
        """String representation of the chain"""
        return f"Chain(chain_id={self.chain_id}, residues={len(self.residues)}, atoms={len(self.atoms)}, total_charge={self.total_charge:.2f})"
