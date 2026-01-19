#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Residue Data Model

Defines the Residue class for storing protein residue information.
"""

from typing import List, Optional
from .atom import PDBAtom
import torch


class Residue:
    """
    Residue data class for storing protein residue information.
    
    Attributes:
        res_name (str): Residue name
        res_seq (int): Residue sequence number
        chain_id (str): Chain identifier
        atoms (list): List of atoms in the residue
        coordinates (torch.Tensor): Tensor of all atom coordinates in the residue
        total_charge (float): Total charge of the residue
        i_code (str): Insertion code
    """
    def __init__(self, res_name: str, res_seq: int, chain_id: str, i_code: str = ' '):
        self.res_name = res_name
        self.res_seq = res_seq
        self.chain_id = chain_id
        self.i_code = i_code
        self.atoms = []     # List of all atoms in the residue
        self.coordinates = torch.empty(0, 3, dtype=torch.float16)
        self.total_charge = 0.0
    
    def add_atom(self, atom: PDBAtom, coordinates: torch.Tensor) -> None:
        """
        Add an atom to the residue.
        
        Args:
            atom (PDBAtom): Atom to add to the residue
            coordinates (torch.Tensor): Atom coordinates as a tensor
        """
        self.atoms.append(atom)
        
        # Update residue coordinates
        self.coordinates = torch.cat([self.coordinates, coordinates.unsqueeze(0)], dim=0)
        
        # Update total charge
        atom_charge = self._parse_atom_charge(atom.charge)
        self.total_charge += atom_charge
    
    def _parse_atom_charge(self, charge_str: str) -> float:
        """
        Parse atom charge from string to float.
        
        Args:
            charge_str (str): Charge string from PDB file
            
        Returns:
            float: Parsed charge value
        """
        if not charge_str:
            return 0.0
        
        try:
            # PDB charge format: e.g., "  +1", "  -2", "+3", "-1"
            # Remove whitespace and convert to float
            return float(charge_str.strip())
        except ValueError:
            # If parsing fails, try to interpret common charge formats
            charge_str = charge_str.strip()
            if charge_str == '+':
                return 1.0
            elif charge_str == '-':
                return -1.0
            elif charge_str == '++':
                return 2.0
            elif charge_str == '--':
                return -2.0
            return 0.0
    
    def get_atom(self, atom_name: str) -> Optional[PDBAtom]:
        """
        Get an atom by its name.
        
        Args:
            atom_name (str): Atom name
            
        Returns:
            Optional[PDBAtom]: Atom object if found, None otherwise
        """
        for atom in self.atoms:
            if atom.atom_name == atom_name:
                return atom
        return None
    
    def get_atoms(self) -> List[PDBAtom]:
        """
        Get all atoms in the residue.
        
        Returns:
            List[PDBAtom]: List of atoms in the residue
        """
        return self.atoms
    
    def calculate_geometric_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the residue.
        
        Returns:
            torch.Tensor: Geometric center coordinates as a tensor [x, y, z]
        """
        if self.coordinates.shape[0] == 0:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
        return torch.mean(self.coordinates, dim=0)
    
    def __repr__(self) -> str:
        """String representation of the residue"""
        return f"Residue(res_name={self.res_name}, res_seq={self.res_seq}, chain_id={self.chain_id}, atoms={len(self.atoms)}, total_charge={self.total_charge:.2f})"
