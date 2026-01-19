#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Statistics Module

Generates statistical information about PDB structures.
"""

from typing import Dict, Any
from src.models.structure import Structure


class StructureStatistics:
    """
    Generates statistics for PDB structures.
    
    Attributes:
        structure (Structure): The structure to analyze
    """
    def __init__(self, structure: Structure):
        self.structure = structure
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the structure.
        
        Returns:
            Dict[str, Any]: Dictionary containing structure statistics
        """
        stats = {
            'total_atoms': self._count_total_atoms(),
            'atom_records': self._count_atom_records(),
            'hetatm_records': self._count_hetatm_records(),
            'other_records': self._count_other_records(),
            'chains': self._count_chains(),
            'residues': self._count_residues(),
            'errors': self._count_errors(),
            'warnings': self._count_warnings()
        }
        return stats
    
    def _count_total_atoms(self) -> int:
        """
        Count total number of atoms.
        
        Returns:
            int: Total atom count
        """
        return len(self.structure.atoms)
    
    def _count_atom_records(self) -> int:
        """
        Count number of ATOM records.
        
        Returns:
            int: ATOM record count
        """
        return sum(1 for atom in self.structure.atoms if atom.record_type == 'ATOM')
    
    def _count_hetatm_records(self) -> int:
        """
        Count number of HETATM records.
        
        Returns:
            int: HETATM record count
        """
        return sum(1 for atom in self.structure.atoms if atom.record_type == 'HETATM')
    
    def _count_other_records(self) -> int:
        """
        Count number of non-ATOM/HETATM records.
        
        Returns:
            int: Other record count
        """
        return len(self.structure.other_records)
    
    def _count_chains(self) -> int:
        """
        Count number of unique chains.
        
        Returns:
            int: Chain count
        """
        chains = set(atom.chain_id for atom in self.structure.atoms if atom.chain_id.strip())
        return len(chains)
    
    def _count_residues(self) -> int:
        """
        Count number of unique residues.
        
        Returns:
            int: Residue count
        """
        residues = set((atom.chain_id, atom.res_seq, atom.res_name) for atom in self.structure.atoms)
        return len(residues)
    
    def _count_errors(self) -> int:
        """
        Count number of errors.
        
        Returns:
            int: Error count
        """
        return len(self.structure.errors)
    
    def _count_warnings(self) -> int:
        """
        Count number of warnings.
        
        Returns:
            int: Warning count
        """
        return len(self.structure.warnings)
