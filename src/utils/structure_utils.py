#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Utilities Module

Provides utility functions for structure-related operations that can be shared across multiple modules.
"""

from typing import List
from src.models.structure import Structure
from src.models.atom import PDBAtom
from src.utils.logger import Logger


def residues_to_atom_indices(structure: Structure, residue_ids: List[str], logger: Logger = None) -> List[int]:
    """
    Convert residue IDs to atom indices.
    
    Args:
        structure (Structure): Protein structure
        residue_ids (List[str]): List of residue IDs in format "chain:residue"
        logger (Logger, optional): Logger instance for error logging
        
    Returns:
        List[int]: List of atom indices
        
    Raises:
        ValueError: If no atoms found for the given residues
    """
    atom_indices = []
    
    for residue_id in residue_ids:
        try:
            chain_id, res_seq = residue_id.split(":")
            res_seq = int(res_seq)
            
            # Find all atoms in this residue
            for i, atom in enumerate(structure.atoms):
                if atom.chain_id == chain_id and atom.res_seq == res_seq:
                    atom_indices.append(i)
        except Exception as e:
            if logger:
                logger.error(f"Error processing residue ID {residue_id}: {e}")
    
    if not atom_indices:
        raise ValueError(f"No atoms found for residues: {residue_ids}")
    
    return atom_indices


def calculate_structure_center(structure: Structure) -> List[float]:
    """
    Calculate the geometric center of a structure.
    
    Args:
        structure (Structure): Protein structure
        
    Returns:
        List[float]: Center coordinates as [x, y, z]
    """
    center = structure.calculate_geometric_center()
    return [float(center[0]), float(center[1]), float(center[2])]
