#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Structure Data Model

Defines the Structure class for storing PDB structure information including atoms,
coordinates, and other records.
"""

import math
import torch
from typing import List, Tuple, Dict, Any, Optional
from .atom import PDBAtom
from .chain import Chain
from .residue import Residue


class Structure:
    """
    Container class for storing PDB structure information.
    
    Attributes:
        atoms (List[PDBAtom]): List of PDBAtom objects
        coordinates (torch.Tensor): Tensor of 3D coordinates (shape: [N, 3])
        other_records (List[str]): List of non-ATOM/HETATM records
        errors (List[str]): List of error messages
        warnings (List[str]): List of warning messages
        chains (dict): Dictionary of chains in the structure, keyed by chain ID
        residues (dict): Dictionary of residues in the structure, keyed by (chain_id, res_seq)
        total_charge (float): Total charge of the structure
    """
    def __init__(self):
        self.atoms: List[PDBAtom] = []  # Store atom structure information
        self.coordinates: torch.Tensor = torch.empty(0, 3, dtype=torch.float16)  # Store 3D coordinates as tensor
        self.other_records: List[str] = []  # Store non-ATOM/HETATM records
        self.errors: List[str] = []  # Store error messages
        self.warnings: List[str] = []  # Store warning messages
        self.chains: Dict[str, Chain] = {}  # Key: chain_id, Value: Chain object
        self.residues: Dict[Tuple[str, int], Residue] = {}  # Key: (chain_id, res_seq), Value: Residue object
        self.total_charge: float = 0.0
        self.bonds: List[Tuple[int, int]] = []  # Store bonds as tuples of atom indices
        self.angles: List[Tuple[int, int, int]] = []  # Store angles as tuples of atom indices
        self.dihedrals: List[Tuple[int, int, int, int]] = []  # Store dihedrals as tuples of atom indices

    def add_atom(self, atom: PDBAtom, coordinates: Tuple[float, float, float]) -> None:
        """
        Add an atom and its coordinates to the structure.
        
        Args:
            atom (PDBAtom): Atom object to add
            coordinates (Tuple[float, float, float]): 3D coordinates of the atom
        """
        self.atoms.append(atom)
        # Convert coordinates to tensor and append to existing tensor
        coord_tensor = torch.tensor(coordinates, dtype=torch.float16).unsqueeze(0)
        self.coordinates = torch.cat([self.coordinates, coord_tensor], dim=0)

    def add_other_record(self, record: str) -> None:
        """
        Add a non-ATOM/HETATM record to the structure.
        
        Args:
            record (str): Record string to add
        """
        self.other_records.append(record)

    def add_error(self, error: str) -> None:
        """
        Add an error message to the structure.
        
        Args:
            error (str): Error message to add
        """
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """
        Add a warning message to the structure.
        
        Args:
            warning (str): Warning message to add
        """
        self.warnings.append(warning)

    def clear(self) -> None:
        """
        Clear all data from the structure.
        """
        self.atoms.clear()
        self.coordinates = torch.empty(0, 3, dtype=torch.float16)
        self.other_records.clear()
        self.errors.clear()
        self.warnings.clear()
        self.chains.clear()
        self.residues.clear()
        self.total_charge = 0.0

    def get_atom_count(self) -> int:
        """
        Get the total number of atoms in the structure.
        
        Returns:
            int: Total number of atoms
        """
        return len(self.atoms)

    def get_atom_by_index(self, index: int) -> PDBAtom:
        """
        Get an atom by its index.
        
        Args:
            index (int): Index of the atom
            
        Returns:
            PDBAtom: Atom object at the specified index
        """
        return self.atoms[index]

    def get_coordinates_by_index(self, index: int) -> Tuple[float, float, float]:
        """
        Get coordinates by atom index.
        
        Args:
            index (int): Index of the atom
            
        Returns:
            Tuple[float, float, float]: Coordinates of the atom
        """
        coord = self.coordinates[index]
        return (float(coord[0]), float(coord[1]), float(coord[2]))

    def calculate_geometric_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the structure.
        
        Returns:
            torch.Tensor: Geometric center coordinates as a tensor [x, y, z]
        """
        if self.coordinates.shape[0] == 0:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
        return torch.mean(self.coordinates, dim=0)

    def build_hierarchy(self) -> None:
        """
        Build the hierarchy of the structure: structure -> chains -> residues -> atoms.
        """
        # Clear existing hierarchy
        self.chains.clear()
        self.residues.clear()
        self.total_charge = 0.0
        
        # Group atoms by (chain_id, res_seq)
        residue_groups: Dict[Tuple[str, int], List[Tuple[PDBAtom, torch.Tensor]]] = {}
        
        for i, atom in enumerate(self.atoms):
            key = (atom.chain_id, atom.res_seq)
            if key not in residue_groups:
                residue_groups[key] = []
            residue_groups[key].append((atom, self.coordinates[i]))
        
        # Build residues and chains
        for (chain_id, res_seq), atom_list in residue_groups.items():
            # Get residue information from the first atom in the group
            first_atom, _ = atom_list[0]
            residue = Residue(first_atom.res_name, res_seq, chain_id, first_atom.i_code)
            
            # Add all atoms to the residue, including HETATM
            for atom, coord in atom_list:
                residue.add_atom(atom, coord)
            
            # Add residue to structure residues dict
            self.residues[(chain_id, res_seq)] = residue
            
            # Add residue to chain, regardless of whether it's HETATM or not
            if chain_id not in self.chains:
                self.chains[chain_id] = Chain(chain_id)
            self.chains[chain_id].add_residue(residue)
        
        # Calculate total structure charge
        for chain in self.chains.values():
            self.total_charge += chain.total_charge
        
        # Skip bond detection as we're using OpenMM which handles bonding automatically
        # This significantly improves performance for large structures
        self.bonds = []
        self.angles = []
        self.dihedrals = []
    
    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """
        Get a chain by its chain ID.
        
        Args:
            chain_id (str): Chain identifier
            
        Returns:
            Optional[Chain]: Chain object if found, None otherwise
        """
        return self.chains.get(chain_id)
    
    def get_residue(self, chain_id: str, res_seq: int) -> Optional[Residue]:
        """
        Get a residue by its chain ID and sequence number.
        
        Args:
            chain_id (str): Chain identifier
            res_seq (int): Residue sequence number
            
        Returns:
            Optional[Residue]: Residue object if found, None otherwise
        """
        return self.residues.get((chain_id, res_seq))
    
    def get_chains(self) -> List[Chain]:
        """
        Get all chains in the structure, sorted by chain ID.
        
        Returns:
            List[Chain]: List of chains sorted by chain ID
        """
        return [self.chains[chain_id] for chain_id in sorted(self.chains.keys())]
    
    def get_residues_by_chain(self, chain_id: str) -> List[Residue]:
        """
        Get all residues in a specific chain, sorted by sequence number.
        
        Args:
            chain_id (str): Chain identifier
            
        Returns:
            List[Residue]: List of residues sorted by sequence number
        """
        chain = self.get_chain(chain_id)
        if chain:
            return chain.get_residues()
        return []
    
    def extract_chains(self, chain_ids: List[str]) -> 'Structure':
        """
        Extract specified chains from the structure into a new Structure object.
        
        Args:
            chain_ids (List[str]): List of chain IDs to extract
            
        Returns:
            Structure: New Structure object containing only the specified chains
        """
        # Create a new structure
        extracted = Structure()
        
        # Copy other records
        extracted.other_records = self.other_records.copy()
        
        # Extract atoms and coordinates for the specified chains
        for i, atom in enumerate(self.atoms):
            # For ATOM records, include only those with matching chain_id
            # For HETATM records, include all of them (they should move with the protein)
            if atom.record_type == 'ATOM':
                if atom.chain_id in chain_ids:
                    extracted.add_atom(atom, self.get_coordinates_by_index(i))
            else:  # HETATM records
                extracted.add_atom(atom, self.get_coordinates_by_index(i))
        
        # Build hierarchy for the extracted structure
        extracted.build_hierarchy()
        
        return extracted
    
    def align_to_standard_coordinate_system(self) -> None:
        """
        Align the structure to the standard coordinate system:
        1. Translate to center at origin
        """
        if self.coordinates.shape[0] == 0:
            return
        
        # Translate to origin
        center = self.calculate_geometric_center()
        self.coordinates -= center
    
    def detect_bonds(self, bond_length_cutoff: float = 1.8) -> None:
        """
        Detect bonds, angles, and dihedrals in the structure based on distance criteria.
        
        Args:
            bond_length_cutoff (float): Maximum distance for considering two atoms as bonded (in Å)
        """
        if self.coordinates.shape[0] == 0:
            return
        
        # Clear existing bond information
        self.bonds.clear()
        self.angles.clear()
        self.dihedrals.clear()
        
        # Detect bonds based on distance
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                # Calculate distance between atoms i and j
                dist = torch.norm(self.coordinates[i] - self.coordinates[j])
                if dist <= bond_length_cutoff:
                    # Add bond as tuple of atom indices (sorted for consistency)
                    self.bonds.append((i, j))
        
        # Detect angles based on bonds
        # For each bond (A-B), find other bonds (B-C) to form angle (A-B-C)
        bond_dict = {}  # Key: atom index, Value: list of bonded atom indices
        for bond in self.bonds:
            a, b = bond
            if a not in bond_dict:
                bond_dict[a] = []
            if b not in bond_dict:
                bond_dict[b] = []
            bond_dict[a].append(b)
            bond_dict[b].append(a)
        
        for bond in self.bonds:
            a, b = bond
            # For each atom c bonded to b (and c != a), form angle (a-b-c)
            if b in bond_dict:
                for c in bond_dict[b]:
                    if c != a:
                        self.angles.append((a, b, c))
        
        # Detect dihedrals based on angles
        # For each angle (A-B-C), find other bonds (C-D) to form dihedral (A-B-C-D)
        for angle in self.angles:
            a, b, c = angle
            # For each atom d bonded to c (and d != b), form dihedral (a-b-c-d)
            if c in bond_dict:
                for d in bond_dict[c]:
                    if d != b:
                        self.dihedrals.append((a, b, c, d))
    
    def __repr__(self) -> str:
        """String representation of the structure"""
        return f"Structure(atoms={len(self.atoms)}, coordinates={tuple(self.coordinates.shape)}, chains={len(self.chains)}, residues={len(self.residues)}, bonds={len(self.bonds)}, angles={len(self.angles)}, dihedrals={len(self.dihedrals)}, total_charge={self.total_charge:.2f})"
