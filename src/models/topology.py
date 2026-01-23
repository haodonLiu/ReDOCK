#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topology Data Model

Defines the Topology class for storing protein topology information including atoms,
chains, residues, and other structural information.
"""

from typing import List, Tuple, Dict, Any, Optional
from .structure.atom import Atom
from .structure.chain import Chain
from .structure.residue import Residue


class Topology:
    """
    Container class for storing protein topology information.
    
    Attributes:
        atoms (List[Atom]): List of Atom objects
        other_records (List[str]): List of non-ATOM/HETATM records
        chains (dict): Dictionary of chains in the structure, keyed by chain ID
        residues (dict): Dictionary of residues in the structure, keyed by (chain_id, res_seq)
        total_charge (float): Total charge of the structure
        atom_types (List[str]): Store atom types for force field parameters
        errors (List[str]): List of error messages
        warnings (List[str]): List of warning messages
    """
    def __init__(self):
        self.atoms: List[Atom] = []  # Store atom structure information
        self.other_records: List[str] = []  # Store non-ATOM/HETATM records
        self.chains: Dict[str, Chain] = {}  # Key: chain_id, Value: Chain object
        self.residues: Dict[Tuple[str, int], Residue] = {}  # Key: (chain_id, res_seq), Value: Residue object
        self.total_charge: float = 0.0
        self.atom_types: List[str] = []  # Store atom types for force field parameters
        self.errors: List[str] = []  # Store error messages
        self.warnings: List[str] = []  # Store warning messages

    def add_atom(self, atom: Atom) -> None:
        """
        Add an atom to the topology.
        
        Args:
            atom (Atom): Atom object to add
        """
        self.atoms.append(atom)
        # Initialize atom type for new atom (will be updated later)
        self.atom_types.append("")

    def add_other_record(self, record: str) -> None:
        """
        Add a non-ATOM/HETATM record to the topology.
        
        Args:
            record (str): Record string to add
        """
        self.other_records.append(record)

    def add_error(self, error: str) -> None:
        """
        Add an error message to the topology.
        
        Args:
            error (str): Error message to add
        """
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """
        Add a warning message to the topology.
        
        Args:
            warning (str): Warning message to add
        """
        self.warnings.append(warning)


    
    def get_atom_count(self) -> int:
        """
        Get the total number of atoms in the topology.
        
        Returns:
            int: Total number of atoms
        """
        return len(self.atoms)
    
    def get_atom_by_index(self, index: int):
        """
        Get an atom by its index.
        
        Args:
            index (int): Index of the atom
            
        Returns:
            PDBAtom: Atom object at the specified index
        """
        return self.atoms[index]

    def build_hierarchy(self) -> None:
        """
        Build the hierarchy of the topology: topology -> chains -> residues -> atoms.
        """
        # Clear existing hierarchy
        self.chains.clear()
        self.residues.clear()
        self.total_charge = 0.0
        
        # Group atoms by (chain_id, res_seq)
        residue_groups: Dict[Tuple[str, int], List[Tuple[Atom, int]]] = {}
        
        for i, atom in enumerate(self.atoms):
            key = (atom.chain_id, atom.res_seq)
            if key not in residue_groups:
                residue_groups[key] = []
            residue_groups[key].append((atom, i))
        
        # Build residues and chains
        for (chain_id, res_seq), atom_list in residue_groups.items():
            # Get residue information from the first atom in the group
            first_atom, _ = atom_list[0]
            residue = Residue(first_atom.res_name, res_seq, chain_id, first_atom.i_code)
            
            # Add all atoms to the residue
            for atom, _ in atom_list:
                # Create a dummy coordinate tensor for residue
                import torch
                dummy_coord = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float16)
                residue.add_atom(atom, dummy_coord)
            
            # Add residue to topology residues dict
            self.residues[(chain_id, res_seq)] = residue
            
            # Add residue to chain
            if chain_id not in self.chains:
                self.chains[chain_id] = Chain(chain_id)
            self.chains[chain_id].add_residue(residue)
        
        # Calculate total structure charge
        for chain in self.chains.values():
            self.total_charge += chain.total_charge

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
        Get all chains in the topology, sorted by chain ID.
        
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



    def to_pdb_lines(self, coordinates) -> List[str]:
        """
        Convert the topology to PDB format lines using provided coordinates.
        
        Args:
            coordinates: Coordinate object with get_coordinates_by_index method
            
        Returns:
            List[str]: List of PDB format lines
        """
        pdb_lines = []
        
        # Add other records except END records
        for record in self.other_records:
            if not record.startswith('END'):
                pdb_lines.append(record)
        
        # Add atom records and TER records per chain
        if self.chains:
            # Get chain IDs in order
            chain_ids = sorted(self.chains.keys())
            atom_index = 1
            
            for chain_id in chain_ids:
                # Add atoms for this chain
                chain_atoms = []
                for i, atom in enumerate(self.atoms):
                    if atom.chain_id == chain_id:
                        chain_atoms.append(atom)
                
                # Add each atom in the chain
                for atom in chain_atoms:
                    # Find the index of this atom
                    for i, a in enumerate(self.atoms):
                        if a == atom:
                            x, y, z = coordinates.get_coordinates_by_index(i)
                            break
                    
                    # Create PDB line
                    pdb_line = atom.to_pdb_line(x, y, z, atom_index)
                    pdb_lines.append(pdb_line)
                    atom_index += 1
                
                # Add TER record after each chain
                if chain_atoms:
                    last_atom = chain_atoms[-1]
                    # Find the index of this atom
                    for i, a in enumerate(self.atoms):
                        if a == last_atom:
                            x, y, z = coordinates.get_coordinates_by_index(i)
                            break
                    
                    # Create TER record
                    ter_line = f"TER   {atom_index:5d}      {last_atom.res_name:3} {chain_id:1}{last_atom.res_seq:4d}          {x:8.3f}{y:8.3f}{z:8.3f}"
                    pdb_lines.append(ter_line)
                    atom_index += 1
        else:
            # No chains, just add all atoms
            for i, atom in enumerate(self.atoms):
                # Get coordinates
                x, y, z = coordinates.get_coordinates_by_index(i)
                
                # Create PDB line
                pdb_line = atom.to_pdb_line(x, y, z, i+1)
                pdb_lines.append(pdb_line)
        
        # Add END record at the end
        pdb_lines.append("END")
        
        return pdb_lines

    def __repr__(self) -> str:
        """String representation of the topology"""
        return f"Topology(atoms={len(self.atoms)}, chains={len(self.chains)}, residues={len(self.residues)}, total_charge={self.total_charge:.2f})"
