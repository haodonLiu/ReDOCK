#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Atom Data Model

Defines the PDBAtom class for storing atom structural information.
"""


class Atom:
    """
    Atom data class for storing individual atom structural information.
    
    Attributes:
        record_type (str): Record type (ATOM or HETATM)
        atom_serial (int): Atom serial number
        atom_name (str): Atom name
        alt_loc (str): Alternate location indicator
        res_name (str): Residue name
        chain_id (str): Chain identifier
        res_seq (int): Residue sequence number
        i_code (str): Insertion code
        occupancy (float): Occupancy
        b_factor (float): Temperature factor
        element (str): Element symbol
        charge (str): Charge
    """
    def __init__(self, record_type: str, atom_serial: int, atom_name: str, alt_loc: str,
                 res_name: str, chain_id: str, res_seq: int, i_code: str, occupancy: float,
                 b_factor: float, element: str, charge: str):
        self.record_type = record_type  # 1-6: Record type
        self.atom_serial = atom_serial  # 7-11: Atom serial number
        self.atom_name = atom_name      # 13-16: Atom name
        self.alt_loc = alt_loc          # 17: Alternate location indicator
        self.res_name = res_name        # 18-20: Residue name
        self.chain_id = chain_id        # 22: Chain identifier
        self.res_seq = res_seq          # 23-26: Residue sequence number
        self.i_code = i_code            # 27: Insertion code
        self.occupancy = occupancy      # 55-60: Occupancy
        self.b_factor = b_factor        # 61-66: Temperature factor
        self.element = element          # 77-78: Element symbol
        self.charge = charge            # 79-80: Charge

    def to_pdb_line(self, x: float, y: float, z: float, atom_serial: int) -> str:
        """
        Convert the atom to a PDB format line.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            atom_serial (int): Atom serial number
            
        Returns:
            str: PDB format line for the atom
        """
        # Format the atom name to fit in columns 13-16
        # Handle special cases for hydrogen atoms and others
        atom_name = self.atom_name
        if len(atom_name) == 1:
            # Single character atom name (e.g., C, N, O)
            formatted_atom_name = f" {atom_name}  "
        elif len(atom_name) == 2:
            # Two character atom name (e.g., CA, CB)
            formatted_atom_name = f" {atom_name} "
        elif len(atom_name) == 3:
            # Three character atom name (e.g., H1, H2)
            formatted_atom_name = f"{atom_name} "
        else:
            # Longer atom names (truncate if necessary)
            formatted_atom_name = atom_name[:4]
        
        # Format the residue name to fit in columns 18-20
        res_name = self.res_name[:3].ljust(3)
        
        # Format the chain ID to fit in column 22
        chain_id = self.chain_id[:1].ljust(1)
        
        # Format the residue sequence number to fit in columns 23-26
        res_seq = f"{self.res_seq:4d}"
        
        # Format the insertion code to fit in column 27
        i_code = self.i_code[:1].ljust(1)
        
        # Format the occupancy to fit in columns 55-60
        occupancy = f"{self.occupancy:6.2f}"
        
        # Format the temperature factor to fit in columns 61-66
        b_factor = f"{self.b_factor:6.2f}"
        
        # Format the element symbol to fit in columns 77-78
        element = self.element[:2].ljust(2)
        
        # Format the charge to fit in columns 79-80
        charge = self.charge[:2].ljust(2)
        
        # Create the PDB line
        pdb_line = (f"{self.record_type:<6}{atom_serial:5d} {formatted_atom_name}{self.alt_loc:1}{res_name:3} {chain_id:1}{res_seq:4}{i_code:1}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6}{b_factor:6}          {element:2}{charge:2}")
        
        # Ensure the line is exactly 80 characters long
        return pdb_line[:80].ljust(80)
    
    def __repr__(self) -> str:
        """String representation of the atom"""
        return f"PDBAtom({self.record_type} {self.atom_serial} {self.atom_name} {self.res_name} {self.chain_id}{self.res_seq})"

