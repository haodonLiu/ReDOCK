#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Common Utilities

Provides common utility functions for the PDB processor.
"""

from typing import Tuple, Optional


# PDB format constants
PDB_FORMAT_CONSTANTS = {
    "RECORD_TYPE_WIDTH": 6,
    "ATOM_SERIAL_WIDTH": 5,
    "ATOM_NAME_WIDTH": 4,
    "ALT_LOC_WIDTH": 1,
    "RES_NAME_WIDTH": 3,
    "CHAIN_ID_WIDTH": 1,
    "RES_SEQ_WIDTH": 4,
    "I_CODE_WIDTH": 1,
    "COORDINATE_WIDTH": 8,
    "OCCUPANCY_WIDTH": 6,
    "B_FACTOR_WIDTH": 6,
    "ELEMENT_WIDTH": 2,
    "CHARGE_WIDTH": 2
}


def is_valid_coordinate(x: float, y: float, z: float) -> bool:
    """
    Check if coordinates are within valid range for PDB structures.
    
    Args:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        
    Returns:
        bool: True if coordinates are valid, False otherwise
    """
    # Typical PDB coordinate range is -1000 to 1000
    valid_range = (-1000.0, 1000.0)
    return (valid_range[0] <= x <= valid_range[1] and
            valid_range[0] <= y <= valid_range[1] and
            valid_range[0] <= z <= valid_range[1])


def format_pdb_line(record_type: str, atom_serial: int, atom_name: str, alt_loc: str,
                    res_name: str, chain_id: str, res_seq: int, i_code: str,
                    x: float, y: float, z: float, occupancy: float, b_factor: float,
                    element: str, charge: str) -> str:
    """
    Format a PDB ATOM/HETATM line according to the PDB specification.
    
    Args:
        record_type (str): Record type (ATOM or HETATM)
        atom_serial (int): Atom serial number
        atom_name (str): Atom name
        alt_loc (str): Alternate location indicator
        res_name (str): Residue name
        chain_id (str): Chain identifier
        res_seq (int): Residue sequence number
        i_code (str): Insertion code
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        occupancy (float): Occupancy
        b_factor (float): Temperature factor
        element (str): Element symbol
        charge (str): Charge
        
    Returns:
        str: Formatted PDB line
    """
    # Ensure record type is exactly 6 characters, left-aligned
    record = f"{record_type[:6]:<6}"
    
    # Ensure atom name is formatted correctly with proper alignment
    # PDB atom name formatting rules:
    # - 4 characters total
    # - Right-aligned for atoms starting with a letter (e.g., N, CA, CB)
    # - Left-aligned for atoms starting with a number (e.g., 1H, 2H)
    atom_name = atom_name.strip()
    if len(atom_name) <= 4:
        if atom_name and not atom_name[0].isdigit():
            # Atom starts with letter - right align
            formatted_atom_name = f"{atom_name:>4}"
        else:
            # Atom starts with number - left align
            formatted_atom_name = f"{atom_name:<4}"
    else:
        # Longer atom names - truncate to 4 characters
        formatted_atom_name = atom_name[:4]
    
    # Ensure residue name is exactly 3 characters (PDB format only allows 3 characters for residue names)
    formatted_res_name = f"{res_name[:3]:>3}"
    
    # Ensure alternate location is exactly 1 character (space if none)
    formatted_alt_loc = alt_loc[:1] if alt_loc else ' '
    
    # Ensure chain ID is exactly 1 character (space if none)
    formatted_chain_id = chain_id[:1] if chain_id else ' '
    
    # Ensure insertion code is exactly 1 character (space if none)
    formatted_i_code = i_code[:1] if i_code else ' '
    
    # Format coordinates according to PDB specification
    # Must be exactly 8 characters with 3 decimal places, right-aligned
    def format_coord(coord):
        # Format as right-aligned 8 characters with 3 decimal places
        # This ensures consistent formatting regardless of coordinate magnitude
        return f"{coord:>8.3f}"
    
    # Format occupancy and B-factor as right-aligned 6 characters with 2 decimal places
    def format_occ_bfac(value):
        return f"{value:>6.2f}"
    
    # Format element as right-aligned 2 characters
    formatted_element = f"{element[:2]:>2}"
    
    # Format charge as right-aligned 2 characters
    formatted_charge = f"{charge[:2]:>2}"
    
    # Build the line strictly according to PDB format specification
    # https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    pdb_line = (f"{record}"                # 1-6: RECORD NAME
               f"{atom_serial:5d}"         # 7-11: ATOM SERIAL NUMBER
               f" "                        # 12: SPACE
               f"{formatted_atom_name}"    # 13-16: ATOM NAME
               f"{formatted_alt_loc}"      # 17: ALTERNATE LOCATION INDICATOR
               f"{formatted_res_name}"     # 18-20: RESIDUE NAME
               f" "                        # 21: SPACE
               f"{formatted_chain_id}"     # 22: CHAIN IDENTIFIER
               f"{res_seq:4d}"             # 23-26: RESIDUE SEQUENCE NUMBER
               f"{formatted_i_code}"       # 27: INSERTION CODE
               f"   "                       # 28-30: 3 SPACES
               f"{format_coord(x)}"        # 31-38: X COORDINATE
               f"{format_coord(y)}"        # 39-46: Y COORDINATE
               f"{format_coord(z)}"        # 47-54: Z COORDINATE
               f"{format_occ_bfac(occupancy)}"  # 55-60: OCCUPANCY
               f"{format_occ_bfac(b_factor)}"   # 61-66: TEMPERATURE FACTOR
               f"      "                    # 67-72: 6 SPACES
               f"{formatted_element}"       # 73-74: ELEMENT SYMBOL (note: PDB 3.3 uses columns 73-74 for element)
               f"{formatted_charge}"        # 75-76: CHARGE (note: PDB 3.3 uses columns 75-76 for charge)
               f"  "                       # 77-80: 2 SPACES (PDB 3.3 has extra space at end)
              )
    
    # Ensure exactly 80 characters
    return pdb_line[:80]
