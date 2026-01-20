#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Parser Module

Handles PDB file reading and parsing.
"""

from typing import List, Optional
from src.models.structure import Structure
from src.models.atom import PDBAtom
from src.utils.logger import Logger
from src.utils.structure_utils import residues_to_atom_indices


class PDBParser:
    """
    PDB file parser for reading and parsing PDB format files.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
    """
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
    
    def parse_file(self, file_path: str) -> Structure:
        """
        Parse a PDB file and return a Structure object.
        
        Args:
            file_path (str): Path to PDB file
            
        Returns:
            Structure: Parsed structure object
        """
        structure = Structure()
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            self.logger.debug(f"Successfully read {len(lines)} lines from {file_path}")
        except FileNotFoundError:
            structure.add_error(f"Error: File '{file_path}' not found")
            self.logger.error(f"File not found: {file_path}")
            return structure
        except PermissionError:
            structure.add_error(f"Error: Permission denied when reading file '{file_path}'")
            self.logger.error(f"Permission denied for file: {file_path}")
            return structure
        except Exception as e:
            structure.add_error(f"Error: An error occurred while reading the file - {str(e)}")
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return structure
        
        line_num = 0
        parse_error_count = 0
        self.logger.debug("Starting PDB parsing process")
        
        for line in lines:
            line_num += 1
            line = line.rstrip('\n')
            
            # Skip empty lines
            if not line:
                continue
            
            # Process ATOM records
            if line.startswith('ATOM'):
                self.logger.debug(f"Processing ATOM record at line {line_num}")
                if not self._parse_atom_line(line, line_num, structure):
                    parse_error_count += 1
                    self.logger.warning(f"Failed to parse ATOM record at line {line_num}")
            # Process HETATM records
            elif line.startswith('HETATM'):
                self.logger.debug(f"Processing HETATM record at line {line_num}")
                if not self._parse_atom_line(line, line_num, structure):
                    parse_error_count += 1
                    self.logger.warning(f"Failed to parse HETATM record at line {line_num}")
            # Save other records
            else:
                structure.add_other_record(line)
                self.logger.debug(f"Saving non-atom record at line {line_num}: {line[:20]}...")
        
        # Build hierarchy after parsing
        structure.build_hierarchy()
        
        # Align structure to standard coordinate system after parsing
        structure.align_to_standard_coordinate_system()
        
        self.logger.info(f"Parsing complete. Errors: {len(structure.errors)}, Warnings: {len(structure.warnings)}")
        return structure
    
    def _parse_atom_line(self, line: str, line_num: int, structure: Structure) -> bool:
        """
        Parse a single ATOM/HETATM line.
        
        Args:
            line (str): Atom record line
            line_num (int): Line number in file
            structure (Structure): Structure to add atom to
            
        Returns:
            bool: True if parsing was successful, False otherwise
        """
        try:
            # Parse according to exact PDB format field widths
            record_type = line[0:6].strip()  # 1-6: Record type
            atom_serial = int(line[6:11].strip())  # 7-11: Atom serial number
            atom_name = line[12:16].strip()  # 13-16: Atom name
            alt_loc = line[16:17].strip() or ' '  # 17: Alternate location indicator
            res_name = line[17:20].strip()  # 18-20: Residue name
            chain_id = line[21:22].strip() or ' '  # 22: Chain identifier
            res_seq = int(line[22:26].strip())  # 23-26: Residue sequence number
            i_code = line[26:27].strip() or ' '  # 27: Insertion code
            
            # Parse coordinates
            x = float(line[30:38].strip())  # 31-38: x coordinate
            y = float(line[38:46].strip())  # 39-46: y coordinate
            z = float(line[46:54].strip())  # 47-54: z coordinate
            
            # Basic coordinate validation (simple range check)
            coord_range = (-1000.0, 1000.0)
            if not (coord_range[0] <= x <= coord_range[1] and 
                    coord_range[0] <= y <= coord_range[1] and 
                    coord_range[0] <= z <= coord_range[1]):
                warning_msg = f"Line {line_num}: Atom {atom_serial} has coordinates outside reasonable range: ({x}, {y}, {z})"
                structure.add_warning(warning_msg)
                self.logger.warning(warning_msg)
            
            occupancy = float(line[54:60].strip())  # 55-60: Occupancy
            b_factor = float(line[60:66].strip())  # 61-66: Temperature factor
            element = line[76:78].strip()  # 77-78: Element symbol
            charge = line[78:80].strip()  # 79-80: Charge
            
            # Check if this is a hydrogen atom (element is H or atom name starts with H)
            is_hydrogen = element == 'H' or (atom_name and atom_name[0] == 'H')
            
            # Skip hydrogen atoms
            if is_hydrogen:
                self.logger.debug(f"Skipping hydrogen atom: {atom_serial} {atom_name} {res_name} {chain_id}{res_seq}")
                return True
            
            # Create atom object
            atom = PDBAtom(record_type, atom_serial, atom_name, alt_loc, res_name, 
                          chain_id, res_seq, i_code, occupancy, b_factor, element, charge)
            
            # Add to structure
            structure.add_atom(atom, (x, y, z))
            
            self.logger.debug(f"Successfully parsed {record_type} record: {atom_serial} {atom_name} {res_name} {chain_id}{res_seq}")
            return True  
        except ValueError as e:
            error_msg = f"Line {line_num}: Data type conversion error - {str(e)}"
            structure.add_error(error_msg)
            return False
        except IndexError as e:
            error_msg = f"Line {line_num}: Line too short to parse all fields - {str(e)}"
            structure.add_error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Line {line_num}: Parsing error - {str(e)}"
            structure.add_error(error_msg)
            return False