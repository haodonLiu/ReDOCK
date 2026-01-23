#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IO Utilities Module

Handles PDB file reading and writing operations.
"""

from typing import List, Optional, Tuple
import torch
from ..models.topology import Topology
from ..models.coordinate import Coordinate
from ..models.structure.atom import Atom
from .logger import Logger


class PDBIO:
    """
    PDB file IO handler for reading and writing PDB format files.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
    """
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
    
    def parse_file(self, file_path: str) -> Tuple[Topology, Coordinate]:
        """
        Parse a PDB file and return Topology and Coordinate objects.
        
        Args:
            file_path (str): Path to PDB file
            
        Returns:
            Tuple[Topology, Coordinate]: Parsed topology and coordinate objects
        """
        topology = Topology()
        coordinate = Coordinate()
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            self.logger.debug(f"Successfully read {len(lines)} lines from {file_path}")
        except FileNotFoundError:
            topology.add_error(f"Error: File '{file_path}' not found")
            self.logger.error(f"File not found: {file_path}")
            return topology, coordinate
        except PermissionError:
            topology.add_error(f"Error: Permission denied when reading file '{file_path}'")
            self.logger.error(f"Permission denied for file: {file_path}")
            return topology, coordinate
        except Exception as e:
            topology.add_error(f"Error: An error occurred while reading the file - {str(e)}")
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return topology, coordinate
        
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
                if not self._parse_atom_line(line, line_num, topology, coordinate):
                    parse_error_count += 1
                    self.logger.warning(f"Failed to parse ATOM record at line {line_num}")
            # Process HETATM records
            elif line.startswith('HETATM'):
                self.logger.debug(f"Processing HETATM record at line {line_num}")
                if not self._parse_atom_line(line, line_num, topology, coordinate):
                    parse_error_count += 1
                    self.logger.warning(f"Failed to parse HETATM record at line {line_num}")
            # Save other records
            else:
                topology.add_other_record(line)
                self.logger.debug(f"Saving non-atom record at line {line_num}: {line[:20]}...")
        
        # Build hierarchy after parsing
        topology.build_hierarchy()
        
        # Align coordinate to standard coordinate system after parsing
        coordinate.align_to_standard_coordinate_system()

        return topology, coordinate
    
    def _parse_atom_line(self, line: str, line_num: int, topology: Topology, coordinate: Coordinate) -> bool:
        """
        Parse a single ATOM/HETATM line.
        
        Args:
            line (str): Atom record line
            line_num (int): Line number in file
            topology (Topology): Topology to add atom to
            coordinate (Coordinate): Coordinate to add coordinates to
            
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
                topology.add_warning(warning_msg)
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
            atom = Atom(record_type, atom_serial, atom_name, alt_loc, res_name, 
                          chain_id, res_seq, i_code, occupancy, b_factor, element, charge)
            
            # Add to topology
            topology.add_atom(atom)
            
            # Add coordinates to coordinate object
            coord_tensor = torch.tensor([x, y, z], dtype=torch.float16).unsqueeze(0)
            if coordinate.coordinates.shape[0] == 0:
                coordinate.coordinates = coord_tensor
            else:
                coordinate.coordinates = torch.cat([coordinate.coordinates, coord_tensor], dim=0)
            
            self.logger.debug(f"Successfully parsed {record_type} record: {atom_serial} {atom_name} {res_name} {chain_id}{res_seq}")
            return True  
        except ValueError as e:
            error_msg = f"Line {line_num}: Data type conversion error - {str(e)}"
            topology.add_error(error_msg)
            return False
        except IndexError as e:
            error_msg = f"Line {line_num}: Line too short to parse all fields - {str(e)}"
            topology.add_error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Line {line_num}: Parsing error - {str(e)}"
            topology.add_error(error_msg)
            return False
    
    def write_file(self, topology: Topology, coordinate: Coordinate, output_path: str) -> bool:
        """
        Write a Topology and Coordinate object to a PDB file.
        
        Args:
            topology (Topology): Topology object to write
            coordinate (Coordinate): Coordinate object to write
            output_path (str): Path to output PDB file
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        self.logger.debug(f"Starting to write PDB file: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                pdb_string = self.write_string(topology, coordinate)
                f.write(pdb_string)
            
            self.logger.debug(f"Successfully wrote PDB file: {output_path}")
            return True
        except PermissionError:
            self.logger.error(f"Permission denied for writing to {output_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error writing PDB file {output_path}: {str(e)}")
            return False
    
    def write_string(self, topology: Topology, coordinate: Coordinate) -> str:
        """
        Write a Topology and Coordinate object to a PDB string.
        
        Args:
            topology (Topology): Topology object to write
            coordinate (Coordinate): Coordinate object to write
            
        Returns:
            str: PDB formatted string
        """
        self.logger.debug(f"Generating PDB string for topology and coordinate")
        
        # Use the topology's built-in to_pdb_lines method
        pdb_lines = topology.to_pdb_lines(coordinate)
        
        # Join all lines with newlines
        return "\n".join(pdb_lines) + "\n"
    
    def write_combined_structure(self, receptor_topology: Topology, receptor_coordinate: Coordinate, ligand_topology: Topology, ligand_coordinate: Coordinate, output_path: str) -> bool:
        """
        Write a combined structure of receptor and ligand to a PDB file.
        
        Args:
            receptor_topology (Topology): Receptor topology
            receptor_coordinate (Coordinate): Receptor coordinate
            ligand_topology (Topology): Ligand topology
            ligand_coordinate (Coordinate): Ligand coordinate
            output_path (str): Path to output PDB file
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        self.logger.debug(f"Writing combined receptor-ligand structure to {output_path}")
        
        try:
            # Create combined topology and coordinate
            combined_topology = Topology()
            combined_coordinate = Coordinate()
            
            # Add receptor atoms and coordinates
            for atom in receptor_topology.atoms:
                combined_topology.add_atom(atom)
            
            if receptor_coordinate.coordinates.shape[0] > 0:
                combined_coordinate.coordinates = receptor_coordinate.coordinates.clone()
            
            # Add ligand atoms and coordinates
            for atom in ligand_topology.atoms:
                combined_topology.add_atom(atom)
            
            if ligand_coordinate.coordinates.shape[0] > 0:
                if combined_coordinate.coordinates.shape[0] > 0:
                    combined_coordinate.coordinates = torch.cat([combined_coordinate.coordinates, ligand_coordinate.coordinates], dim=0)
                else:
                    combined_coordinate.coordinates = ligand_coordinate.coordinates.clone()
            
            # Build hierarchy
            combined_topology.build_hierarchy()
            
            # Write the combined structure
            return self.write_file(combined_topology, combined_coordinate, output_path)
        except Exception as e:
            self.logger.error(f"Error writing combined structure: {str(e)}")
            return False

