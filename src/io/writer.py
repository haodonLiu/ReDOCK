#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Writer Module

Handles PDB file writing.
"""

from typing import Optional
from src.models.structure import Structure
from src.utils.logger import Logger
from src.utils.common import format_pdb_line


class PDBWriter:
    """
    PDB file writer for generating PDB format files.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
    """
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
    
    def write_file(self, structure: Structure, output_path: str) -> bool:
        """
        Write a Structure object to a PDB file.
        
        Args:
            structure (Structure): Structure object to write
            output_path (str): Path to output PDB file
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        self.logger.debug(f"Starting to write PDB file: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                pdb_string = self.write_string(structure)
                f.write(pdb_string)
            
            self.logger.debug(f"Successfully wrote PDB file: {output_path}")
            return True
        except PermissionError:
            self.logger.error(f"Permission denied for writing to {output_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error writing PDB file {output_path}: {str(e)}")
            return False
    
    def write_string(self, structure: Structure) -> str:
        """
        Write a Structure object to a PDB string.
        
        Args:
            structure (Structure): Structure object to write
            
        Returns:
            str: PDB formatted string
        """
        self.logger.debug(f"Generating PDB string for structure")
        
        # Separate other records into different categories
        header_records = []  # Records that should be written at the beginning
        conect_records = []  # CONECT records that should be written after ATOM records
        ter_end_records = []  # TER and END records
        
        for record in structure.other_records:
            if record.startswith('TER') or record.startswith('END'):
                ter_end_records.append(record)
            elif record.startswith('CONECT'):
                conect_records.append(record)
            else:
                header_records.append(record)
        
        # Build the PDB string
        pdb_lines = []
        
        # Write header records
        self.logger.debug(f"Writing {len(header_records)} header records")
        for record in header_records:
            pdb_lines.append(record)
        
        # Write atom records
        self.logger.debug(f"Writing {len(structure.atoms)} atom records")
        for atom, coord in zip(structure.atoms, structure.coordinates):
            x, y, z = coord
            # Format ATOM/HETATM record according to PDB spec
            line = format_pdb_line(
                atom.record_type,
                atom.atom_serial,
                atom.atom_name,
                atom.alt_loc,
                atom.res_name,
                atom.chain_id,
                atom.res_seq,
                atom.i_code,
                x, y, z,
                atom.occupancy,
                atom.b_factor,
                atom.element,
                atom.charge
            )
            pdb_lines.append(line)
        
        # Write CONECT records after atom records (before TER/END)
        self.logger.debug(f"Writing {len(conect_records)} CONECT records")
        for record in conect_records:
            pdb_lines.append(record)
        
        # Write TER/END records
        self.logger.debug(f"Writing {len(ter_end_records)} TER/END records")
        for record in ter_end_records:
            pdb_lines.append(record)
        
        # Ensure there's an END record
        if not any(record.startswith('END') for record in structure.other_records):
            pdb_lines.append("END")
            self.logger.debug("Added END record")
        
        # Join all lines with newlines
        return "\n".join(pdb_lines) + "\n"
    
    def write_atom_summary(self, structure: Structure, start: int = 0, end: Optional[int] = None) -> None:
        """
        Print an atom summary to stdout using the logger.
        
        Args:
            structure (Structure): Structure to summarize
            start (int): Start index for summary
            end (Optional[int]): End index for summary, None for all atoms
        """
        if end is None or end > len(structure.atoms):
            end = len(structure.atoms)
        
        if start >= end:
            self.logger.error("Start index is greater than or equal to end index")
            return
        
        # Format atom summary as a table
        headers = ["Index", "Record", "Serial", "Atom", "Residue", "Chain", "Res Seq", "x", "y", "z"]
        rows = []
        
        for i in range(start, end):
            atom = structure.atoms[i]
            x, y, z = structure.coordinates[i]
            rows.append([
                f"{i:5d}",
                f"{atom.record_type:>10}",
                f"{atom.atom_serial:>8d}",
                f"{atom.atom_name:>8}",
                f"{atom.res_name:>8}",
                f"{atom.chain_id:>5}",
                f"{atom.res_seq:>8d}",
                f"{x:8.3f}",
                f"{y:8.3f}",
                f"{z:8.3f}"
            ])
        
        # Log the summary using logger table method
        self.logger.section(f"Atom Information Summary ({start} to {end-1})")
        self.logger.table(headers, rows)
    
    def write_parsing_report(self, structure: Structure, file_path: str) -> None:
        """
        Write parsing report to stdout using the logger.
        
        Args:
            structure (Structure): Parsed structure
            file_path (str): Original file path
        """
        # Log the parsing report header and summary
        self.logger.section("PDB File Parsing Report")
        self.logger.info(f"File: {file_path}")
        self.logger.info(f"Total atoms: {len(structure.atoms)}")
        self.logger.info(f"Non-atom records: {len(structure.other_records)}")
        self.logger.info(f"Parsing errors: {len(structure.errors)}")
        self.logger.info(f"Parsing warnings: {len(structure.warnings)}")
        
        # Output errors and warnings
        if structure.errors:
            self.logger.section(f"Error Messages ({len(structure.errors)})")
            for error in structure.errors:
                self.logger.error(f"  {error}", indent=2)
        
        if structure.warnings:
            self.logger.section(f"Warning Messages ({len(structure.warnings)})")
            for warning in structure.warnings:
                self.logger.warning(f"  {warning}", indent=2)
