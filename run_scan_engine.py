#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run script for Scan Engine based on input_config.json
"""

import json
import os
import sys
import torch

# Add src directory to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_path)

# Use absolute imports with package name
from src.utils import Logger, PDBIO
from src.models.force_field import ForceField
from src.core.energy_calculator import EnergyCalculator
from src.models.topology import Topology
from src.models.coordinate import Coordinate
from src.utils import residues_to_atom_indices
from src.core.scan_engine import ScanEngine

def load_complex_structure(parser, complex_files, receptor_chains, ligand_chains, logger):
    """
    Load complex structure from one or two PDB files.
    
    Args:
        parser (PDBParser): PDB parser instance
        complex_files (list): List of PDB files (1 or 2 files)
        receptor_chains (list): List of receptor chain IDs
        ligand_chains (list): List of ligand chain IDs
        logger (Logger): Logger instance
        
    Returns:
        tuple: (receptor_topology, receptor_coordinate, ligand_topology, ligand_coordinate)
    """
    if len(complex_files) == 1:
        # Single complex file
        logger.info(f"Loading complex structure from file: {complex_files[0]}")
        complex_topology, complex_coordinate = parser.parse_file(complex_files[0])
        
        # Split into receptor and ligand based on chains
        receptor_topology = Topology()
        receptor_coordinate = Coordinate()
        ligand_topology = Topology()
        ligand_coordinate = Coordinate()
        
        # Split atoms by chains
        for i, atom in enumerate(complex_topology.atoms):
            chain_id = atom.chain_id
            x, y, z = complex_coordinate.get_coordinates_by_index(i)
            if chain_id in receptor_chains:
                receptor_topology.add_atom(atom)
                coord_tensor = torch.tensor([x, y, z], dtype=torch.float16).unsqueeze(0)
                if receptor_coordinate.coordinates.shape[0] == 0:
                    receptor_coordinate.coordinates = coord_tensor
                else:
                    receptor_coordinate.coordinates = torch.cat([receptor_coordinate.coordinates, coord_tensor], dim=0)
            elif chain_id in ligand_chains:
                ligand_topology.add_atom(atom)
                coord_tensor = torch.tensor([x, y, z], dtype=torch.float16).unsqueeze(0)
                if ligand_coordinate.coordinates.shape[0] == 0:
                    ligand_coordinate.coordinates = coord_tensor
                else:
                    ligand_coordinate.coordinates = torch.cat([ligand_coordinate.coordinates, coord_tensor], dim=0)
        
        # Copy other attributes
        receptor_topology.other_records = complex_topology.other_records.copy()
        ligand_topology.other_records = complex_topology.other_records.copy()
        
        # Build hierarchy
        receptor_topology.build_hierarchy()
        ligand_topology.build_hierarchy()
        
    elif len(complex_files) == 2:
        # Two files to be combined
        logger.info(f"Loading first structure file: {complex_files[0]}")
        topology1, coordinate1 = parser.parse_file(complex_files[0])
        logger.info(f"Loading second structure file: {complex_files[1]}")
        topology2, coordinate2 = parser.parse_file(complex_files[1])
        
        # Combine topologies
        complex_topology = Topology()
        for atom in topology1.atoms:
            complex_topology.add_atom(atom)
        for atom in topology2.atoms:
            complex_topology.add_atom(atom)
        
        # Combine coordinates
        complex_coordinate = Coordinate()
        if coordinate1.coordinates.shape[0] > 0:
            complex_coordinate.coordinates = coordinate1.coordinates.clone()
        if coordinate2.coordinates.shape[0] > 0:
            if complex_coordinate.coordinates.shape[0] == 0:
                complex_coordinate.coordinates = coordinate2.coordinates.clone()
            else:
                complex_coordinate.coordinates = torch.cat([complex_coordinate.coordinates, coordinate2.coordinates], dim=0)
        
        # Copy other attributes
        complex_topology.other_records = topology1.other_records.copy()
        complex_topology.other_records.extend(topology2.other_records)
        
        # Build hierarchy
        complex_topology.build_hierarchy()
        
        # Split into receptor and ligand based on chains
        receptor_topology = Topology()
        receptor_coordinate = Coordinate()
        ligand_topology = Topology()
        ligand_coordinate = Coordinate()
        
        # Split atoms by chains
        for i, atom in enumerate(complex_topology.atoms):
            chain_id = atom.chain_id
            x, y, z = complex_coordinate.get_coordinates_by_index(i)
            if chain_id in receptor_chains:
                receptor_topology.add_atom(atom)
                coord_tensor = torch.tensor([x, y, z], dtype=torch.float16).unsqueeze(0)
                if receptor_coordinate.coordinates.shape[0] == 0:
                    receptor_coordinate.coordinates = coord_tensor
                else:
                    receptor_coordinate.coordinates = torch.cat([receptor_coordinate.coordinates, coord_tensor], dim=0)
            elif chain_id in ligand_chains:
                ligand_topology.add_atom(atom)
                coord_tensor = torch.tensor([x, y, z], dtype=torch.float16).unsqueeze(0)
                if ligand_coordinate.coordinates.shape[0] == 0:
                    ligand_coordinate.coordinates = coord_tensor
                else:
                    ligand_coordinate.coordinates = torch.cat([ligand_coordinate.coordinates, coord_tensor], dim=0)
        
        # Copy other attributes
        receptor_topology.other_records = complex_topology.other_records.copy()
        ligand_topology.other_records = complex_topology.other_records.copy()
        
        # Build hierarchy
        receptor_topology.build_hierarchy()
        ligand_topology.build_hierarchy()
    else:
        raise ValueError("complex_files should contain 1 or 2 files")
    
    return receptor_topology, receptor_coordinate, ligand_topology, ligand_coordinate

def main():
    """
    Main function to run the scan engine based on input_config.json
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run protein-protein docking scan engine')
    parser.add_argument('--config', default='input_config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration file
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Initialize logger
    logger = Logger(debug=config.get('debug', False))
    
    try:
        # Get input type
        input_type = config.get('input_type', 'separate')
        
        # Initialize parser
        parser = PDBIO()
        
        if input_type == 'complex':
            # Load complex structure
            complex_files = config.get('complex_files', [])
            receptor_chains = config.get('receptor_chains', [])
            ligand_chains = config.get('ligand_chains', [])
            
            if not complex_files:
                raise ValueError("complex_files must be provided for input_type='complex'")
            if not receptor_chains:
                raise ValueError("receptor_chains must be provided for input_type='complex'")
            if not ligand_chains:
                raise ValueError("ligand_chains must be provided for input_type='complex'")
            
            receptor_topology, receptor_coordinate, ligand_topology, ligand_coordinate = load_complex_structure(parser, complex_files, receptor_chains, ligand_chains, logger)
            
        else:
            # Separate receptor and ligand files
            receptor_file = config['receptor_file']
            ligand_file = config['ligand_file']
            
            logger.info(f"Loading receptor PDB file: {receptor_file}")
            receptor_topology, receptor_coordinate = parser.parse_file(receptor_file)
            
            logger.info(f"Loading ligand PDB file: {ligand_file}")
            ligand_topology, ligand_coordinate = parser.parse_file(ligand_file)
        
        # Convert residue groups to atom indices
        logger.info("Converting residue groups to atom indices...")
        receptor_residues = config.get('receptor_residues', [])
        ligand_residues = config.get('ligand_residues', [])
        
        receptor_group = residues_to_atom_indices(receptor_topology, receptor_residues)
        ligand_group = residues_to_atom_indices(ligand_topology, ligand_residues)
        
        logger.info(f"Receptor residue group: {len(receptor_group)} atoms")
        logger.info(f"Ligand residue group: {len(ligand_group)} atoms")
        
        # Initialize force field and energy calculator
        force_field = ForceField()
        use_gpu = config.get('use_gpu', True)
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Calculate atom types for topology
        logger.info("Calculating atom types using force field parameters...")
        
        # Calculate atom types for receptor
        receptor_atom_types = []
        for atom in receptor_topology.atoms:
            atom_type = force_field.get_atom_type_for_residue(atom.res_name, atom.atom_name)
            receptor_atom_types.append(atom_type)
        receptor_topology.atom_types = receptor_atom_types
        
        # Calculate atom types for ligand
        ligand_atom_types = []
        for atom in ligand_topology.atoms:
            atom_type = force_field.get_atom_type_for_residue(atom.res_name, atom.atom_name)
            ligand_atom_types.append(atom_type)
        ligand_topology.atom_types = ligand_atom_types
        
        energy_calculator = EnergyCalculator(force_field, device)
        
        logger.info(f"Using device: {device}")
        
        # Initialize scan engine
        scan_engine = ScanEngine(logger, energy_calculator, device)
        
        # Run the scan
        logger.info("Starting dedicated scan engine...")
        min_dist = config.get('min_dist', 5.0)
        max_dist = config.get('max_dist', 8.0)
        conformations = scan_engine.dedicated_scan(
            receptor_top=receptor_topology,
            receptor_coord=receptor_coordinate,
            ligand_top=ligand_topology,
            ligand_coord=ligand_coordinate,
            receptor_group=receptor_group,
            ligand_group=ligand_group,
            initial_distance=max_dist,
            coarse_translation_step=config.get('step_size', 1.0),
            fine_translation_step=config.get('step_size', 1.0) * 0.5,
            rotation_step=360.0 / config.get('num_rotations', 36),
            max_translation_range=max_dist,
            min_residue_distance=min_dist,
            max_residue_distance=max_dist
        )
        
        logger.info(f"Generated {len(conformations)} valid conformations")
        
        if not conformations:
            logger.warning("No valid conformations found")
            return 1
        
        # Save the best conformation
        best_conformation = conformations[0]
        output_file = config.get('output_file', 'scan_results.pdb')
        if output_file is None:
            output_file = 'scan_results.pdb'
        logger.info(f"Saving best conformation to {output_file}")
        
        # Use enhanced IO module to save the combined structure
        writer = PDBIO(logger=logger)
        best_top, best_coord = best_conformation
        writer.write_combined_structure(receptor_topology, receptor_coordinate, best_top, best_coord, output_file)
        
        logger.info(f"Scan completed successfully! Output saved to {output_file}")
        logger.info(f"Generated {len(conformations)} valid conformations")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
