#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB CLI Module

Provides command-line interface functionality for the PDB processor.
"""

import sys
import argparse
import json
import os
from src.io.parser import PDBParser
from src.io.writer import PDBWriter
from src.utils.logger import Logger
from src.core.statistics import StructureStatistics
from src.core.docking import Docking


class PDBCLI:
    """
    Command-line interface for the PDB processor.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
        parser (PDBParser): PDB file parser
        writer (PDBWriter): PDB file writer
    """
    def __init__(self):
        self.logger = None
        self.parser = None
        self.writer = None
    
    def parse_arguments(self, args: list) -> dict:
        """
        Parse command-line arguments using argparse.
        
        Args:
            args (list): Command-line arguments
            
        Returns:
            dict: Parsed arguments as a dictionary
        """
        # Default values
        default_values = {
            'debug': False,
            'mode': 'parse',
            'input_file': None,
            'output_file': None,
            'dx': 0.0,
            'dy': 0.0,
            'dz': 0.0,
            'receptor_file': None,
            'ligand_file': None,
            'receptor_chains': [],
            'ligand_chains': [],
            'receptor_residues': [],
            'ligand_residues': [],
            'force_field': None,
            'max_dist': 5.0,
            'num_rotations': 36,
            'use_gpu': True,
            'step_size': 1.0,
            'solvent_penalty_coeff': 0.1,
            'distance_penalty_coeff': 0.5,
            'save_all': False,
            'num_output_confs': 10
        }
        
        # Check for JSON argument first
        json_arg = None
        if '--json' in args or '-j' in args:
            json_idx = args.index('--json') if '--json' in args else args.index('-j')
            if json_idx + 1 < len(args):
                json_arg = args[json_idx + 1]
                # Remove JSON arguments from args list
                args = args[:json_idx] + args[json_idx + 2:]
        
        # Parse JSON if provided
        if json_arg:
            if os.path.isfile(json_arg):
                with open(json_arg, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            else:
                try:
                    json_data = json.loads(json_arg)
                except json.JSONDecodeError:
                    self.logger = Logger(False, None)
                    self.logger.error(f"Error: Invalid JSON string or file path: {json_arg}")
                    return default_values
            
            # Update default values with JSON data
            for key, value in json_data.items():
                if key in default_values:
                    default_values[key] = value
            
            if 'mode' in json_data:
                default_values['mode'] = json_data['mode']
        
        # Create argparse parser
        parser = argparse.ArgumentParser(
            description='PDB Processor - Protein Structure Analysis and Docking Tool',
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        # General options
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration (if available)')
        
        # Mode-specific options
        subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
        
        # Parse mode (default)
        parse_parser = subparsers.add_parser('parse', help='Basic PDB file processing mode')
        parse_parser.add_argument('input_file', nargs='?', help='Input PDB file path')
        parse_parser.add_argument('output_file', nargs='?', help='Output PDB file path')
        
        # Translate mode
        translate_parser = subparsers.add_parser('translate', help='Protein coordinate translation mode')
        translate_parser.add_argument('input_file', help='Input PDB file path')
        translate_parser.add_argument('output_file', nargs='?', help='Output PDB file path')
        translate_parser.add_argument('--dx', type=float, default=0.0, help='X-axis translation distance (Å)')
        translate_parser.add_argument('--dy', type=float, default=0.0, help='Y-axis translation distance (Å)')
        translate_parser.add_argument('--dz', type=float, default=0.0, help='Z-axis translation distance (Å)')
        
        # Prealign mode
        prealign_parser = subparsers.add_parser('prealign', help='Protein pre-alignment mode')
        prealign_parser.add_argument('--receptor', required=True, help='Receptor protein PDB file path')
        prealign_parser.add_argument('--ligand', required=True, help='Ligand protein PDB file path')
        prealign_parser.add_argument('--receptor-chains', type=str, help='Receptor chain IDs (comma-separated)')
        prealign_parser.add_argument('--ligand-chains', type=str, help='Ligand chain IDs (comma-separated)')
        prealign_parser.add_argument('--receptor-residues', required=True, type=str, help='Receptor residue group (comma-separated)')
        prealign_parser.add_argument('--ligand-residues', required=True, type=str, help='Ligand residue group (comma-separated)')
        prealign_parser.add_argument('--max-dist', type=float, default=5.0, help='Maximum search distance (Å)')
        prealign_parser.add_argument('--force-field', help='Force field XML file path')
        prealign_parser.add_argument('-o', '--output-file', help='Output file path')
        
        # Dock mode
        dock_parser = subparsers.add_parser('dock', help='Protein docking conformation search mode')
        dock_parser.add_argument('--receptor', required=True, help='Receptor protein PDB file path')
        dock_parser.add_argument('--ligand', required=True, help='Ligand protein PDB file path')
        dock_parser.add_argument('--receptor-chains', type=str, help='Receptor chain IDs (comma-separated)')
        dock_parser.add_argument('--ligand-chains', type=str, help='Ligand chain IDs (comma-separated)')
        dock_parser.add_argument('--receptor-residues', required=True, type=str, help='Receptor residue group (comma-separated)')
        dock_parser.add_argument('--ligand-residues', required=True, type=str, help='Ligand residue group (comma-separated)')
        dock_parser.add_argument('--max-dist', type=float, default=5.0, help='Maximum search distance (Å)')
        dock_parser.add_argument('--num-rotations', type=int, default=36, help='Number of rotations')
        dock_parser.add_argument('--step-size', type=float, default=1.0, help='Step size for intermediate conformation generation (Å)')
        dock_parser.add_argument('--solvent-penalty', type=float, default=0.1, help='Solvent penalty coefficient')
        dock_parser.add_argument('--distance-penalty', type=float, default=0.5, help='Distance penalty coefficient')
        dock_parser.add_argument('--num-output-confs', type=int, default=10, help='Number of output conformations')
        dock_parser.add_argument('--force-field', help='Force field XML file path')
        dock_parser.add_argument('-o', '--output-file', help='Output file path')
        dock_parser.add_argument('--save-all', action='store_true', help='Save all generated conformations')
        dock_parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration (if available)')
        
        # Add --gpu to prealign parser as well
        prealign_parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration (if available)')
        
        # Parse arguments
        parsed_args = parser.parse_args(args)
        
        # Convert parsed args to dictionary
        args_dict = vars(parsed_args)
        
        # Update default values with parsed args
        for key, value in args_dict.items():
            if value is not None:
                default_values[key] = value
        
        # Map argparse parameters to expected keys
        if 'receptor' in args_dict and args_dict['receptor'] is not None:
            default_values['receptor_file'] = args_dict['receptor']
        if 'ligand' in args_dict and args_dict['ligand'] is not None:
            default_values['ligand_file'] = args_dict['ligand']
        
        # Handle comma-separated values only if they are strings
        if default_values.get('receptor_chains') and isinstance(default_values['receptor_chains'], str):
            default_values['receptor_chains'] = default_values['receptor_chains'].split(',')
        if default_values.get('ligand_chains') and isinstance(default_values['ligand_chains'], str):
            default_values['ligand_chains'] = default_values['ligand_chains'].split(',')
        if default_values.get('receptor_residues') and isinstance(default_values['receptor_residues'], str):
            default_values['receptor_residues'] = default_values['receptor_residues'].split(',')
        if default_values.get('ligand_residues') and isinstance(default_values['ligand_residues'], str):
            default_values['ligand_residues'] = default_values['ligand_residues'].split(',')
        
        # Special handling for mode (if not set via JSON or command line)
        if default_values['mode'] is None:
            default_values['mode'] = 'parse'
        
        return default_values
    
    def print_help(self) -> None:
        """
        Print help message.
        """
        print("=== PDB Processor Usage Instructions ===")
        print("\nBasic Usage:")
        print("  python main.py [mode] [options]")
        print("\nModes:")
        print("  parse       Basic PDB file processing")
        print("  translate   Protein coordinate translation")
        print("  prealign    Protein pre-alignment for docking")
        print("  dock        Protein docking conformation search")
        print("\nCommon Options:")
        print("  --help, -h  Show this help message")
        print("  --debug, -d Enable debug mode")
        print("  --gpu       Enable GPU acceleration")
        print("  --json, -j  JSON parameter string or file path")
        print("\nExamples:")
        print("  # Basic parsing")
        print("  python main.py parse structure/PP5_CD.pdb output.pdb")
        print("  ")
        print("  # Translation")
        print("  python main.py translate structure/PP5_CD.pdb --dx 5.0 --dz 10.0 translated.pdb")
        print("  ")
        print("  # Docking")
        print("  python main.py dock --receptor structure/PP5_CD.pdb --ligand structure/triP-KD_AKT1.pdb ")
        print("                   --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 ")
        print("                   --max-dist 5.0 --num-rotations 10 --gpu")
        print("  ")
        print("  # JSON mode")
        print("  python main.py --json '{\"mode\":\"dock\",\"receptor_file\":\"structure/PP5_CD.pdb\",\"ligand_file\":\"structure/triP-KD_AKT1.pdb\",\"receptor_residues\":[\"B:184\",\"B:185\"],\"ligand_residues\":[\"A:144\",\"A:145\"]}'")
    
    def run(self, args: list) -> int:
        """
        Run the CLI application.
        
        Args:
            args (list): Command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        # Initialize logger early for error handling during parsing
        self.logger = Logger(False, None)  # Default to non-debug mode for parsing
        
        # Parse arguments
        parsed_args = self.parse_arguments(args)
        
        # Re-initialize logger with correct debug setting
        log_file = None
        self.logger = Logger(parsed_args['debug'], log_file)
        self.parser = PDBParser(self.logger)
        self.writer = PDBWriter(self.logger)
        
        # Check mode and execute appropriate functionality
        if parsed_args['mode'] == 'dock':
            # Docking mode
            return self.run_docking(parsed_args)
        elif parsed_args['mode'] == 'prealign':
            # Pre-align mode
            return self.run_prealign(parsed_args)
        elif parsed_args['mode'] == 'translate':
            # Translate mode
            return self.run_translate(parsed_args)
        else:
            # Basic parse mode
            return self.run_parse(parsed_args)
    
    def run_parse(self, parsed_args: dict) -> int:
        """
        Run basic PDB parsing mode.
        
        Args:
            parsed_args (dict): Parsed command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        # Check for input file
        if not parsed_args['input_file']:
            self.print_help()
            return 1
        
        # Parse the PDB file
        input_file = parsed_args['input_file']
        self.logger.info(f"Parsing PDB file: {input_file}")
        structure = self.parser.parse_file(input_file)
        
        # Check if parsing was successful
        if not structure.atoms:
            self.logger.error("\nParsing failed, exiting program")
            return 1
        
        # Write parsing report
        self.writer.write_parsing_report(structure, input_file)
        
        # Display statistics
        stats = StructureStatistics(structure).get_statistics()
        self.logger.section("Statistics")
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        
        # Print atom summary
        self.writer.write_atom_summary(structure, 0, min(10, len(structure.atoms)))
        
        # If output file is specified, write to it
        output_file = parsed_args['output_file']
        if output_file:
            self.logger.info(f"\nWriting PDB file: {output_file}")
            if self.writer.write_file(structure, output_file):
                self.logger.info("Write successful")
            else:
                self.logger.error("Write failed")
                return 1
        
        return 0
    
    def run_translate(self, parsed_args: dict) -> int:
        """
        Run protein coordinate translation mode.
        
        Args:
            parsed_args (dict): Parsed command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        from src.core.coordinate_manager import CoordinateManager
        import os
        from datetime import datetime
        
        # Check for input file
        if not parsed_args['input_file']:
            self.logger.error("Error: Missing input PDB file path")
            self.print_help()
            return 1
        
        # Parse the PDB file
        input_file = parsed_args['input_file']
        self.logger.info(f"Parsing PDB file: {input_file}")
        structure = self.parser.parse_file(input_file)
        
        # Check if parsing was successful
        if not structure.atoms:
            self.logger.error("\nParsing failed, exiting program")
            return 1
        
        # Create project folder
        project_name = f"translate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        project_dir = os.path.join(os.getcwd(), project_name)
        os.makedirs(project_dir, exist_ok=True)
        self.logger.info(f"\nCreated project folder: {project_dir}")
        
        # Print translation parameters
        dx, dy, dz = parsed_args['dx'], parsed_args['dy'], parsed_args['dz']
        self.logger.section("Translation Parameters")
        self.logger.info(f"X-axis translation: {dx} Å")
        self.logger.info(f"Y-axis translation: {dy} Å")
        self.logger.info(f"Z-axis translation: {dz} Å")
        
        # Perform translation
        self.logger.section("Starting Translation Operation")
        coord_manager = CoordinateManager(structure)
        coord_manager.translate_coordinates(dx, dy, dz)
        self.logger.info("Translation completed")
        
        # Save translated structure
        translated_file = os.path.join(project_dir, "translated.pdb")
        self.logger.info(f"\nWriting translated structure to PDB file: {translated_file}")
        if self.writer.write_file(structure, translated_file):
            self.logger.info("Write successful")
        else:
            self.logger.error("Write failed")
            return 1
        
        # If custom output file is specified, also write to it
        output_file = parsed_args['output_file']
        if output_file:
            self.logger.info(f"\nWriting translated structure to specified PDB file: {output_file}")
            if self.writer.write_file(structure, output_file):
                self.logger.info("Write successful")
            else:
                self.logger.error("Write failed")
        
        return 0
    
    def run_prealign(self, parsed_args: dict) -> int:
        """
        Run protein pre-alignment mode.
        
        Args:
            parsed_args (dict): Parsed command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        import os
        from datetime import datetime
        
        # Check required prealign parameters
        if not parsed_args['receptor_file'] or not parsed_args['ligand_file']:
            self.logger.error("Error: Missing receptor or ligand file path")
            self.print_help()
            return 1
        
        if not parsed_args['receptor_residues'] or not parsed_args['ligand_residues']:
            self.logger.error("Error: Missing receptor or ligand residue groups")
            self.print_help()
            return 1
        
        # Create project folder
        project_name = f"{parsed_args['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')[:-3]}"
        project_dir = os.path.join(os.getcwd(), project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create log file and update logger
        log_file = os.path.join(project_dir, "docking.log")
        self.logger = Logger(parsed_args['debug'], log_file)
        
        # Parse receptor and ligand files
        # Check if receptor and ligand are from the same file
        same_file = parsed_args['receptor_file'] == parsed_args['ligand_file']
        
        receptor = None
        ligand = None
        
        if same_file:
            self.logger.info(f"Parsing PDB file: {parsed_args['receptor_file']}")
            structure = self.parser.parse_file(parsed_args['receptor_file'])
            
            # Extract receptor chains
            if parsed_args['receptor_chains']:
                self.logger.info(f"Extracting receptor chains: {', '.join(parsed_args['receptor_chains'])}")
                receptor = structure.extract_chains(parsed_args['receptor_chains'])
            else:
                # If no receptor chains specified, use the entire structure as receptor
                receptor = structure
            
            # Extract ligand chains
            if parsed_args['ligand_chains']:
                self.logger.info(f"Extracting ligand chains: {', '.join(parsed_args['ligand_chains'])}")
                ligand = structure.extract_chains(parsed_args['ligand_chains'])
            else:
                # This case should not happen as we're in same file mode
                self.logger.error("Error: No ligand chains specified for same-file docking")
                return 1
        else:
            # Traditional mode: separate files
            self.logger.info(f"Parsing receptor file: {parsed_args['receptor_file']}")
            receptor = self.parser.parse_file(parsed_args['receptor_file'])
            
            self.logger.info(f"Parsing ligand file: {parsed_args['ligand_file']}")
            ligand = self.parser.parse_file(parsed_args['ligand_file'])
        
        # Check if parsing was successful
        if not receptor.atoms or not ligand.atoms:
            self.logger.error("\nParsing failed, exiting program")
            return 1
        
        # Initialize docking for pre-alignment
        docking = Docking(self.logger, parsed_args['use_gpu'])
        docking.set_proteins(receptor, ligand)
        
        # Load force field if specified
        if parsed_args['force_field']:
            self.logger.info(f"Loading force field file: {parsed_args['force_field']}")
            if docking.load_force_field(parsed_args['force_field']):
                self.logger.info("Force field loaded successfully")
            else:
                self.logger.warning("Failed to load force field, will use default parameters for energy calculation")
        
        # Print initial statistics
        self.logger.section("Initial Protein Statistics")
        self.logger.info(f"Receptor protein: {parsed_args['receptor_file']}")
        self.logger.info(f"  Number of atoms: {len(receptor.atoms)}")
        self.logger.info(f"  Specified residue group: {', '.join(parsed_args['receptor_residues'])}")
        
        self.logger.info(f"Ligand protein: {parsed_args['ligand_file']}")
        self.logger.info(f"  Number of atoms: {len(ligand.atoms)}")
        self.logger.info(f"  Specified residue group: {', '.join(parsed_args['ligand_residues'])}")
        
        self.logger.section("Pre-alignment Parameters")
        self.logger.info(f"  Maximum search distance: {parsed_args['max_dist']} Å")
        
        # Perform pre-alignment
        self.logger.section("Starting Pre-alignment Operation")
        if not docking.prealign(
            parsed_args['receptor_residues'],
            parsed_args['ligand_residues'],
            parsed_args['max_dist']
        ):
            self.logger.error("\nPre-alignment failed: Unable to set residue groups")
            return 1
        
        self.logger.section("Pre-alignment Completed")
        
        self.logger.info(f"Project folder: {project_dir}")
        
        # Save parsed arguments to JSON file
        import json
        json_dest = os.path.join(project_dir, "input_config.json")
        with open(json_dest, 'w', encoding='utf-8') as f:
            json.dump(parsed_args, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved input parameters to JSON file: {json_dest}")
        
        return 0
    
    def run_docking(self, parsed_args: dict) -> int:
        """
        Run protein docking mode.
        
        Args:
            parsed_args (dict): Parsed command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        import os
        import csv
        from datetime import datetime
        
        # Check required docking parameters
        if not parsed_args['receptor_file'] or not parsed_args['ligand_file']:
            self.logger.error("Error: Missing receptor or ligand file path")
            self.print_help()
            return 1
        
        if not parsed_args['receptor_residues'] or not parsed_args['ligand_residues']:
            self.logger.error("Error: Missing receptor or ligand residue groups")
            self.print_help()
            return 1
        
        # Create project folder
        project_name = f"{parsed_args['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        project_dir = os.path.join(os.getcwd(), project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Create log file and update logger
        log_file = os.path.join(project_dir, "docking.log")
        self.logger = Logger(parsed_args['debug'], log_file)
        
        # Parse receptor and ligand files
        # Check if receptor and ligand are from the same file
        same_file = parsed_args['receptor_file'] == parsed_args['ligand_file']
        
        receptor = None
        ligand = None
        
        if same_file:
            self.logger.info(f"Parsing PDB file: {parsed_args['receptor_file']}")
            structure = self.parser.parse_file(parsed_args['receptor_file'])
            
            # Extract receptor chains
            if parsed_args['receptor_chains']:
                self.logger.info(f"Extracting receptor chains: {', '.join(parsed_args['receptor_chains'])}")
                receptor = structure.extract_chains(parsed_args['receptor_chains'])
            else:
                # If no receptor chains specified, use the entire structure as receptor
                receptor = structure
            
            # Extract ligand chains
            if parsed_args['ligand_chains']:
                self.logger.info(f"Extracting ligand chains: {', '.join(parsed_args['ligand_chains'])}")
                ligand = structure.extract_chains(parsed_args['ligand_chains'])
            else:
                # This case should not happen as we're in same file mode
                self.logger.error("Error: No ligand chains specified for same-file prealignment")
                return 1
        else:
            # Traditional mode: separate files
            self.logger.info(f"Parsing receptor file: {parsed_args['receptor_file']}")
            receptor = self.parser.parse_file(parsed_args['receptor_file'])
            
            self.logger.info(f"Parsing ligand file: {parsed_args['ligand_file']}")
            ligand = self.parser.parse_file(parsed_args['ligand_file'])
        
        # Check if parsing was successful
        if not receptor.atoms or not ligand.atoms:
            self.logger.error("\nParsing failed, exiting program")
            return 1
        
        # Initialize docking
        docking = Docking(self.logger, parsed_args['use_gpu'])
        docking.set_proteins(receptor, ligand)
        
        # Set search and penalty parameters
        docking.step_size = parsed_args['step_size']
        
        # Load force field if specified
        if parsed_args['force_field']:
            self.logger.info(f"Loading force field file: {parsed_args['force_field']}")
            if docking.load_force_field(parsed_args['force_field']):
                self.logger.info("Force field loaded successfully")
            else:
                self.logger.warning("Failed to load force field, will use default parameters for energy calculation")
        
        # Print initial statistics
        self.logger.section("Initial Protein Statistics")
        self.logger.info(f"Receptor protein: {parsed_args['receptor_file']}")
        self.logger.info(f"  Number of atoms: {len(receptor.atoms)}")
        self.logger.info(f"  Specified residue group: {', '.join(parsed_args['receptor_residues'])}")
        
        self.logger.info(f"Ligand protein: {parsed_args['ligand_file']}")
        self.logger.info(f"  Number of atoms: {len(ligand.atoms)}")
        self.logger.info(f"  Specified residue group: {', '.join(parsed_args['ligand_residues'])}")
        
        self.logger.section("Docking Parameters")
        self.logger.info(f"  Maximum search distance: {parsed_args['max_dist']} Å")
        self.logger.info(f"  Number of rotations: {parsed_args['num_rotations']}")
        self.logger.info(f"Project folder: {project_dir}")
        
        # Save parsed arguments to JSON file
        import json
        json_dest = os.path.join(project_dir, "input_config.json")
        with open(json_dest, 'w', encoding='utf-8') as f:
            json.dump(parsed_args, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved input parameters to JSON file: {json_dest}")
        
        # Perform docking
        self.logger.section("Starting Docking Conformation Search")
        scored_conformations = docking.dock(
            parsed_args['receptor_residues'],
            parsed_args['ligand_residues'],
            parsed_args['max_dist'],
            parsed_args['num_rotations']
        )
        
        # Limit the number of output conformations if specified
        if parsed_args['num_output_confs'] > 0:
            scored_conformations = scored_conformations[:parsed_args['num_output_confs']]
        
        # Print results
        self.logger.section("Docking Results")
        self.logger.info(f"Total conformations generated: {len(scored_conformations)}")
        
        if not scored_conformations:
            self.logger.error("Error: No conformations generated. Please check if the specified residue groups exist in the PDB files.")
            return 1
        
        self.logger.info(f"Top 5 Best Conformation Scores:")
        for i, (conf, score) in enumerate(scored_conformations[:5]):
            self.logger.info(f"  Conformation {i+1}: Score = {score:.2f}")
        

        # Get detailed score for the best conformation
        best_conf = scored_conformations[0][0]
        best_score, detailed_scores = docking.score_conformation(best_conf)
        
        # Calculate score breakdown percentages
        self.logger.section("Best Conformation Score Details")
        self.logger.info(f"Total score: {best_score:.2f} kcal/mol")
        self.logger.info("Score breakdown:")
        
        # Calculate total absolute value for percentage calculation
        total_abs = sum(abs(value) for value in detailed_scores.values())
        
        # Print each component with its percentage
        for component, value in detailed_scores.items():
            # Format component name for better readability
            component_name = {
            'van_der_waals': 'Van der Waals',
            'electrostatic': 'Electrostatic',
            'solvent_penalty': 'Solvent Penalty',
            'distance_penalty': 'Distance Penalty'
        }.get(component, component)
            
            # Calculate percentage
            if total_abs > 0:
                percentage = (abs(value) / total_abs) * 100
            else:
                percentage = 0.0
            
            self.logger.info(f"  {component_name}: {value:.2f} kcal/mol ({percentage:.1f}%)")
        
        self.logger.section("Saving Conformations")
        
        # Save best conformation separately (always save)
        best_file = os.path.join(project_dir, "best_conformation.pdb")
        merged_best = docking.merge_structures(receptor, best_conf)
        if self.writer.write_file(merged_best, best_file):
            self.logger.info(f"Best conformation saved to: {best_file}")
        
        # Save all conformations if save_all is True
        if parsed_args['save_all']:
            conformations_dir = os.path.join(project_dir, "conformations")
            os.makedirs(conformations_dir, exist_ok=True)
            self.logger.info(f"Saving all conformations to directory: {conformations_dir}")
            
            # Save all other conformations
            for i, (conf, score) in enumerate(scored_conformations):
                conf_file = os.path.join(conformations_dir, f"conformation_{i+1}.pdb")
                merged_conf = docking.merge_structures(receptor, conf)
                if self.writer.write_file(merged_conf, conf_file):
                    if i % 10 == 0 or i == len(scored_conformations) - 1:
                        self.logger.info(f"Saved conformation {i+1}/{len(scored_conformations)}")
        else:
            self.logger.info("Skipping saving all conformations (use --save-all option to save all)")
        
        return 0


def main_cli() -> int:
    """
    Main CLI entry point.
    
    Returns:
        int: Exit code
    """
    cli = PDBCLI()
    return cli.run(sys.argv[1:])
