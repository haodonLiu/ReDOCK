#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB CLI Module

Provides command-line interface functionality for the PDB processor.
"""

import sys
from typing import List
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
    
    def parse_arguments(self, args: List[str]) -> dict:
        """
        Parse command-line arguments.
        
        Args:
            args (List[str]): Command-line arguments
            
        Returns:
            dict: Parsed arguments as a dictionary
        """
        import json
        import os
        
        parsed = {
            'debug': False,
            'mode': 'parse',  # 'parse', 'dock' or 'prealign'
            'input_file': None,
            'output_file': None,
            # Translation parameters
            'translate': False,
            'dx': 0.0,
            'dy': 0.0,
            'dz': 0.0,
            # Docking parameters
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
            'optimal_distance': 10.0,
            'save_all': False,
            'num_output_confs': 10
        }
        
        # Check for JSON argument
        json_arg = None
        if '--json' in args:
            idx = args.index('--json')
            if idx + 1 < len(args):
                json_arg = args[idx + 1]
                args.pop(idx + 1)
                args.pop(idx)
        elif '-j' in args:
            idx = args.index('-j')
            if idx + 1 < len(args):
                json_arg = args[idx + 1]
                args.pop(idx + 1)
                args.pop(idx)
        
        # Parse JSON argument if provided
        if json_arg:
            # Check if it's a file path or JSON string
            if os.path.isfile(json_arg):
                # Read from file
                with open(json_arg, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            else:
                # Parse as JSON string
                try:
                    json_data = json.loads(json_arg)
                except json.JSONDecodeError:
                    self.logger.error(f"Error: Invalid JSON string or file path: {json_arg}")
                    return parsed
            
            # Update parsed arguments with JSON data
            for key, value in json_data.items():
                if key in parsed:
                    parsed[key] = value
            
            # Set mode from JSON if not explicitly set in command line
            if 'mode' in json_data:
                parsed['mode'] = json_data['mode']
        
        # Check for debug flag
        if '--debug' in args:
            parsed['debug'] = True
            args.remove('--debug')
        elif '-d' in args:
            parsed['debug'] = True
            args.remove('-d')
        
        # Check for GPU flag
        if '--gpu' in args:
            parsed['use_gpu'] = True
            args.remove('--gpu')
        

        
        # Check for translate mode (overrides JSON if specified)
        if '--translate' in args:
            parsed['mode'] = 'translate'
            args.remove('--translate')
            
            # Parse translation values
            if '--dx' in args:
                idx = args.index('--dx')
                if idx + 1 < len(args):
                    try:
                        parsed['dx'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            if '--dy' in args:
                idx = args.index('--dy')
                if idx + 1 < len(args):
                    try:
                        parsed['dy'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            if '--dz' in args:
                idx = args.index('--dz')
                if idx + 1 < len(args):
                    try:
                        parsed['dz'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
        # Check for dock mode or prealign mode (overrides JSON if specified)
        elif '--dock' in args:
            parsed['mode'] = 'dock'
            args.remove('--dock')
        elif '--prealign' in args:
            parsed['mode'] = 'prealign'
            args.remove('--prealign')
        
        # Check for receptor file (in dock or prealign mode)
        if parsed['mode'] in ['dock', 'prealign']:
            if '--receptor' in args:
                idx = args.index('--receptor')
                if idx + 1 < len(args):
                    parsed['receptor_file'] = args[idx + 1]
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for receptor chains (in dock or prealign mode)
            if '--receptor-chains' in args:
                idx = args.index('--receptor-chains')
                if idx + 1 < len(args):
                    parsed['receptor_chains'] = args[idx + 1].split(',')
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for ligand file (in dock or prealign mode)
            if '--ligand' in args:
                idx = args.index('--ligand')
                if idx + 1 < len(args):
                    parsed['ligand_file'] = args[idx + 1]
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for ligand chains (in dock or prealign mode)
            if '--ligand-chains' in args:
                idx = args.index('--ligand-chains')
                if idx + 1 < len(args):
                    parsed['ligand_chains'] = args[idx + 1].split(',')
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for receptor residues (in dock or prealign mode)
            if '--receptor-residues' in args:
                idx = args.index('--receptor-residues')
                if idx + 1 < len(args):
                    parsed['receptor_residues'] = args[idx + 1].split(',')
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for ligand residues (in dock or prealign mode)
            if '--ligand-residues' in args:
                idx = args.index('--ligand-residues')
                if idx + 1 < len(args):
                    parsed['ligand_residues'] = args[idx + 1].split(',')
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for max_dist (in dock or prealign mode)
            if '--max-dist' in args:
                idx = args.index('--max-dist')
                if idx + 1 < len(args):
                    try:
                        parsed['max_dist'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for force field file (in dock or prealign mode)
            if '--force-field' in args:
                idx = args.index('--force-field')
                if idx + 1 < len(args):
                    parsed['force_field'] = args[idx + 1]
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for num rotations (in dock mode only)
            if parsed['mode'] == 'dock' and '--num-rotations' in args:
                idx = args.index('--num-rotations')
                if idx + 1 < len(args):
                    try:
                        parsed['num_rotations'] = int(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for step size (in dock mode only)
            if parsed['mode'] == 'dock' and '--step-size' in args:
                idx = args.index('--step-size')
                if idx + 1 < len(args):
                    try:
                        parsed['step_size'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for solvent penalty coefficient (in dock mode only)
            if parsed['mode'] == 'dock' and '--solvent-penalty' in args:
                idx = args.index('--solvent-penalty')
                if idx + 1 < len(args):
                    try:
                        parsed['solvent_penalty_coeff'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for distance penalty coefficient (in dock mode only)
            if parsed['mode'] == 'dock' and '--distance-penalty' in args:
                idx = args.index('--distance-penalty')
                if idx + 1 < len(args):
                    try:
                        parsed['distance_penalty_coeff'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for optimal distance (in dock mode only)
            if parsed['mode'] == 'dock' and '--optimal-distance' in args:
                idx = args.index('--optimal-distance')
                if idx + 1 < len(args):
                    try:
                        parsed['optimal_distance'] = float(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for number of output conformations (in dock mode only)
            if parsed['mode'] == 'dock' and '--num-output-confs' in args:
                idx = args.index('--num-output-confs')
                if idx + 1 < len(args):
                    try:
                        parsed['num_output_confs'] = int(args[idx + 1])
                    except ValueError:
                        pass
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for output file (in dock or prealign mode)
            if '-o' in args:
                idx = args.index('-o')
                if idx + 1 < len(args):
                    parsed['output_file'] = args[idx + 1]
                    args.pop(idx + 1)
                    args.pop(idx)
            
            # Check for save all conformations option (in dock mode only)
            if parsed['mode'] == 'dock' and '--save-all' in args:
                parsed['save_all'] = True
                args.remove('--save-all')
        else:
            # Parse mode: input and output files
            # Check for input file
            if len(args) >= 1:
                parsed['input_file'] = args[0]
            
            # Check for output file
            if len(args) >= 2:
                parsed['output_file'] = args[1]
        
        return parsed
    
    def print_help(self) -> None:
        """
        Print help message.
        """
        print("=== PDB Processor Usage Instructions ===")
        print("\n0. Basic Usage:")
        print("   Usage: python main.py [--help/-h]")
        print("   Function: Print this help message")
        print("   Parameter description:")
        print("     --help/-h                  Print this help message")
        print("\n1. JSON Parameter Mode (for all modes):")
        print("   Usage: python main.py [--json/-j <json_string_or_file>] [--debug/-d]")
        print("   Function: Specify all parameters through JSON string or file to simplify command line input")
        print("   Parameter description:")
        print("     --json/-j <string/file>      JSON parameter string or file path")
        print("     --debug/-d                  Enable debug mode")
        print("   JSON parameter format example：")
        print("   {")
        print("     \"mode\": \"dock\",")
        print("     \"receptor_file\": \"structure/PP5_CD.pdb\",")
        print("     \"ligand_file\": \"structure/triP-KD_AKT1.pdb\",")
        print("     \"receptor_residues\": [\"B:184\", \"B:185\"],")
        print("     \"ligand_residues\": [\"A:144\", \"A:145\"],")
        print("     \"max_dist\": 5.0,")
        print("     \"num_rotations\": 10,")
        print("     \"force_field\": \"data/force_field/ff14SB.xml\",")
        print("     \"output_file\": \"docked.pdb\",")
        print("     \"use_gpu\": true")
        print("   }")
        
        print("\n1. Basic PDB File Processing Mode：")
        print("   Usage：python main.py <pdb_file_path> [output_file_path] [--debug/-d]")
        print("   Function：Parse PDB file and display statistics, optionally write parsed data to new file")
        
        print("\n2. Protein Coordinate Translation Mode：")
        print("   Usage：python main.py --translate <pdb_file_path> [--dx <value>] [--dy <value>] [--dz <value>] [output_file_path] [--debug/-d]")
        print("   Function：Directly translate protein coordinates, supporting translation in X, Y, Z directions")
        print("   Parameter description：")
        print("     --translate                 Enable translation mode")
        print("     --dx <value>                X-axis translation distance (Å), default：0.0")
        print("     --dy <value>                Y-axis translation distance (Å), default：0.0")
        print("     --dz <value>                Z-axis translation distance (Å), default：0.0")
        print("     --debug/-d                  Enable debug mode")
        
        print("\n3. Protein Pre-alignment Mode：")
        print("   Usage：python main.py --prealign --receptor <receptor_file> --ligand <ligand_file> ")
        print("             --receptor-residues <residues> --ligand-residues <residues> [--max-dist <distance>] ")
        print("             [--force-field <file>] [-o <output_file>] [--gpu] [--debug/-d]")
        print("   Function：Pre-align receptor and ligand proteins, perform spatial arrangement based on specified residue groups, no conformation search")
        print("   Parameter description：")
        print("     --receptor <file>           Receptor protein PDB file path")
        print("     --ligand <file>             Ligand protein PDB file path")
        print("     --receptor-chains <list>    Receptor chain IDs, format：A,B,C (for same-file docking)")
        print("     --ligand-chains <list>      Ligand chain IDs, format：D,E,F (for same-file docking)")
        print("     --receptor-residues <list>  Receptor residue group, format：chain:residue,chain:residue (e.g.：A:100,A:101)")
        print("     --ligand-residues <list>    Ligand residue group, same format")
        print("     --max-dist <distance>       Maximum search distance, default：5.0 Å")
        print("     --force-field <file>        Force field XML file path for energy calculation")
        print("     --step-size <value>         Step size for intermediate conformation generation, default：1.0 Å")
        print("     --solvent-penalty <value>   Solvent penalty coefficient, default：0.1 kcal/mol per contact")
        print("     --distance-penalty <value>  Distance penalty coefficient, default：0.5 kcal/mol per Å")
        print("     --optimal-distance <value>  Optimal center distance, default：10.0 Å")
        print("     -o <file>                   Output file path")
        print("     --gpu                       Enable GPU acceleration (if available)")
        print("     --debug/-d                  Enable debug mode")
        
        print("\n4. Protein Docking Conformation Search Mode：")
        print("   Usage：python main.py --dock --receptor <receptor_file> --ligand <ligand_file> ")
        print("             --receptor-residues <residues> --ligand-residues <residues> [--max-dist <distance>] ")
        print("             [--num-rotations <number>] [--force-field <file>] [-o <output_file>] [--gpu] [--debug/-d]")
        print("   Function：Search protein docking conformations, perform spatial arrangement and scoring based on specified residue groups")
        print("   Parameter description：")
        print("     --receptor <file>           Receptor protein PDB file path")
        print("     --ligand <file>             Ligand protein PDB file path")
        print("     --receptor-chains <list>    Receptor chain IDs, format：A,B,C (for same-file docking)")
        print("     --ligand-chains <list>      Ligand chain IDs, format：D,E,F (for same-file docking)")
        print("     --receptor-residues <list>  Receptor residue group, format：chain:residue,chain:residue (e.g.：A:100,A:101)")
        print("     --ligand-residues <list>    Ligand residue group, same format")
        print("     --max-dist <distance>       Maximum search distance, default：5.0 Å")
        print("     --num-rotations <number>    Number of rotations, default：36 times (10 degrees each)")
        print("     --force-field <file>        Force field XML file path for energy calculation")
        print("     --step-size <value>         Step size for intermediate conformation generation, default：1.0 Å")
        print("     --solvent-penalty <value>   Solvent penalty coefficient, default：0.1 kcal/mol per contact")
        print("     --distance-penalty <value>  Distance penalty coefficient, default：0.5 kcal/mol per Å")
        print("     --optimal-distance <value>  Optimal center distance, default：10.0 Å")
        print("     --num-output-confs <number> Number of output conformations, default：10")
        print("     -o <file>                   Output file path")
        print("     --save-all                  Save all generated conformations, default：False")
        print("     --gpu                       Enable GPU acceleration (if available)")
        print("     --debug/-d                  Enable debug mode")
        
        print("\nExamples：")
        print("   1. Basic mode：python main.py structure/PP5_CD.pdb output.pdb")
        print("   2. Translation mode：python main.py --translate structure/PP5_CD.pdb --dx 5.0 --dz 10.0 translated.pdb")
        print("   3. Pre-alignment mode：python main.py --prealign --receptor structure/PP5_CD.pdb --ligand structure/triP-KD_AKT1.pdb ")
        print("                 --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --gap 5.0 -o prealigned.pdb")
        print("   4. Docking mode：python main.py --dock --receptor structure/PP5_CD.pdb --ligand structure/triP-KD_AKT1.pdb ")
        print("                 --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --gap 5.0 --num-rotations 10")
        print("                 -o docked.pdb --gpu")
        print("   5. Docking mode with save all conformations：python main.py --dock --receptor structure/PP5_CD.pdb --ligand structure/triP-KD_AKT1.pdb ")
        print("                 --receptor-residues B:184,B:185 --ligand-residues A:144,A:145 --num-rotations 10")
        print("                 -o docked.pdb --save-all")
        print("   6. JSON mode：python main.py -j '{\"mode\":\"parse\",\"input_file\":\"structure/PP5_CD.pdb\"}'")
        print("   7. JSON file mode：python main.py --json config.json")
    
    def run(self, args: List[str]) -> int:
        """
        Run the CLI application.
        
        Args:
            args (List[str]): Command-line arguments
            
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        # Check for help flag first, before parsing arguments
        if '--help' in args or '-h' in args:
            self.print_help()
            return 0
        
        # Parse arguments
        parsed_args = self.parse_arguments(args)
        
        # Initialize logger and components
        # Create log file path if output folder is specified or we're in dock/prealign mode
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
        
        self.logger.info(f"\nProject folder: {project_dir}")
        
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
        # Use the new method to set scoring parameters, ensuring they're passed to EnergyCalculator
        docking.set_scoring_parameters(
            parsed_args['solvent_penalty_coeff'],
            parsed_args['distance_penalty_coeff'],
            parsed_args['optimal_distance']
        )
        
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
        self.logger.info(f"\nProject folder: {project_dir}")
        
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
        
        self.logger.info(f"\nTop 5 Best Conformation Scores:")
        for i, (conf, score) in enumerate(scored_conformations[:5]):
            self.logger.info(f"  Conformation {i+1}: Score = {score:.2f}")
        

        # Get detailed score for the best conformation
        best_conf = scored_conformations[0][0]
        best_score, detailed_scores = docking.score_conformation(best_conf)
        
        # Calculate score breakdown percentages
        self.logger.section("Best Conformation Score Details")
        self.logger.info(f"Total score: {best_score:.2f} kcal/mol")
        self.logger.info("\nScore breakdown:")
        
        # Calculate total absolute value for percentage calculation
        total_abs = abs(detailed_scores['van_der_waals']) + abs(detailed_scores['electrostatic']) + \
                    abs(detailed_scores['solvent_penalty']) + abs(detailed_scores['distance_penalty'])
        
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
        
        # Write results to CSV file
        csv_file = os.path.join(project_dir, "docking_results.csv")
        self.logger.info(f"\nWriting docking results to CSV file: {csv_file}")
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Conformation ID", "Score", "File Path"])
            
            # Best conformation
            writer.writerow([1, scored_conformations[0][1], "best_conformation.pdb"])
            
            # All other conformations
            if parsed_args['save_all']:
                for i, (conf, score) in enumerate(scored_conformations[1:], start=2):
                    writer.writerow([i, score, f"conformations/conformation_{i}.pdb"])
            else:
                # Only write best conformation to CSV if save_all is False
                self.logger.info("CSV file contains only the best conformation information")
        
        self.logger.info("CSV results file written successfully")
        
        return 0


def main_cli() -> int:
    """
    Main CLI entry point.
    
    Returns:
        int: Exit code
    """
    cli = PDBCLI()
    return cli.run(sys.argv[1:])
