#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Docking Module

Handles protein-protein docking conformation search with PyTorch acceleration.
"""

from typing import List, Tuple, Dict
import torch
from copy import deepcopy
from src.models.structure import Structure
from src.models.force_field import ForceField
from src.core.coordinate_manager import CoordinateManager
from src.core.energy_calculator import EnergyCalculator
from src.utils.logger import Logger
from tqdm import tqdm


class Docking:
    """
    Protein-protein docking conformation search with PyTorch acceleration.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
        receptor (Structure): Receptor protein structure
        ligand (Structure): Ligand protein structure
        receptor_group (List[int]): Indices of atoms in receptor group
        ligand_group (List[int]): Indices of atoms in ligand group
        max_dist (float): Maximum search distance between residue groups
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
    """
    def __init__(self, logger: Logger, use_gpu: bool = True):
        self.logger = logger
        self.receptor = None
        self.ligand = None
        self.receptor_group = []
        self.ligand_group = []
        # Search parameters
        self.step_size = 0.1  # Distance step size for intermediate conformations
        self.force_field = ForceField()
        # Set device for PyTorch calculations
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        # Initialize energy calculator
        self.energy_calculator = EnergyCalculator(
            self.force_field,
            self.device
        )
        self.logger.log(f"Using device: {self.device}")
    
    def load_force_field(self, force_field_path: str) -> bool:
        """
        Load force field parameters from XML file.
        
        Args:
            force_field_path (str): Path to the force field XML file
            
        Returns:
            bool: True if force field was successfully loaded, False otherwise
        """
        try:
            self.force_field.read_xml(force_field_path)
            self.logger.log(f"Successfully loaded force field from {force_field_path}")
            self.logger.log(f"Force field contains {len(self.force_field.atom_types)} atom types and {len(self.force_field.residues)} residues")
            return True
        except Exception as e:
            self.logger.log(f"Error loading force field: {e}")
            return False
    
    # ---------------------------
    # Structure Processing Module
    # ---------------------------
    def set_proteins(self, receptor: Structure, ligand: Structure) -> None:
        """
        Set receptor and ligand proteins.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
        """
        self.receptor = receptor
        self.ligand = ligand
    
    def identify_residue_group(self, structure: Structure, residue_ids: List[str]) -> List[int]:
        """
        Identify atoms belonging to specified residue group.
        
        Args:
            structure (Structure): Protein structure
            residue_ids (List[str]): List of residue IDs (e.g., ["A:100", "A:101"])
            
        Returns:
            List[int]: List of atom indices belonging to the residue group
        """
        atom_indices = []
        
        for i, atom in enumerate(structure.atoms):
            res_id = f"{atom.chain_id}:{atom.res_seq}"
            if res_id in residue_ids:
                atom_indices.append(i)
        
        # Log if no atoms found for residue group
        if not atom_indices:
            self.logger.log(f"Warning: No atoms found for residue group {residue_ids}")
        
        return atom_indices
    
    def set_residue_groups(self, receptor_residues: List[str], ligand_residues: List[str]) -> bool:
        """
        Set receptor and ligand residue groups.
        
        Args:
            receptor_residues (List[str]): List of receptor residue IDs
            ligand_residues (List[str]): List of ligand residue IDs
            
        Returns:
            bool: True if residue groups were successfully set, False otherwise
        """
        if self.receptor:
            self.receptor_group = self.identify_residue_group(self.receptor, receptor_residues)
            if not self.receptor_group:
                self.logger.log(f"Error: No atoms found for receptor residue group {receptor_residues}")
                return False
        
        if self.ligand:
            self.ligand_group = self.identify_residue_group(self.ligand, ligand_residues)
            if not self.ligand_group:
                self.logger.log(f"Error: No atoms found for ligand residue group {ligand_residues}")
                return False
        
        return True
    
    # ------------------------
    # Spatial Positioning Module
    # ------------------------
    def calculate_center(self, structure: Structure, atom_indices: List[int] = None) -> torch.Tensor:
        """
        Calculate center coordinates of specified atoms.
        Uses Structure's built-in calculate_geometric_center method for all atoms.
        
        Args:
            structure (Structure): Protein structure
            atom_indices (List[int], optional): List of atom indices to calculate center for.
                If None, calculate center for all atoms.
                
        Returns:
            torch.Tensor: Center coordinates as a tensor
        """
        if atom_indices is None:
            # Use Structure's built-in method for all atoms
            return structure.calculate_geometric_center().to(self.device)
        else:
            # Calculate mean for specified atoms
            coords = structure.coordinates.to(self.device)
            return torch.mean(coords[atom_indices], dim=0)
    
    def calculate_vector(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """
        Calculate vector from start to end coordinates.
        
        Args:
            start (torch.Tensor): Start coordinates
            end (torch.Tensor): End coordinates
            
        Returns:
            torch.Tensor: Vector from start to end
        """
        return end - start
        
    def align_proteins(self) -> None:
        """
        Align receptor and ligand proteins according to specified constraints.
        
        The alignment ensures:
        1. Each protein's geometric center is at origin (already done in parser.py)
        2. Protein vector (center to residue group center) aligns with z-axis
        3. Ligand is positioned along z-axis at desired separation distance
        4. Preserve internal geometric relationships within each protein
        """
        if not self.receptor or not self.ligand:
            raise ValueError("[align_proteins] Receptor or ligand structure is None")
        
        self.logger.info("===== Protein Alignment =====")
        
        # Import alignment functions
        from src.core.alignment import align_proteins as alignment_align_proteins
        
        # Use the new alignment algorithm
        alignment_align_proteins(self.receptor, self.receptor_group, self.ligand, self.ligand_group)
        
        # --------------------------
        # Position Ligand Along Z-axis
        # --------------------------
        
        # Calculate current centers after alignment using Structure's built-in method
        rec_center = self.receptor.calculate_geometric_center().to(self.device)
        lig_center = self.ligand.calculate_geometric_center().to(self.device)
        
        # Calculate group centers
        def calculate_group_center(structure, atom_indices):
            coords = structure.coordinates.to(self.device)
            return torch.mean(coords[atom_indices], dim=0)
        
        final_rec_group_center = calculate_group_center(self.receptor, self.receptor_group)
        pre_lig_group_center = calculate_group_center(self.ligand, self.ligand_group)
        
        # Update ligand center after translation
        lig_center = calculate_protein_center(self.ligand)
        
        # Calculate final ligand group center after translation
        final_lig_group_center = calculate_group_center(self.ligand, self.ligand_group)
        
        # --------------------------
        # Final verification
        # --------------------------
        final_lig_group_center = calculate_group_center(self.ligand, self.ligand_group)
        actual_distance = torch.norm(final_rec_group_center - final_lig_group_center).item()
        
        self.logger.info(f"=== final alignment results ===")
        self.logger.info(f"receptor center: ({rec_center[0]:.4f}, {rec_center[1]:.4f}, {rec_center[2]:.4f})")
        self.logger.info(f"ligand center: ({lig_center[0]:.4f}, {lig_center[1]:.4f}, {lig_center[2]:.4f})")
        self.logger.info(f"distance: {actual_distance:.4f} Å")

    # ------------------------
    # Conformation Search Module
    # ------------------------

    def _create_ligand_copy(self, base_ligand: Structure) -> Structure:
        """
        Create a copy of the ligand structure with detached coordinates.
        
        Args:
            base_ligand (Structure): Base ligand structure to copy
            
        Returns:
            Structure: Copy of the ligand structure
        """
        ligand_copy = Structure()
        ligand_copy.atoms = base_ligand.atoms.copy()
        ligand_copy.coordinates = base_ligand.coordinates.detach().clone()
        ligand_copy.other_records = base_ligand.other_records.copy()
        ligand_copy.chains = base_ligand.chains.copy()
        ligand_copy.residues = base_ligand.residues.copy()
        ligand_copy.total_charge = base_ligand.total_charge
        return ligand_copy
    
    def generate_intermediate_conformations(self, min_distance: float = 1.0) -> List[Structure]:
        """
        Generate intermediate conformations by gradually decreasing distance between proteins.
        Optimizes search efficiency by stopping early when collisions are detected based on atom distances.
        
        Args:
            min_distance (float): Minimum distance between residue groups (Å), reduced for better docking
            
        Returns:
            List[Structure]: List of intermediate ligand conformations
        """
        if not self.receptor or not self.ligand:
            return []
        
        conformations = []
        energies = []
        
        # Calculate initial centers and vectors
        rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
        lig_group_center = self.calculate_center(self.ligand, self.ligand_group)

        count = 0
        max_iterations = 1000  # Safety counter to prevent infinite loops
        
        # Generate conformations at different distances
        while current_distance > min_distance and count < max_iterations:
            # Create copy of ligand with detached coordinates
            ligand_copy = self._create_ligand_copy(self.ligand)
            
            # Calculate translation vector to achieve the current distance
            delta = rec_group_center - lig_group_center
            current_vector_length = torch.norm(delta)
            
            if current_vector_length < 1e-6:  # Avoid division by zero
                break
            
            # Calculate the target position for ligand group center
            target_lig_group_center = rec_group_center - delta * (current_distance / current_vector_length)
            
            # Calculate translation needed to move ligand group center to target position
            translation = target_lig_group_center - lig_group_center
            
            # First check for collisions before calculating energy - more efficient
            collision_threshold = 0.5  # Minimum allowed distance between any two atoms (Å), reduced for better docking
            
            # Translate ligand to the target position
            CoordinateManager(ligand_copy, device=self.device).translate_coordinates(translation)
            
            # Get coordinates of both proteins after translation
            rec_coords = self.receptor.coordinates.to(self.device, dtype=torch.float16)
            lig_coords = ligand_copy.coordinates.to(self.device, dtype=torch.float16)
            
            # Calculate pairwise distances between all atoms
            rec_expanded = rec_coords.unsqueeze(1)  # (N, 1, 3)
            lig_expanded = lig_coords.unsqueeze(0)  # (1, M, 3)
            all_distances = torch.norm(rec_expanded - lig_expanded, dim=2)  # (N, M)
            
            # Find minimum distance
            min_atom_distance = torch.min(all_distances).item()
            
            # Calculate step size before collision check
            # Dynamic step size adjustment based on energy landscape and current distance
            if len(energies) > 1:
                energy_change = energies[-1] - energies[-2]
                if energy_change < 0:  # Energy is decreasing
                    # Increase step size by up to 50%
                    current_step = min(self.step_size * 1.5, 2.0)
                else:  # Energy is increasing
                    # Decrease step size by half
                    current_step = max(self.step_size * 0.5, 0.2)
            else:
                current_step = self.step_size
            
            # Use smaller step size at close distances for better precision
            if current_distance < 10.0:
                current_step = max(current_step * 0.5, 0.2)
            
            # Check for collisions first before calculating energy - more efficient
            if min_atom_distance < collision_threshold:
                self.logger.warning(f"  Atom collision detected: Minimum atom distance {min_atom_distance:.2f} Å < threshold {collision_threshold:.2f} Å")
                # Continue to next iteration instead of breaking
                current_distance -= current_step
                continue
            
            # Calculate energy only for non-colliding conformations
            score, detailed_scores = self.score_conformation(ligand_copy)
            # Add conformation to list
            conformations.append(ligand_copy)
            energies.append(score)
            
            # Update current distance
            current_distance -= current_step
            
            # Prevent negative distance
            if current_distance < min_distance:
                current_distance = min_distance
            
            # Increment iteration counter
            count += 1
        
        self.logger.info(f"  Generated {len(conformations)} intermediate conformations")
        return conformations
    
    def search_conformations(self, num_rotations: int = 36, initial_conformation: Structure = None) -> List[Structure]:
        """
        Search for docking conformations by rotating ligand around center line.
        
        Args:
            num_rotations (int): Number of rotations (default: 36, 10 degrees each)
            initial_conformation (Structure): Initial ligand conformation to start from
            
        Returns:
            List[Structure]: List of ligand conformations
        """
        if not self.receptor or not self.ligand:
            return []
        
        conformations = []
        rotation_step = 360.0 / num_rotations
        
        # Calculate rotation axis (vector from ligand group center to receptor group center)
        rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        # Use initial_conformation if provided, otherwise use current ligand
        base_ligand = initial_conformation if initial_conformation else self.ligand
        
        # Calculate ligand group center based on base_ligand
        lig_group_center = self.calculate_center(base_ligand, self.ligand_group)
        rotation_axis = self.calculate_vector(lig_group_center, rec_group_center)
        
        # Create copies of ligand for each rotation
        for i in range(num_rotations):
            # Create copy of base ligand with detached coordinates
            ligand_copy = self._create_ligand_copy(base_ligand)
            
            # Rotate ligand
            angle = i * rotation_step
            CoordinateManager(ligand_copy, device=str(self.device)).rotate_around_axis(rotation_axis, angle, lig_group_center)
            
            conformations.append(ligand_copy)
        
        return conformations
    
    def perturb_conformation(self, initial_conformation: Structure, num_perturbations: int = 50, perturbation_magnitude: float = 0.5) -> List[Structure]:
        """
        Generate multiple conformations by applying random perturbations to the initial conformation.
        
        Args:
            initial_conformation (Structure): Initial ligand conformation to perturb
            num_perturbations (int): Number of perturbed conformations to generate
            perturbation_magnitude (float): Magnitude of random perturbations in Å
            
        Returns:
            List[Structure]: List of perturbed ligand conformations
        """
        if not initial_conformation:
            return []
        
        perturbed_conformations = []
        
        # Calculate rotation axis (vector from ligand group center to receptor group center)
        rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
        lig_group_center = self.calculate_center(initial_conformation, self.ligand_group)
        
        for i in range(num_perturbations):
            # Create copy of initial conformation
            ligand_copy = self._create_ligand_copy(initial_conformation)
            
            # Apply random translation to the entire ligand
            random_trans = torch.randn(3, device=self.device) * perturbation_magnitude
            CoordinateManager(ligand_copy, device=self.device).translate_coordinates(random_trans)
            
            # Apply random rotation around the axis between the two group centers
            random_angle = (torch.rand(1, device=self.device) - 0.5) * 20.0  # Random angle between -10 and 10 degrees
            rotation_axis = self.calculate_vector(lig_group_center, rec_group_center)
            CoordinateManager(ligand_copy, device=str(self.device)).rotate_around_axis(
                rotation_axis, random_angle.item(), lig_group_center
            )
            
            perturbed_conformations.append(ligand_copy)
        
        return perturbed_conformations
    
    def systematic_sampling(self, initial_conformation: Structure, num_perturbations: int = 50, perturbation_magnitude: float = 0.5) -> Tuple[Structure, float]:
        """
        Perform systematic sampling around the initial conformation using random perturbations.
        
        Args:
            initial_conformation (Structure): Initial ligand conformation to start from
            num_perturbations (int): Number of perturbed conformations to generate
            perturbation_magnitude (float): Magnitude of random perturbations in Å
            
        Returns:
            Tuple[Structure, float]: Best conformation and its score
        """
        # Generate perturbed conformations
        perturbed_conformations = self.perturb_conformation(initial_conformation, num_perturbations, perturbation_magnitude)
        
        # Include the initial conformation in the evaluation
        all_conformations = [initial_conformation] + perturbed_conformations
        
        # Score all conformations
        scored_conformations = []
        for conf in all_conformations:
            score, _ = self.score_conformation(conf)
            scored_conformations.append((conf, score))
        
        # Sort by score (lowest first)
        scored_conformations.sort(key=lambda x: x[1])
        
        # Return the best conformation and its score
        return scored_conformations[0]
    
    # ---------------------------
    # Scoring Module
    # ---------------------------

    
    def score_conformation(self, ligand_conformation: Structure) -> Tuple[float, Dict[str, float]]:
        """
        Score a docking conformation using the energy calculator.
        
        Args:
            ligand_conformation (Structure): Ligand conformation
            
        Returns:
            Tuple[float, Dict[str, float]]: 
                - Total conformation score (lower is better)
                - Dictionary containing detailed scores for each component
        """
        if not self.receptor:
            return float('inf'), {
                'electrostatic': 0.0
            }
        
        return self.energy_calculator.score_conformation(
            self.receptor, ligand_conformation, 
            self.receptor_group, self.ligand_group
        )
    
    # ---------------------------
    # Main Docking Module
    # ---------------------------
    def prealign(self, receptor_residues: List[str], ligand_residues: List[str], max_dist: float) -> bool:
        """
        Pre-align receptor and ligand proteins according to specified constraints.
        
        Args:
            receptor_residues (List[str]): Receptor residue IDs
            ligand_residues (List[str]): Ligand residue IDs
            max_dist (float): Maximum search distance
            
        Returns:
            bool: True if pre-alignment was successful, False otherwise
        """
        self.max_dist = max_dist
        
        # Set residue groups and check if successful
        if not self.set_residue_groups(receptor_residues, ligand_residues):
            self.logger.log("Pre-alignment failed: Could not set residue groups")
            return False
        
        # Calculate center distances before alignment
        rec_center = self.calculate_center(self.receptor)
        rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
        lig_center = self.calculate_center(self.ligand)
        lig_group_center = self.calculate_center(self.ligand, self.ligand_group)
        
        # Calculate distances
        rec_center_to_group = torch.norm(rec_center - rec_group_center).item()
        
        lig_center_to_group = torch.norm(lig_center - lig_group_center).item()
        
        # Check if max_dist is sufficient
        min_required_dist = rec_center_to_group + lig_center_to_group
        if max_dist < min_required_dist:
            self.logger.log(f"[Warning]max_dist {max_dist:.2f} Å is smaller than min_required_dist {min_required_dist:.2f} Å")
            self.logger.log(f"  Suggested max_dist should be greater than receptor center to group center distance ({rec_center_to_group:.2f} Å) + ligand center to group center distance ({lig_center_to_group:.2f} Å)")
            self.logger.log(f"  Automatically adjust max_dist to {min_required_dist + 10.0:.2f} Å")
            self.max_dist = min_required_dist + 10.0
        
        # Align proteins
        self.align_proteins()
        
        # Calculate final distances after alignment
        rec_group_center_after = self.calculate_center(self.receptor, self.receptor_group)
        lig_group_center_after = self.calculate_center(self.ligand, self.ligand_group)
        
        # Calculate actual gap distance between residue groups
        actual_gap = torch.norm(rec_group_center_after - lig_group_center_after).item()
        
        rec_center_after = self.calculate_center(self.receptor)
        lig_center_after = self.calculate_center(self.ligand)
        
        final_protein_distance = torch.norm(rec_center_after - lig_center_after).item()
        
        self.logger.info(f"  Pre-alignment completed, detailed distance information:")
        self.logger.info(f"  - Maximum search distance (target): {self.max_dist:.2f} Å")
        self.logger.info(f"  - Residue group center distance (actual): {actual_gap:.2f} Å")
        self.logger.info(f"  - Receptor center to ligand center distance: {final_protein_distance:.2f} Å")
        self.logger.info(f"  - Receptor center to residue group center distance: {rec_center_to_group:.2f} Å")
        self.logger.info(f"  - Ligand center to residue group center distance: {lig_center_to_group:.2f} Å")
        
        return True
    
    def dock(self, receptor_residues: List[str], ligand_residues: List[str], max_dist: float, 
             num_rotations: int = 36, num_perturbations: int = 1000, perturbation_magnitude: float = 0.5) -> List[Tuple[Structure, float]]:
        """
        Perform docking with specified parameters, including systematic sampling around the best conformation.
        
        Args:
            receptor_residues (List[str]): Receptor residue IDs
            ligand_residues (List[str]): Ligand residue IDs
            max_dist (float): Maximum search distance
            num_rotations (int): Number of rotations for initial conformation search
            num_perturbations (int): Number of perturbed conformations to generate during systematic sampling
            perturbation_magnitude (float): Magnitude of random perturbations in Å
            
        Returns:
            List[Tuple[Structure, float]]: List of (ligand_conformation, score) tuples, sorted by score
        """
        # Pre-align proteins
        if not self.prealign(receptor_residues, ligand_residues, max_dist):
            return []
        
        # Step 1: Generate intermediate conformations
        self.logger.section("Generating Intermediate Conformations")
        intermediate_conformations = self.generate_intermediate_conformations()
        
        if not intermediate_conformations:
            return []
        
        # Step 2: Score intermediate conformations
        self.logger.section("Scoring Intermediate Conformations")
        scored_intermediates = []
        for i, conf in enumerate(intermediate_conformations):
            if i % 5 == 0 or i == len(intermediate_conformations) - 1:
                self.logger.info(f"  Scoring conformation {i+1}/{len(intermediate_conformations)}...")
            score, _ = self.score_conformation(conf)
            scored_intermediates.append((conf, score))
        
        scored_intermediates.sort(key=lambda x: x[1])
        best_intermediate = scored_intermediates[0][0]
        best_intermediate_score = scored_intermediates[0][1]
        self.logger.info(f"Best intermediate score: {best_intermediate_score:.2f}")
        
        # Step 3: Perform rotation scan
        self.logger.section("Performing Rotation Scan")
        final_conformations = self.search_conformations(num_rotations, best_intermediate)
        
        # Step 4: Score final conformations
        self.logger.info("Scoring final conformations...")
        scored_conformations = []
        for conf in tqdm(final_conformations, desc="Scoring conformations"):
            score, _ = self.score_conformation(conf)
            scored_conformations.append((conf, score))
        
        scored_conformations.sort(key=lambda x: x[1])
        
        # Step 5: Systematic sampling around best conformation
        if scored_conformations:
            self.logger.section("Systematic Sampling")
            best_rotation_conf, best_rotation_score = scored_conformations[0]
            self.logger.info(f"Initial best score: {best_rotation_score:.2f}")
            
            # Perform systematic sampling
            best_sampled_conf, best_sampled_score = self.systematic_sampling(
                best_rotation_conf, num_perturbations, perturbation_magnitude
            )
            
            self.logger.info(f"Best sampled score: {best_sampled_score:.2f}")
            
            # Replace if better
            if best_sampled_score < best_rotation_score:
                self.logger.info("Found better conformation through systematic sampling")
                scored_conformations[0] = (best_sampled_conf, best_sampled_score)
                
                # Step 6: Second rotation scan around improved conformation
                self.logger.section("Second Rotation Scan")
                second_rotation_confs = self.search_conformations(num_rotations, best_sampled_conf)
                
                # Score additional conformations
                second_scored_confs = []
                for conf in tqdm(second_rotation_confs, desc="Scoring second rotation conformations"):
                    score, _ = self.score_conformation(conf)
                    second_scored_confs.append((conf, score))
                
                # Merge and re-sort
                all_conformations = scored_conformations + second_scored_confs
                all_conformations.sort(key=lambda x: x[1])
                return all_conformations
            else:
                self.logger.info("Original best conformation remains optimal")
        
        return scored_conformations
    
    def merge_structures(self, receptor: Structure, ligand: Structure) -> Structure:
        """
        Merge receptor and ligand structures into a single structure.
        
        Args:
            receptor (Structure): Receptor structure
            ligand (Structure): Ligand structure
            
        Returns:
            Structure: Merged structure
        """
        # Create a new structure with only non-TER/END other records
        merged = Structure()
        
        # Copy non-TER/END other records from receptor
        for record in receptor.other_records:
            if not (record.startswith('TER') or record.startswith('END')):
                merged.add_other_record(record)
        
        # Add receptor atoms with their original coordinates
        for i, atom in enumerate(receptor.atoms):
            coord = receptor.coordinates[i]
            merged.add_atom(atom, (float(coord[0]), float(coord[1]), float(coord[2])))
        
        # Add TER record after receptor atoms
        merged.add_other_record("TER")
        
        # Add ligand atoms, updating atom serial numbers
        next_serial = len(merged.atoms) + 1
        for i, atom in enumerate(ligand.atoms):
            coord = ligand.coordinates[i]
            # Create a copy of the atom with updated serial number
            new_atom = deepcopy(atom)
            new_atom.atom_serial = next_serial
            next_serial += 1
            merged.add_atom(new_atom, (float(coord[0]), float(coord[1]), float(coord[2])))
        
        # Add final END record
        merged.add_other_record("END")
        
        return merged