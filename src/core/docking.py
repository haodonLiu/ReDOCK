#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Docking Module

Handles protein-protein docking conformation search with PyTorch acceleration.
"""

import math
from typing import List, Tuple, Dict
import torch
from src.core.alignment import calculate_rotation_parameters
from src.core.coordinate_manager import CoordinateManager
from src.core.energy_calculator import EnergyCalculator
from src.models.force_field import ForceField
from src.models.structure import Structure
from src.utils.logger import Logger
from src.utils.structure_utils import residues_to_atom_indices


class Docking:
    """
    Protein-protein docking conformation search with PyTorch acceleration.
    
    Attributes:
        logger (Logger): Logger instance for debug logging
        receptor (Structure): Receptor protein structure
        ligand (Structure): Ligand protein structure
        receptor_group (List[int]): Indices of atoms in receptor group
        ligand_group (List[int]): Indices of atoms in ligand group
        device (torch.device): Device for PyTorch calculations (CPU or GPU)
    """
    def __init__(self, logger: Logger, use_gpu: bool = False):
        """
        Initialize the Docking class.
        
        Args:
            logger (Logger): Logger instance for debug logging
            use_gpu (bool, optional): Whether to use GPU for calculations
        """
        self.logger = logger
        self.receptor = None
        self.ligand = None
        self.receptor_group = []
        self.ligand_group = []
        self.step_size = 0.1  # Default step size
        self.force_field = ForceField()
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.energy_calculator = EnergyCalculator(self.force_field, self.device)
        self.logger.log(f"Using device: {self.device}")
    
    def set_proteins(self, receptor: Structure, ligand: Structure) -> None:
        """
        Set receptor and ligand proteins.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
        
        Raises:
            ValueError: If receptor or ligand is None or empty
        """
        if not receptor:
            raise ValueError("Receptor structure is None")
        
        if not ligand:
            raise ValueError("Ligand structure is None")
        
        if receptor.get_atom_count() == 0:
            raise ValueError("Receptor structure is empty")
        
        if ligand.get_atom_count() == 0:
            raise ValueError("Ligand structure is empty")
        
        self.receptor = receptor
        self.ligand = ligand
        self.logger.info(f"Set proteins: Receptor with {receptor.get_atom_count()} atoms, Ligand with {ligand.get_atom_count()} atoms")
    
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
    
    def standardize_coordinates(self, structure: Structure, target_indices: List[int], desired_direction: str = 'positive') -> None:
        """
        Standardize protein coordinates to a common coordinate system.
        Note: Structure is assumed to be already centered at origin (done during PDB parsing)
        
        Steps:
        1. Calculate protein vector (from origin to target group center)
        2. Rotate protein to align protein vector with z-axis
        
        Args:
            structure (Structure): Protein structure to standardize
            target_indices (List[int]): Indices of atoms in the target residue group
            desired_direction (str, optional): Desired direction of protein vector along z-axis. Can be 'positive' or 'negative'
        """
        if not structure or structure.get_atom_count() == 0:
            raise ValueError("Structure is None or empty")
        
        if not target_indices:
            raise ValueError("Target atom indices list is empty")
        
        if desired_direction not in ['positive', 'negative']:
            raise ValueError("desired_direction must be 'positive' or 'negative'")
        
        coord_manager = CoordinateManager(structure, device=self.device)
        
        # Step 1: Calculate protein vector (from origin to target group center)
        target_center = self.calculate_center(structure, target_indices)
        protein_vector = target_center  # Since structure is already at origin
        
        # Step 2: Rotate to align protein vector with z-axis
        # Create desired vector with the same data type as protein_vector
        if desired_direction == 'positive':
            desired_vector = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=protein_vector.dtype)
        else:
            desired_vector = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=protein_vector.dtype)
        
        # Calculate rotation parameters
        rotation_axis, angle = calculate_rotation_parameters(protein_vector, desired_vector, device=self.device)
        
        if rotation_axis is not None and angle > 1e-6:
            # Rotation center is origin
            rotation_center = torch.zeros(3, device=self.device, dtype=protein_vector.dtype)
            coord_manager.rotate_around_axis(rotation_axis, angle, rotation_center)
        
        # Verify final alignment
        final_target_center = self.calculate_center(structure, target_indices)
        final_protein_vector = final_target_center
        
        # Ensure the final direction is correct
        if desired_direction == 'positive' and final_protein_vector[2] < 0:
            # If direction is wrong, rotate 180 degrees around x-axis
            rotation_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=final_protein_vector.dtype)
            rotation_center = torch.zeros(3, device=self.device, dtype=final_protein_vector.dtype)
            coord_manager.rotate_around_axis(rotation_axis, 180.0, rotation_center)
            # Recalculate final vector after correction
            final_target_center = self.calculate_center(structure, target_indices)
            final_protein_vector = final_target_center
        elif desired_direction == 'negative' and final_protein_vector[2] > 0:
            # If direction is wrong, rotate 180 degrees around x-axis
            rotation_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=final_protein_vector.dtype)
            rotation_center = torch.zeros(3, device=self.device, dtype=final_protein_vector.dtype)
            coord_manager.rotate_around_axis(rotation_axis, 180.0, rotation_center)
            # Recalculate final vector after correction
            final_target_center = self.calculate_center(structure, target_indices)
            final_protein_vector = final_target_center
        
        self.logger.debug(f"Standardized coordinates for structure with {structure.get_atom_count()} atoms")
        self.logger.debug(f"Desired direction: {desired_direction} z-axis")
        self.logger.debug(f"Initial protein vector: ({protein_vector[0]:.4f}, {protein_vector[1]:.4f}, {protein_vector[2]:.4f})")
        self.logger.debug(f"Final protein vector: ({final_protein_vector[0]:.4f}, {final_protein_vector[1]:.4f}, {final_protein_vector[2]:.4f})")
        self.logger.debug(f"Protein vector length: {torch.norm(final_protein_vector).item():.4f} Å")
    
    def calculate_center(self, structure: Structure, atom_indices: List[int]) -> torch.Tensor:
        """
        Calculate the center of a group of atoms.
        
        Args:
            structure (Structure): Protein structure
            atom_indices (List[int]): Indices of atoms in the group
            
        Returns:
            torch.Tensor: Center coordinates of the atom group
        
        Raises:
            ValueError: If structure is None or atom_indices is empty
            IndexError: If any atom index is out of range
        """
        
        if not atom_indices:
            return torch.zeros(3, device=self.device)
        
        # Check if all atom indices are within valid range
        max_index = structure.get_atom_count() - 1
        for index in atom_indices:
            if index < 0 or index > max_index:
                raise IndexError(f"Atom index {index} is out of range (0-{max_index})")
        
        coords = structure.coordinates[atom_indices]
        return coords.mean(dim=0).to(self.device)
    
    def _validate_ligand_orientation(self) -> bool:
        """
        Validate that the ligand protein vector is properly oriented towards the receptor residue group.
        
        Returns:
            bool: True if ligand is properly oriented, False otherwise
        """
        # Calculate vectors
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
        ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
        
        # Calculate ligand protein vector (from ligand center to ligand residue group)
        ligand_protein_vector = ligand_group_center - ligand_geometric_center
        
        # Calculate vector from ligand residue group to receptor residue group
        lig_to_rec_vector = receptor_group_center - ligand_group_center
        
        # Check if vectors are properly oriented using dot product
        if torch.norm(ligand_protein_vector) > 1e-6 and torch.norm(lig_to_rec_vector) > 1e-6:
            normalized_lig_protein_vector = ligand_protein_vector / torch.norm(ligand_protein_vector)
            normalized_lig_to_rec_vector = lig_to_rec_vector / torch.norm(lig_to_rec_vector)
            
            dot_product = torch.dot(normalized_lig_protein_vector, normalized_lig_to_rec_vector)
            
            # If dot product is positive, ligand is facing away from receptor
            # We want ligand to face receptor, so dot product should be negative
            return dot_product < 0
        
        return True

    @property
    def receptor_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the receptor.
        
        Returns:
            torch.Tensor: Center coordinates of the receptor
        """
        if not self.receptor:
            return torch.zeros(3, device=self.device)
        return self.receptor.calculate_geometric_center().to(self.device)
    
    @property
    def ligand_center(self) -> torch.Tensor:
        """
        Calculate the geometric center of the ligand.
        
        Returns:
            torch.Tensor: Center coordinates of the ligand
        """
        if not self.ligand:
            return torch.zeros(3, device=self.device)
        return self.ligand.calculate_geometric_center().to(self.device)
    
    @property
    def rec_group_center(self) -> torch.Tensor:
        """
        Calculate the center of the receptor group.
        
        Returns:
            torch.Tensor: Center coordinates of the receptor group
        """
        return self.calculate_center(self.receptor, self.receptor_group)
    
    @property
    def lig_group_center(self) -> torch.Tensor:
        """
        Calculate the center of the ligand group.
        
        Returns:
            torch.Tensor: Center coordinates of the ligand group
        """
        return self.calculate_center(self.ligand, self.ligand_group)
    
    # ------------------------
    # Spatial Positioning Module
    # ------------------------
    def align_proteins(self) -> None:
        """
        Align receptor and ligand proteins according to specified constraints.
        
        The alignment ensures:
        1. Protein vector (center to residue group center) aligns with z-axis
        2. Directly positions ligand near receptor residue group (2-5Å range)
        3. Ensures ligand residue group faces receptor residue group
        4. Preserve internal geometric relationships within each protein
        """
        if not self.receptor or not self.ligand:
            raise ValueError("[align_proteins] Receptor or ligand structure is None")
        
        if not self.receptor_group or not self.ligand_group:
            raise ValueError("Receptor or ligand group atom indices are empty")
        
        self.logger.info("===== Protein Alignment =====")
        
        # Step 1: Standardize receptor coordinates (align with positive z-axis)
        self.logger.info("Standardizing receptor coordinates...")
        self.standardize_coordinates(self.receptor, self.receptor_group, desired_direction='positive')
        
        # Step 2: Standardize ligand coordinates (align with negative z-axis)
        self.logger.info("Standardizing ligand coordinates...")
        self.standardize_coordinates(self.ligand, self.ligand_group, desired_direction='negative')
        
        # Step 3: Calculate protein vector lengths
        rec_length = torch.norm(self.rec_group_center).item()  # Since receptor is at origin
        lig_length = torch.norm(self.lig_group_center).item()  # Since ligand is at origin
        self.logger.info("=== Initial Position ===")
        self.logger.info(f"Receptor length: {rec_length:.4f} Å")
        self.logger.info(f"Ligand length: {lig_length:.4f} Å")

        # --------------------------
        # Step 4: Position ligand near receptor residue group
        # --------------------------
        # Create coordinate manager for ligand
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Calculate receptor residue group center
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        # Calculate ligand residue group center (relative to ligand center)
        ligand_group_center_local = self.calculate_center(self.ligand, self.ligand_group) - self.ligand_center
        
        # Position ligand such that its residue group is near receptor residue group with safe initial distance
        # Start with a much larger offset to avoid atom overlap
        base_target_position = receptor_group_center - ligand_group_center_local
        
        # Try multiple initial offsets to find a safe starting position
        safe_offset_found = False
        # Try larger initial offsets first
        for offset_z in [20.0, 15.0, 10.0, 8.0, 6.0]:
            # Add offset along positive z-axis
            offset = torch.tensor([0.0, 0.0, offset_z], device=self.device)
            target_position = base_target_position + offset
            current_ligand_center = self.ligand_center
            translation = target_position - current_ligand_center
            coord_manager.translate_coordinates(translation)
            
            # Calculate initial vdw energy
            initial_vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            self.logger.info(f"Testing offset {offset_z}Å: VDW energy = {initial_vdw_energy:.2f} kcal/mol")
            
            # If vdw energy is reasonable for total protein interaction, use this position
            # For large proteins, total VDW can be positive but manageable
            if initial_vdw_energy < 50000 and not math.isnan(initial_vdw_energy):
                safe_offset_found = True
                self.logger.info(f"Using initial offset of {offset_z}Å with VDW energy: {initial_vdw_energy:.2f} kcal/mol")
                break
            
            # If energy is still too high, try a different offset
            # Reset position first
            coord_manager.translate_coordinates(-translation)
        
        if not safe_offset_found:
            # If no safe offset found, use the largest offset and let the adjustment logic handle it
            offset = torch.tensor([0.0, 0.0, 20.0], device=self.device)
            target_position = base_target_position + offset
            current_ligand_center = self.ligand_center
            translation = target_position - current_ligand_center
            coord_manager.translate_coordinates(translation)
            initial_vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            self.logger.warning(f"Could not find safe initial offset. Using 20Å offset with VDW energy: {initial_vdw_energy:.2f} kcal/mol")
        
        # Calculate initial vdw energy again after final positioning
        initial_vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        self.logger.info(f"Final initial VDW energy: {initial_vdw_energy:.2f} kcal/mol")
        
        # If vdw energy is too high (>50000) or nan, adjust position
        if initial_vdw_energy > 50000 or math.isnan(initial_vdw_energy):
            self.logger.info("Initial position has high VDW energy. Adjusting...")
            
            # Try small adjustments in different directions
            adjustment_directions = [
                torch.tensor([0.0, 0.0, 2.0], device=self.device),  # Move along positive z-axis
                torch.tensor([0.0, 0.0, -2.0], device=self.device),  # Move along negative z-axis
                torch.tensor([1.0, 1.0, 1.0], device=self.device) / torch.sqrt(torch.tensor(3.0)),  # Diagonal
                torch.tensor([-1.0, -1.0, -1.0], device=self.device) / torch.sqrt(torch.tensor(3.0)),  # Opposite diagonal
                torch.tensor([1.0, 0.0, 0.0], device=self.device),  # Move along x-axis
                torch.tensor([0.0, 1.0, 0.0], device=self.device)   # Move along y-axis
            ]
            
            best_vdw_energy = initial_vdw_energy
            best_translation = torch.zeros(3, device=self.device)
            
            for direction in adjustment_directions:
                # Try adjustment with different distances
                for distance in [1.0, 2.0, 3.0]:
                    # Save current position
                    current_coords = self.ligand.coordinates.clone()
                    
                    # Apply adjustment
                    adjust_translation = direction * distance
                    coord_manager.translate_coordinates(adjust_translation)
                    
                    # Calculate vdw energy
                    vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                    
                    # Check if this is better
                    if vdw_energy < best_vdw_energy:
                        best_vdw_energy = vdw_energy
                        best_translation = adjust_translation
                    
                    # Restore position
                    self.ligand.coordinates = current_coords.clone()
            
            # Apply best adjustment
            if best_vdw_energy < initial_vdw_energy:
                self.logger.info(f"Applying adjustment of {torch.norm(best_translation).item():.2f} Å, new VDW energy: {best_vdw_energy:.2f} kcal/mol")
                coord_manager.translate_coordinates(best_translation)
            else:
                self.logger.warning(f"Could not find better position. Using initial position with VDW energy: {initial_vdw_energy:.2f} kcal/mol")
        
        # --------------------------
        # Step 5: Final verification
        # --------------------------
        # Calculate final positions
        final_receptor_center = self.receptor_center
        final_ligand_center = self.ligand_center
        final_receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        final_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
        
        # Calculate distance between residue groups and protein centers
        group_distance = torch.norm(final_receptor_group_center - final_ligand_group_center).item()
        center_distance = torch.norm(final_receptor_center - final_ligand_center).item()
        
        # Calculate final vdw energy
        final_vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        
        # Ensure residue group distance is less than protein center distance
        # If not, adjust ligand position
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        adjustment_applied = False
        
        if group_distance >= center_distance:
            self.logger.info(f"Residue group distance ({group_distance:.4f} Å) >= protein center distance ({center_distance:.4f} Å), adjusting ligand position")
            
            # Calculate direction from ligand center to ligand residue group
            ligand_residue_dir = final_ligand_group_center - final_ligand_center
            ligand_residue_dir_normalized = ligand_residue_dir / torch.norm(ligand_residue_dir)
            
            # Move ligand in the direction away from its residue group relative to receptor
            # This will increase the protein center distance while keeping residue group distance similar
            adjustment_distance = center_distance - group_distance + 2.0  # Add 2Å buffer
            adjustment = ligand_residue_dir_normalized * adjustment_distance
            
            # Apply adjustment
            coord_manager.translate_coordinates(adjustment)
            adjustment_applied = True
            
            # Recalculate distances after adjustment
            final_receptor_center = self.receptor_center
            final_ligand_center = self.ligand_center
            final_receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
            final_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            
            group_distance = torch.norm(final_receptor_group_center - final_ligand_group_center).item()
            center_distance = torch.norm(final_receptor_center - final_ligand_center).item()
            final_vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            self.logger.info(f"After adjustment: Residue group distance = {group_distance:.4f} Å, Protein center distance = {center_distance:.4f} Å")
        
        self.logger.info("=== Final Position ===")
        self.logger.info(f"Receptor center: ({final_receptor_center[0]:.4f}, {final_receptor_center[1]:.4f}, {final_receptor_center[2]:.4f})")
        self.logger.info(f"Ligand center: ({final_ligand_center[0]:.4f}, {final_ligand_center[1]:.4f}, {final_ligand_center[2]:.4f})")
        self.logger.info(f"Distance between centers: {center_distance:.4f} Å")
        self.logger.info(f"Distance between residue groups: {group_distance:.4f} Å")
        self.logger.info(f"Final VDW energy: {final_vdw_energy:.2f}")
        
        # Verify protein vector directions
        rec_vector = self.rec_group_center  # Since receptor is at origin
        lig_vector = self.lig_group_center  # Since ligand is at origin
        
        self.logger.info(f"Receptor vector direction: ({rec_vector[0]:.4f}, {rec_vector[1]:.4f}, {rec_vector[2]:.4f})")
        self.logger.info(f"Ligand vector direction: ({lig_vector[0]:.4f}, {lig_vector[1]:.4f}, {lig_vector[2]:.4f})")
        
        # Final verification: Ensure residue group distance <= ligand center to receptor residue group distance
        ligand_to_receptor_group = torch.norm(final_ligand_center - final_receptor_group_center).item()
        self.logger.info(f"Ligand center to receptor residue group distance: {ligand_to_receptor_group:.4f} Å")
        
        # This is the correct condition: residue group distance should be less than ligand center to receptor residue group distance
        if group_distance <= ligand_to_receptor_group:
            self.logger.info(f"✓ Residue group distance ({group_distance:.4f} Å) <= ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Valid configuration")
        else:
            self.logger.warning(f"⚠ Residue group distance ({group_distance:.4f} Å) > ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Consider adjusting docking parameters")
    
    def generate_initial_conformations(self, num_conformations: int = 36) -> None:
        """
        Generate initial conformations of ligand along xz plane around receptor.
        Evaluate each conformation and select the best one as initial position.
        
        Args:
            num_conformations (int): Number of conformations to generate (default: 36)
        """
        if not self.receptor or not self.ligand:
            raise ValueError("Receptor or ligand structure is None")
        
        if not self.receptor_group or not self.ligand_group:
            raise ValueError("Receptor or ligand group atom indices are empty")
        
        self.logger.info("===== Generating Initial Conformations =====")
        
        # Get receptor center
        receptor_center = self.receptor.calculate_geometric_center().to(self.device)
        
        # Calculate initial radius (distance between centers)
        initial_ligand_center = self.ligand.calculate_geometric_center().to(self.device)
        initial_radius = torch.norm(receptor_center - initial_ligand_center).item()
        
        self.logger.info(f"Receptor center: ({receptor_center[0]:.4f}, {receptor_center[1]:.4f}, {receptor_center[2]:.4f})")
        self.logger.info(f"Initial ligand center: ({initial_ligand_center[0]:.4f}, {initial_ligand_center[1]:.4f}, {initial_ligand_center[2]:.4f})")
        self.logger.info(f"Initial radius: {initial_radius:.4f} Å")
        self.logger.info(f"Generating {num_conformations} conformations along xz plane...")
        
        # Generate angles evenly distributed around xz plane
        angles = torch.linspace(0, 360, num_conformations, device=self.device)[:-1]  # Exclude 360 to avoid duplication
        
        # Store best conformation info
        best_score = float('inf')
        best_conformation = None
        best_position = None
        
        # Create coordinate manager for ligand
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Save original position
        original_position = self.ligand.coordinates.clone()
        
        # Generate and evaluate conformations
        for i, angle_deg in enumerate(angles):
            # Convert angle to radians
            angle_rad = math.radians(angle_deg)
            
            # Calculate target position on xz plane
            x = initial_radius * math.cos(angle_rad)
            z = initial_radius * math.sin(angle_rad)
            y = 0.0  # Stay on xz plane
            
            # Calculate translation vector
            target_position = receptor_center + torch.tensor([x, y, z], device=self.device, dtype=receptor_center.dtype)
            ligand_center = self.ligand.calculate_geometric_center().to(self.device)
            translation = target_position - ligand_center
            
            # Move ligand to target position
            coord_manager.translate_coordinates(translation)
            
            # Validate ligand orientation before evaluating conformation
            if not self._validate_ligand_orientation():
                # Try to correct orientation by rotating 180 degrees around y-axis
                flip_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
                coord_manager.rotate_around_axis(flip_axis, 180.0, ligand_geometric_center)
                
                # Check again if orientation is valid
                if not self._validate_ligand_orientation():
                    # Reset to original position for next iteration
                    self.ligand.coordinates = original_position.clone()
                    continue
            
            # Evaluate conformation
            vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            # Calculate distance between target groups
            rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
            lig_group_center = self.calculate_center(self.ligand, self.ligand_group)
            group_distance = torch.norm(rec_group_center - lig_group_center).item()
            
            # Calculate score (weighted sum of energy and distance)
            score = vdw_energy + (group_distance * 10)  # Weight distance to prioritize closer positions
            
            # Log conformation info
            if i % 10 == 0:
                self.logger.info(f"Conformation {i+1}/{num_conformations-1}: Angle = {angle_deg:.1f}°, VDW = {vdw_energy:.2f}, Distance = {group_distance:.4f} Å, Score = {score:.2f}")
            
            # Update best conformation
            if score < best_score and vdw_energy < 50000.0:
                best_score = score
                best_conformation = self.ligand.coordinates.clone()
                best_position = target_position
                self.logger.info(f"New best conformation at angle {angle_deg:.1f}°: Score = {score:.2f}")
            
            # Reset to original position for next iteration
            self.ligand.coordinates = original_position.clone()
        
        # Set best conformation
        if best_conformation is not None:
            self.ligand.coordinates = best_conformation
            best_lig_group_center = self.calculate_center(self.ligand, self.ligand_group)
            best_group_distance = torch.norm(receptor_center - best_lig_group_center).item()
            best_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            self.logger.info("=== Best Conformation Selected ===")
            self.logger.info(f"Best score: {best_score:.2f}")
            self.logger.info(f"Best ligand position: ({best_position[0]:.4f}, {best_position[1]:.4f}, {best_position[2]:.4f})")
            self.logger.info(f"Best group distance: {best_group_distance:.4f} Å")
            self.logger.info(f"Best VDW energy: {best_vdw:.2f}")
        else:
            self.logger.warning("No valid conformation found. Using original position.")
    
    def optimize_position(self) -> None:
        """
        Optimize ligand position in 3D space to align with receptor target group.
        Moves ligand towards receptor target group while ensuring van der Waals energy stays reasonable.
        """
        if not self.receptor or not self.ligand:
            raise ValueError("Receptor or ligand structure is None")
        
        if not self.receptor_group or not self.ligand_group:
            raise ValueError("Receptor or ligand group atom indices are empty")
        
        self.logger.info("===== Optimizing Ligand Position =====")
        
        # Get receptor and ligand group centers
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        initial_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
        
        self.logger.info(f"Initial receptor group center: ({receptor_group_center[0]:.4f}, {receptor_group_center[1]:.4f}, {receptor_group_center[2]:.4f})")
        self.logger.info(f"Initial ligand group center: ({initial_ligand_group_center[0]:.4f}, {initial_ligand_group_center[1]:.4f}, {initial_ligand_group_center[2]:.4f})")
        
        # Create coordinate manager for ligand
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Initial distance and energy
        initial_distance = torch.norm(receptor_group_center - initial_ligand_group_center).item()
        initial_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        
        self.logger.info(f"Initial distance: {initial_distance:.4f} Å")
        self.logger.info(f"Initial VDW energy: {initial_vdw:.2f}")
        
        # Optimization parameters
        max_iterations = 300
        step_size = 0.5  # Small step size to avoid sudden collisions
        min_step_size = 0.1
        vdw_threshold = 50000.0  # Higher threshold for initial approach
        target_distance = 6.0  # Target distance for residue groups
        
        best_distance = initial_distance
        best_position = self.ligand.coordinates.clone()
        best_vdw = initial_vdw
        
        # Start moving towards receptor
        for iteration in range(max_iterations):
            # Calculate current ligand group center
            current_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            
            # Calculate direction vector from ligand to receptor group
            direction = receptor_group_center - current_ligand_group_center
            direction_norm = torch.norm(direction)
            
            if direction_norm < target_distance:
                self.logger.info(f"Converged: Distance below target ({direction_norm:.4f} Å)")
                break
            
            # Normalize direction vector
            direction_normalized = direction / direction_norm
            
            # Ensure ligand protein vector continues to face the receptor residue group
            # Calculate ligand geometric center
            ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
            
            # Calculate ligand protein vector (from ligand center to ligand residue group)
            ligand_protein_vector = current_ligand_group_center - ligand_geometric_center
            
            # Calculate vector from ligand to receptor residue group
            lig_to_rec_vector = receptor_group_center - current_ligand_group_center
            
            # Check orientation using dot product
            if torch.norm(ligand_protein_vector) > 1e-6 and torch.norm(lig_to_rec_vector) > 1e-6:
                normalized_lig_protein_vector = ligand_protein_vector / torch.norm(ligand_protein_vector)
                normalized_lig_to_rec_vector = lig_to_rec_vector / torch.norm(lig_to_rec_vector)
                
                dot_product = torch.dot(normalized_lig_protein_vector, normalized_lig_to_rec_vector)
                
                # If dot product is positive, ligand protein vector is facing away from receptor
                if dot_product > 0:
                    # Rotate 180 degrees around y-axis to correct orientation
                    flip_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                    coord_manager.rotate_around_axis(flip_axis, 180.0, ligand_geometric_center)
                    
                    # Recalculate after rotation
                    current_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                    direction = receptor_group_center - current_ligand_group_center
                    direction_norm = torch.norm(direction)
                    if direction_norm < target_distance:
                        self.logger.info(f"Converged: Distance below target ({direction_norm:.4f} Å)")
                        break
                    direction_normalized = direction / direction_norm
            
            # Save current position
            current_position = self.ligand.coordinates.clone()
            
            # Calculate step vector
            step_vector = direction_normalized * step_size
            
            # Move ligand towards receptor
            coord_manager.translate_coordinates(step_vector)
            
            # Calculate new distance and energy
            new_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            new_distance = torch.norm(receptor_group_center - new_ligand_group_center).item()
            new_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            # Handle NaN vdw energy (usually due to exact atom overlap)
            if math.isnan(new_vdw):
                self.logger.warning("Got NaN VDW energy, reverting position")
                self.ligand.coordinates = current_position
                # Try smaller step size
                step_size = max(step_size * 0.5, min_step_size)
                continue
            
            # Check if position is acceptable
            if new_vdw < vdw_threshold:
                # Update best position if closer and VDW is reasonable
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_position = self.ligand.coordinates.clone()
                    best_vdw = new_vdw
                    
                    # Log improvement
                    if iteration % 50 == 0:
                        self.logger.info(f"Iteration {iteration}: Distance = {new_distance:.4f} Å, VDW = {new_vdw:.2f}, Step size = {step_size:.4f}")
                
                # Gradually reduce step size as we get closer
                if new_distance < best_distance * 0.95:
                    step_size = max(step_size * 0.9, min_step_size)
            else:
                # Position not acceptable, revert and try smaller step
                self.ligand.coordinates = current_position
                step_size = max(step_size * 0.7, min_step_size)
                
                # If still too high, try lateral movement
                if step_size <= min_step_size:
                    # Try slight lateral movement to find better path
                    found_better_path = False
                    for angle in [-20.0, 20.0]:
                        # Rotate direction vector slightly
                        angle_rad = math.radians(angle)
                        rotated_direction = torch.tensor([
                            direction_normalized[0] * math.cos(angle_rad) - direction_normalized[1] * math.sin(angle_rad),
                            direction_normalized[0] * math.sin(angle_rad) + direction_normalized[1] * math.cos(angle_rad),
                            direction_normalized[2]
                        ], device=self.device)
                        
                        # Try this direction
                        lateral_step = rotated_direction * step_size
                        coord_manager.translate_coordinates(lateral_step)
                        
                        lateral_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                        lateral_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
                        lateral_distance = torch.norm(receptor_group_center - lateral_ligand_center).item()
                        
                        if lateral_vdw < vdw_threshold and lateral_distance < best_distance:
                            # Found better path
                            best_distance = lateral_distance
                            best_position = self.ligand.coordinates.clone()
                            best_vdw = lateral_vdw
                            found_better_path = True
                            break
                        else:
                            # Revert
                            self.ligand.coordinates = current_position
                    
                    if not found_better_path:
                        # No better path found, break early
                        break
            
            # Check if we've reached target distance
            if best_distance < target_distance:
                break
        
        # Set best position
        self.ligand.coordinates = best_position
        
        # Final verification and adjustment
        final_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
        final_distance = torch.norm(receptor_group_center - final_ligand_group_center).item()
        final_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        
        # If final VDW is still too high, move slightly away
        if final_vdw > vdw_threshold:
            self.logger.info(f"Final VDW energy ({final_vdw:.2f}) is too high, adjusting...")
            # Calculate direction away from receptor
            direction_away = final_ligand_group_center - receptor_group_center
            direction_away_normalized = direction_away / torch.norm(direction_away)
            
            # Move in small steps until VDW is acceptable
            for adjust_step in [0.5, 1.0, 1.5, 2.0, 3.0]:
                adjustment = direction_away_normalized * adjust_step
                coord_manager.translate_coordinates(adjustment)
                
                adjusted_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                adjusted_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                adjusted_distance = torch.norm(receptor_group_center - adjusted_ligand_group_center).item()
                
                if adjusted_vdw < vdw_threshold:
                    final_vdw = adjusted_vdw
                    final_distance = adjusted_distance
                    final_ligand_group_center = adjusted_ligand_group_center
                    self.logger.info(f"Adjusted position: Distance = {final_distance:.4f} Å, VDW = {final_vdw:.2f}")
                    break
        
        # Removed old condition check - we now use the correct condition below (lines 860-893)
        # The correct condition is: residue group distance <= ligand center to receptor residue group distance
        
        # Calculate distances for verification
        final_receptor_center = self.receptor_center
        final_ligand_center = self.ligand_center
        center_distance = torch.norm(final_receptor_center - final_ligand_center).item()
        
        # Calculate ligand center to receptor residue group distance
        ligand_to_receptor_group = torch.norm(final_ligand_center - receptor_group_center).item()
        
        self.logger.info("=== Final Position ===")
        self.logger.info(f"Final receptor group center: ({receptor_group_center[0]:.4f}, {receptor_group_center[1]:.4f}, {receptor_group_center[2]:.4f})")
        self.logger.info(f"Final ligand group center: ({final_ligand_group_center[0]:.4f}, {final_ligand_group_center[1]:.4f}, {final_ligand_group_center[2]:.4f})")
        self.logger.info(f"Final residue group distance: {final_distance:.4f} Å")
        self.logger.info(f"Final protein center distance: {center_distance:.4f} Å")
        self.logger.info(f"Final ligand center to receptor residue group distance: {ligand_to_receptor_group:.4f} Å")
        self.logger.info(f"Final VDW energy: {final_vdw:.2f}")
        
        # Verify residue group distance <= ligand center to receptor residue group distance
        # This is the correct condition: residue group distance should be less than ligand center to receptor residue group distance
        if final_distance <= ligand_to_receptor_group:
            self.logger.info(f"✓ Residue group distance ({final_distance:.4f} Å) <= ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Valid configuration")
        else:
            self.logger.warning(f"⚠ Residue group distance ({final_distance:.4f} Å) > ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Adjusting ligand position")
            
            # Calculate adjustment vector to move ligand so that residue group distance <= ligand center to receptor residue group distance
            # We'll move ligand away from receptor residue group
            direction = final_ligand_group_center - receptor_group_center
            direction_normalized = direction / torch.norm(direction)
            
            # Calculate required adjustment distance to satisfy the condition
            # We need to ensure that final_distance <= ligand_to_receptor_group
            # For simplicity, we'll move ligand far enough to ensure the condition is met
            required_distance = (final_distance - ligand_to_receptor_group) + 5.0  # Move away from receptor
            adjustment = direction_normalized * required_distance
            
            # Apply adjustment
            coord_manager.translate_coordinates(adjustment)
            
            # Recalculate distances after adjustment
            final_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            final_distance = torch.norm(receptor_group_center - final_ligand_group_center).item()
            final_ligand_center = self.ligand_center
            ligand_to_receptor_group = torch.norm(final_ligand_center - receptor_group_center).item()
            final_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            self.logger.info(f"After adjustment: Residue group distance = {final_distance:.4f} Å, Ligand center to receptor residue group distance = {ligand_to_receptor_group:.4f} Å, VDW = {final_vdw:.2f}")
            
            if final_distance <= ligand_to_receptor_group:
                self.logger.info(f"✓ After adjustment: Residue group distance ({final_distance:.4f} Å) <= ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Valid configuration")
            else:
                self.logger.warning(f"⚠ Even after adjustment, residue group distance ({final_distance:.4f} Å) > ligand center to receptor residue group distance ({ligand_to_receptor_group:.4f} Å) - Consider adjusting docking parameters")
        
        if final_vdw > vdw_threshold:
            self.logger.warning(f"Final VDW energy ({final_vdw:.2f}) exceeds threshold ({vdw_threshold:.2f})")
        else:
            self.logger.info(f"Final VDW energy ({final_vdw:.2f}) is within threshold ({vdw_threshold:.2f})")
    
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
    
    def _translation_scan(self, step_size: float, distance_range: Tuple[float, float], vdw_threshold: float = 10000.0) -> List[Structure]:
        """
        Perform systematic translation scan of ligand around receptor residue group.
        
        Args:
            step_size (float): Translation step size in Å
            distance_range (Tuple[float, float]): Min and max distance from receptor residue group
            vdw_threshold (float): VDW energy threshold for collision detection
            
        Returns:
            List[Structure]: List of valid conformations from translation scan
        """
        self.logger.info(f"=== Translation Scan (Step size: {step_size} Å) ===")
        
        conformations = []
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        # Calculate ligand residue group position relative to ligand geometric center
        ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
        ligand_group_offset = self.calculate_center(self.ligand, self.ligand_group) - ligand_geometric_center
        
        # Generate grid of positions around receptor residue group
        min_dist, max_dist = distance_range
        grid_size = int((max_dist - min_dist) / step_size) + 1
        
        # Save original ligand position
        original_position = self.ligand.coordinates.clone()
        
        # Create coordinate manager
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Generate positions around receptor residue group
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Calculate position offset from receptor residue group
                    dx = (i - grid_size//2) * step_size
                    dy = (j - grid_size//2) * step_size
                    dz = (k - grid_size//2) * step_size
                    
                    # Calculate target ligand residue group position
                    target_ligand_group_position = receptor_group_center + torch.tensor([dx, dy, dz], device=self.device)
                    
                    # Calculate required ligand geometric center position
                    target_ligand_center = target_ligand_group_position - ligand_group_offset
                    
                    # Calculate translation vector to move ligand to this position
                    translation = target_ligand_center - ligand_geometric_center
                    
                    # Apply translation
                    self.ligand.coordinates = original_position.clone()
                    coord_manager.translate_coordinates(translation)
                    
                    # Calculate current distance from receptor residue group
                    current_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                    current_distance = torch.norm(receptor_group_center - current_ligand_group_center).item()
                    
                    # Check if within distance range
                    if current_distance < min_dist or current_distance > max_dist:
                        continue
                    
                    # Validate ligand orientation
                    if not self._validate_ligand_orientation():
                        # Try to correct orientation by rotating 180 degrees around y-axis
                        ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
                        CoordinateManager(self.ligand, device=self.device).rotate_around_axis(
                            torch.tensor([0.0, 1.0, 0.0], device=self.device),
                            180.0,
                            ligand_geometric_center
                        )
                        
                        # Check if orientation is now valid
                        if not self._validate_ligand_orientation():
                            continue
                    
                    # Calculate VDW energy
                    vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                    
                    # Get centers for distance calculations
                    receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
                    ligand_center = self.ligand.calculate_geometric_center().to(self.device)
                    ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                    
                    # Calculate distances
                    ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
                    residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
                    
                    # Check if VDW energy is within threshold and distance conditions are met
                    # During search: use relaxed VDW threshold, only enforce distance range and orientation
                    # Conditions: 1) VDW < vdw_threshold, 2) 2A <= residue_group_distance <= 5A, 3) residue_group_distance <= ligand_to_receptor_group
                    if (vdw_energy < vdw_threshold and 
                        2.0 <= residue_group_distance <= 5.0 and 
                        residue_group_distance <= ligand_to_receptor_group and 
                        not math.isnan(vdw_energy)):
                        # Create copy of valid conformation
                        valid_conformation = self._create_ligand_copy(self.ligand)
                        conformations.append(valid_conformation)
        
        self.logger.info(f"Found {len(conformations)} valid conformations from translation scan")
        
        # Restore original ligand position
        self.ligand.coordinates = original_position.clone()
        
        return conformations

    def _rotation_scan(self, conformations: List[Structure], num_rotations: int = 36, vdw_threshold: float = 10000.0) -> List[Structure]:
        """
        Perform rotation scan on valid conformations from translation scan.
        
        Args:
            conformations (List[Structure]): List of valid conformations from translation scan
            num_rotations (int): Number of rotations per conformation
            vdw_threshold (float): VDW energy threshold for collision detection
            
        Returns:
            List[Structure]: List of valid conformations after rotation scan
        """
        self.logger.info(f"=== Rotation Scan (Rotations: {num_rotations}) ===")
        
        rotated_conformations = []
        rotation_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device)  # Rotate around z-axis
        rotation_step = 360.0 / num_rotations
        
        for base_conformation in conformations:
            # Calculate rotation center (ligand geometric center)
            ligand_center = base_conformation.calculate_geometric_center().to(self.device)
            
            # Generate rotations
            for i in range(num_rotations):
                # Create copy of base conformation
                ligand_copy = self._create_ligand_copy(base_conformation)
                
                # Apply rotation
                angle = i * rotation_step
                CoordinateManager(ligand_copy, device=self.device).rotate_around_axis(rotation_axis, angle, ligand_center)
                
                # Calculate VDW energy
                vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, ligand_copy)
                
                # Get centers for distance calculations
                receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
                ligand_center = ligand_copy.calculate_geometric_center().to(self.device)
                ligand_group_center = self.calculate_center(ligand_copy, self.ligand_group)
                
                # Calculate distances
                ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
                residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
                
                # Check if VDW energy is within threshold and distance conditions are met
                # During search: use relaxed VDW threshold, only enforce distance range and orientation
                # Conditions: 1) VDW < vdw_threshold, 2) 2A <= residue_group_distance <= 5A, 3) residue_group_distance <= ligand_to_receptor_group
                if (vdw_energy < vdw_threshold and 
                    2.0 <= residue_group_distance <= 5.0 and 
                    residue_group_distance <= ligand_to_receptor_group and 
                    not math.isnan(vdw_energy)):
                    # Validate orientation after rotation
                    # Temporarily set ligand to this conformation for validation
                    original_coords = self.ligand.coordinates.clone()
                    self.ligand.coordinates = ligand_copy.coordinates.clone()
                    
                    if self._validate_ligand_orientation():
                        rotated_conformations.append(ligand_copy)
                    
                    # Restore original ligand position
                    self.ligand.coordinates = original_coords.clone()
        
        self.logger.info(f"Found {len(rotated_conformations)} valid conformations after rotation scan")
        return rotated_conformations

    def _fine_grained_optimization(self, conformations: List[Structure], step_size: float = 0.1, vdw_threshold: float = 10000.0) -> List[Structure]:
        """
        Perform fine-grained optimization on valid conformations.
        
        Args:
            conformations (List[Structure]): List of valid conformations to optimize
            step_size (float): Fine optimization step size in Å
            vdw_threshold (float): VDW energy threshold for collision detection
            
        Returns:
            List[Structure]: List of optimized conformations
        """
        self.logger.info(f"=== Fine-Grained Optimization (Step size: {step_size} Å) ===")
        
        optimized_conformations = []
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        for conformation in conformations:
            # Save original conformation
            optimized_conformation = self._create_ligand_copy(conformation)
            
            # Create coordinate manager for this conformation
            coord_manager = CoordinateManager(optimized_conformation, device=self.device)
            
            # Get initial score
            initial_score, _ = self.score_conformation(optimized_conformation, include_vdw=True, include_distance=True, distance_weight=1000.0)
            best_score = initial_score
            best_coords = optimized_conformation.coordinates.clone()
            
            # Try small translations in all directions
            for dx in [-step_size, 0.0, step_size]:
                for dy in [-step_size, 0.0, step_size]:
                    for dz in [-step_size, 0.0, step_size]:
                        # Skip zero translation
                        if dx == 0.0 and dy == 0.0 and dz == 0.0:
                            continue
                        
                        # Apply translation
                        translation = torch.tensor([dx, dy, dz], device=self.device)
                        coord_manager.translate_coordinates(translation)
                        
                        # Calculate VDW energy
                        vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, optimized_conformation)
                        
                        # Check if VDW energy is within threshold
                        if vdw_energy < vdw_threshold and not math.isnan(vdw_energy):
                            # Calculate score
                            current_score, _ = self.score_conformation(optimized_conformation, include_vdw=True, include_distance=True, distance_weight=1000.0)
                            
                            # Update best score if this is better
                            if current_score < best_score:
                                best_score = current_score
                                best_coords = optimized_conformation.coordinates.clone()
                        
                        # Restore original coordinates for next iteration
                        optimized_conformation.coordinates = best_coords.clone()
            
            # Update to best conformation
            optimized_conformation.coordinates = best_coords.clone()
            
            # Calculate final VDW energy
            final_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, optimized_conformation)
            
            # Get centers for distance calculations
            receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
            ligand_center = optimized_conformation.calculate_geometric_center().to(self.device)
            ligand_group_center = self.calculate_center(optimized_conformation, self.ligand_group)
            
            # Calculate distances
            ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
            residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
            
            # Check if VDW energy is within threshold and distance conditions are met
            # During optimization: use relaxed VDW threshold, only enforce final VDW < 10000
            # Conditions: 1) VDW < 10000, 2) 2A <= residue_group_distance <= 5A, 3) residue_group_distance <= ligand_to_receptor_group
            if (final_vdw < 10000.0 and 
                2.0 <= residue_group_distance <= 5.0 and 
                residue_group_distance <= ligand_to_receptor_group):
                optimized_conformations.append(optimized_conformation)
        
        self.logger.info(f"Found {len(optimized_conformations)} optimized conformations")
        return optimized_conformations
    
    def search_conformations(self, num_rotations: int = 36) -> List[Structure]:
        """
        Search for docking conformations using molecular dynamics simulation approach.
        
        Args:
            num_rotations (int): Number of rotations (default: 36, 10 degrees each)
            
        Returns:
            List[Structure]: List of ligand conformations
        
        Raises:
            ValueError: If receptor or ligand is not set
            ValueError: If num_rotations is not a positive integer
        """
        if not self.receptor:
            raise ValueError("Receptor structure is not set")
        
        if not self.ligand:
            raise ValueError("Ligand structure is not set")
        
        if not isinstance(num_rotations, int) or num_rotations <= 0:
            raise ValueError("num_rotations must be a positive integer")
        
        self.logger.info("=== Molecular Dynamics Based Conformation Search ===")
        
        # Initial ligand positioning with xyz translation
        self.logger.info("Step 1: Initial Ligand Positioning (XYZ Translation)")
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
        
        # Calculate initial translation vector (ligand group -> receptor group)
        initial_translation = receptor_group_center - ligand_group_center
        
        # Apply initial translation with some offset to avoid direct collision
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        coord_manager.translate_coordinates(initial_translation * 0.5)  # Move halfway initially
        
        # Get current positions
        current_ligand_center = self.ligand.calculate_geometric_center().to(self.device)
        current_distance = torch.norm(receptor_group_center - ligand_group_center).item()
        
        self.logger.info(f"Initial positioning: Ligand residue group {current_distance:.2f} Å from receptor residue group")
        
        # Step 2: Enhanced molecular dynamics simulation with all force components
        self.logger.info("Step 2: Enhanced Molecular Dynamics Simulation with Electrostatic Forces")
        
        # Simulation parameters
        num_steps = 10000
        time_step = 0.01  # Small time step for stability
        friction = 0.9  # Damping factor for velocity
        
        # Save original position for reference
        original_position = self.ligand.coordinates.clone()
        
        # Store conformations
        conformations = []
        best_energy = float('inf')
        best_conformation = None
        
        # Initialize velocity
        velocity = torch.zeros_like(self.ligand.coordinates, device=self.device)
        
        # Enhanced MD simulation loop that uses all force components including electrostatic
        for step in range(num_steps):
            # Save current position before move
            current_position = self.ligand.coordinates.clone()
            
            # Calculate all force components including VDW, electrostatic, and bias
            forces = self.energy_calculator.calculate_total_forces(
                self.receptor, self.ligand,
                self.receptor_group, self.ligand_group,
                bias_strength=5.0  # Adjusted bias strength for balance
            )
            
            # Ensure forces are finite
            forces = torch.nan_to_num(forces, nan=0.0, posinf=1000.0, neginf=-1000.0)
            
            # Calculate average force on all atoms (for rigid body motion)
            avg_force = torch.mean(forces, dim=0)
            
            # Update velocity using force (F = ma, assume mass=1 for simplicity)
            velocity = velocity * friction + avg_force * time_step
            
            # Limit maximum velocity to prevent excessive movement
            max_velocity = 1.0  # Å per step
            velocity = torch.clamp(velocity, -max_velocity, max_velocity)
            
            # Calculate translation step from average velocity
            avg_velocity = torch.mean(velocity, dim=0)
            translation_step = avg_velocity
            
            # Apply translation using CoordinateManager (maintains rigidity)
            coord_manager.translate_coordinates(translation_step)
            
            # Calculate current energy
            current_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            current_electrostatic = self.energy_calculator.calculate_electrostatic_energy(self.receptor, self.ligand)
            total_energy = current_vdw + current_electrostatic
            
            # Recalculate distance after move
            current_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            updated_distance = torch.norm(receptor_group_center - current_ligand_group_center).item()
            
            # Ensure all values are finite
            if not (math.isfinite(current_vdw) and math.isfinite(updated_distance) and math.isfinite(total_energy)):
                self.logger.warning(f"Step {step}: Invalid values detected, reverting move")
                self.ligand.coordinates = current_position.clone()
                velocity = torch.zeros_like(velocity, device=self.device)  # Reset velocity
                continue
            
            # Add collision detection: if VDW energy is too high, try rotation first
            vdw_threshold = 500000.0  # Increased threshold for more exploration
            if current_vdw > vdw_threshold:
                self.logger.warning(f"Step {step}: Collision detected (VDW = {current_vdw:.2f}), attempting rotation to resolve")
                
                # Try multiple rotation attempts to resolve collision
                rotation_success = False
                for rotation_attempt in range(3):
                    # Calculate rotation axis that might resolve collision
                    # Use random axis with strong bias towards receptor to move atoms away from collision
                    rotation_direction = receptor_group_center - current_ligand_group_center
                    rotation_direction = rotation_direction / torch.norm(rotation_direction)
                    
                    # More random axis for better chance of resolving collision
                    random_axis = torch.randn(3, device=self.device) * 0.7 + rotation_direction * 0.3
                    random_axis = random_axis / torch.norm(random_axis)
                    
                    # Larger rotation angle to try to move atoms out of collision
                    random_angle = (torch.rand(1, device=self.device) - 0.5) * 10.0  # ±5 degrees for more dramatic movement
                    
                    # Apply rotation using CoordinateManager (maintains rigidity)
                    coord_manager.rotate_around_axis(
                        random_axis,
                        random_angle.item(),
                        current_ligand_group_center
                    )
                    
                    # Recalculate VDW energy after rotation
                    rotated_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                    rotated_electrostatic = self.energy_calculator.calculate_electrostatic_energy(self.receptor, self.ligand)
                    rotated_energy = rotated_vdw + rotated_electrostatic
                    rotated_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                    rotated_distance = torch.norm(receptor_group_center - rotated_ligand_group_center).item()
                    
                    self.logger.info(f"  Attempt {rotation_attempt+1}: VDW after rotation = {rotated_vdw:.2f} kcal/mol")
                    
                    # Check if rotation resolved the collision
                    if rotated_vdw <= vdw_threshold:
                        # Collision resolved by rotation
                        self.logger.info(f"✓ Collision resolved with rotation! New VDW = {rotated_vdw:.2f} kcal/mol")
                        
                        # Update values after successful rotation
                        current_vdw = rotated_vdw
                        current_electrostatic = rotated_electrostatic
                        total_energy = rotated_energy
                        updated_distance = rotated_distance
                        current_ligand_group_center = rotated_ligand_group_center
                        
                        rotation_success = True
                        break
                    else:
                        # Rotation didn't resolve collision, keep the rotated position and try next rotation
                        # We don't revert yet, as the next rotation might resolve it
                        pass
                
                # After multiple rotation attempts, check if collision was resolved
                if not rotation_success:
                    # Collision still present after rotation attempts, revert the move
                    self.logger.warning(f"✗ Could not resolve collision with rotation, reverting to previous position")
                    self.ligand.coordinates = current_position.clone()
                    velocity = torch.zeros_like(velocity, device=self.device)  # Reset velocity
                    continue
                else:
                    # Collision resolved, continue with the rotated position
                    self.logger.info(f"✓ Collision resolved, continuing with rotated position")
            
            # Save conformation if it meets criteria
            if (current_vdw < 10000.0 and 
                2.0 <= updated_distance <= 5.0 and 
                math.isfinite(total_energy)):
                
                conformation = self._create_ligand_copy(self.ligand)
                conformations.append(conformation)
                
                # Update best conformation
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_conformation = conformation
                    self.logger.info(f"✓ New best conformation found!")
                    self.logger.info(f"  Best Energy: {best_energy:.2f} kcal/mol, Best Distance: {updated_distance:.2f} Å")
            
            # Additional summary every 10 steps
            if step % 100 == 0:
                self.logger.info(f"--- Step {step} Summary ---")
                self.logger.info(f"  Conformations saved: {len(conformations)}")
                self.logger.info(f"  Best energy so far: {best_energy:.2f} kcal/mol")
                self.logger.info(f"  Current distance to target: {updated_distance:.2f} Å (target range: 2.0-5.0 Å)")
                self.logger.info(f"  VDW energy status: {'Low' if current_vdw < 10000.0 else 'High'} (<10000.0 is good)")
            
            # Allow rotation during dynamics simulation
            if step % 5 == 0:  # More frequent rotation
                # Calculate rotation axis that points towards receptor residue group for better orientation
                # This ensures ligand residues tend to point towards receptor residues during rotation
                rotation_direction = receptor_group_center - current_ligand_group_center
                rotation_direction = rotation_direction / torch.norm(rotation_direction)
                
                # Generate random axis that has a component towards the receptor (biased rotation)
                random_axis = torch.randn(3, device=self.device) * 0.5 + rotation_direction * 0.5
                random_axis = random_axis / torch.norm(random_axis)
                
                # Increase rotation angle range slightly (±3 degrees) for more exploration
                random_angle = (torch.rand(1, device=self.device) - 0.5) * 6.0  # ±3 degrees
                
                # Calculate energy before rotation for comparison
                energy_before_rotation = current_vdw + current_electrostatic
                
                # Apply rotation using CoordinateManager (maintains rigidity)
                coord_manager.rotate_around_axis(
                    random_axis,
                    random_angle.item(),
                    current_ligand_group_center
                )
                
                # Recalculate energy after rotation to see effect
                energy_after_rotation = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand) + \
                                     self.energy_calculator.calculate_electrostatic_energy(self.receptor, self.ligand)
                
                # Log rotation details with energy change
                #self.logger.info(f"  Rotation applied: Axis = [{random_axis[0]:.3f}, {random_axis[1]:.3f}, {random_axis[2]:.3f}], Angle = {random_angle.item():.2f} degrees")
                #self.logger.info(f"  Energy change from rotation: {energy_before_rotation:.2f} → {energy_after_rotation:.2f} kcal/mol")
                
                # Recalculate distance and energy after rotation for next iteration
                current_ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
                updated_distance = torch.norm(receptor_group_center - current_ligand_group_center).item()
                current_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                current_electrostatic = self.energy_calculator.calculate_electrostatic_energy(self.receptor, self.ligand)
                total_energy = current_vdw + current_electrostatic
        
        # If no conformations found during MD, try additional sampling
        if not conformations:
            self.logger.warning("No conformations found during MD simulation, trying additional sampling")
            
            # Try systematic translation around receptor group
            for dx in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                for dy in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                    for dz in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                        # Create copy of original ligand
                        ligand_copy = self._create_ligand_copy(self.ligand)
                        
                        # Apply translation
                        translation = torch.tensor([dx, dy, dz], device=self.device)
                        temp_coord_manager = CoordinateManager(ligand_copy, device=self.device)
                        temp_coord_manager.translate_coordinates(translation)
                        
                        # Calculate energy and distance
                        vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, ligand_copy)
                        electrostatic_energy = self.energy_calculator.calculate_electrostatic_energy(self.receptor, ligand_copy)
                        total_energy = vdw_energy + electrostatic_energy
                        
                        # Calculate residue group distance
                        lig_group_center = self.calculate_center(ligand_copy, self.ligand_group)
                        group_distance = torch.norm(receptor_group_center - lig_group_center).item()
                        
                        # Check if conformation meets criteria
                        if (vdw_energy < 10000.0 and 
                            2.0 <= group_distance <= 5.0 and 
                            math.isfinite(total_energy)):
                            conformations.append(ligand_copy)
                            
                            if total_energy < best_energy:
                                best_energy = total_energy
                                best_conformation = ligand_copy
        
        # If still no conformations, add current ligand position
        if not conformations:
            self.logger.warning("No valid conformations found, using current position")
            conformations.append(self._create_ligand_copy(self.ligand))
        
        # Filter conformations to ensure residue group distance <= ligand center to receptor residue group distance
        filtered_conformations = []
        for conformation in conformations:
            # Calculate distances for this conformation
            ligand_group_center = self.calculate_center(conformation, self.ligand_group)
            ligand_center = conformation.calculate_geometric_center().to(self.device)
            
            group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
            ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
            
            if group_distance <= ligand_to_receptor_group:
                filtered_conformations.append(conformation)
        
        if not filtered_conformations:
            filtered_conformations = conformations
        
        self.logger.info(f"Generated {len(filtered_conformations)} valid conformations after MD simulation")
        
        return filtered_conformations
    
    def _random_sampling(self, num_samples: int = 100, vdw_threshold: float = 50000.0) -> List[Structure]:
        """
        Perform random sampling of ligand positions around receptor residue group.
        
        Args:
            num_samples (int): Number of random samples to generate
            vdw_threshold (float): VDW energy threshold for collision detection
            
        Returns:
            List[Structure]: List of valid conformations from random sampling
        """
        self.logger.info(f"=== Random Sampling (Samples: {num_samples}) ===")
        
        conformations = []
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        # Calculate ligand residue group position relative to ligand geometric center
        ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
        ligand_group_offset = self.calculate_center(self.ligand, self.ligand_group) - ligand_geometric_center
        
        # Save original ligand position
        original_position = self.ligand.coordinates.clone()
        
        # Create coordinate manager
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Generate random positions
        for i in range(num_samples):
            # Generate random position around receptor residue group (3-12Å)
            import random
            distance = random.uniform(3.0, 12.0)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)
            
            # Convert spherical coordinates to cartesian
            dx = distance * math.sin(phi) * math.cos(theta)
            dy = distance * math.sin(phi) * math.sin(theta)
            dz = distance * math.cos(phi)
            
            # Calculate target ligand residue group position
            target_ligand_group_position = receptor_group_center + torch.tensor([dx, dy, dz], device=self.device)
            
            # Calculate required ligand geometric center position
            target_ligand_center = target_ligand_group_position - ligand_group_offset
            
            # Calculate translation vector
            translation = target_ligand_center - ligand_geometric_center
            
            # Apply translation
            self.ligand.coordinates = original_position.clone()
            coord_manager.translate_coordinates(translation)
            
            # Calculate VDW energy
            vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            # Calculate centers for distance calculations
            receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
            ligand_center = self.ligand.calculate_geometric_center().to(self.device)
            ligand_group_center = self.calculate_center(self.ligand, self.ligand_group)
            
            # Calculate distances
            ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
            residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
            
            # Check if VDW energy is within threshold and distance conditions are met
            # During random sampling: use relaxed VDW threshold, only enforce distance range and orientation
            # Conditions: 1) VDW < vdw_threshold, 2) 2A <= residue_group_distance <= 5A, 3) residue_group_distance <= ligand_to_receptor_group
            if (vdw_energy < vdw_threshold and 
                2.0 <= residue_group_distance <= 5.0 and 
                residue_group_distance <= ligand_to_receptor_group and 
                not math.isnan(vdw_energy)):
                # Validate ligand orientation
                if not self._validate_ligand_orientation():
                    # Try to correct orientation
                    ligand_geometric_center = self.ligand.calculate_geometric_center().to(self.device)
                    CoordinateManager(self.ligand, device=self.device).rotate_around_axis(
                        torch.tensor([0.0, 1.0, 0.0], device=self.device),
                        180.0,
                        ligand_geometric_center
                    )
                    
                    if not self._validate_ligand_orientation():
                        continue
                
                # Create copy of valid conformation
                valid_conformation = self._create_ligand_copy(self.ligand)
                conformations.append(valid_conformation)
        
        self.logger.info(f"Found {len(conformations)} valid conformations from random sampling")
        
        # Restore original ligand position
        self.ligand.coordinates = original_position.clone()
        
        return conformations
    
    # ---------------------------
    # Scoring Module
    # ---------------------------
    def score_conformation(self, ligand_conformation: Structure, include_vdw: bool = True, include_distance: bool = True, distance_weight: float = 1.0, charge_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """
        Score a docking conformation using the energy calculator.
        
        Args:
            ligand_conformation (Structure): Ligand conformation
            include_vdw (bool, optional): Whether to include van der Waals energy
            include_distance (bool, optional): Whether to include distance penalty/reward
            distance_weight (float, optional): Weight for distance term (only used if include_distance is True)
            charge_weight (float, optional): Weight for electrostatic term
            
        Returns:
            Tuple[float, Dict[str, float]]: 
                - Total conformation score (lower is better)
                - Dictionary containing detailed scores for each component
        
        Raises:
            ValueError: If receptor is not set
            ValueError: If ligand_conformation is None or empty
        """
        if not self.receptor:
            raise ValueError("Receptor structure is not set")
        
        if not ligand_conformation:
            raise ValueError("Ligand conformation is None")
        
        if ligand_conformation.get_atom_count() == 0:
            raise ValueError("Ligand conformation is empty")
        
        if not self.receptor_group:
            raise ValueError("Receptor group atom indices are not set. Please ensure that dock() method has been called with valid residue IDs.")
        
        if not self.ligand_group:
            raise ValueError("Ligand group atom indices are not set. Please ensure that dock() method has been called with valid residue IDs.")
        
        return self.energy_calculator.score_conformation(
            self.receptor, ligand_conformation, 
            self.receptor_group, self.ligand_group,
            include_vdw=include_vdw,
            include_distance=include_distance,
            distance_weight=distance_weight,
            charge_weight=charge_weight
        )
    
    def _local_optimize_conformation(self, ligand_conformation: Structure) -> Structure:
        """
        Apply local optimization to a ligand conformation using gradient descent.
        
        Args:
            ligand_conformation (Structure): Initial ligand conformation
            
        Returns:
            Structure: Optimized ligand conformation
        """
        # Create a copy of the ligand conformation to optimize
        optimized_ligand = self._create_ligand_copy(ligand_conformation)
        
        # Local optimization parameters
        max_iterations = 20
        learning_rate = 0.1
        vdw_threshold = 1000.0
        improvement_threshold = 0.1
        
        # Create coordinate manager for optimization
        coord_manager = CoordinateManager(optimized_ligand, device=self.device)
        
        # Get receptor and ligand group centers
        receptor_group_center = self.calculate_center(self.receptor, self.receptor_group)
        
        # Initial energy
        current_score, _ = self.score_conformation(optimized_ligand, include_vdw=True, include_distance=True)
        
        for iteration in range(max_iterations):
            # Calculate current ligand group center
            current_ligand_group_center = self.calculate_center(optimized_ligand, self.ligand_group)
            
            # Calculate direction vector from ligand to receptor group center
            direction = receptor_group_center - current_ligand_group_center
            direction_norm = torch.norm(direction)
            
            if direction_norm < 0.1:  # Already very close, no need to optimize further
                break
            
            # Normalize direction vector
            direction_normalized = direction / direction_norm
            
            # Check if ligand protein vector is facing receptor residue group
            ligand_geometric_center = optimized_ligand.calculate_geometric_center().to(self.device)
            
            # Calculate ligand protein vector (from ligand center to ligand residue group)
            ligand_protein_vector = current_ligand_group_center - ligand_geometric_center
            
            # Calculate vector from ligand to receptor residue group (direction ligand should face)
            lig_to_rec_vector = receptor_group_center - current_ligand_group_center
            
            # Normalize vectors for dot product calculation
            if torch.norm(ligand_protein_vector) > 1e-6 and torch.norm(lig_to_rec_vector) > 1e-6:
                normalized_lig_protein_vector = ligand_protein_vector / torch.norm(ligand_protein_vector)
                normalized_lig_to_rec_vector = lig_to_rec_vector / torch.norm(lig_to_rec_vector)
                
                # Calculate dot product to check orientation
                dot_product = torch.dot(normalized_lig_protein_vector, normalized_lig_to_rec_vector)
                
                # If the dot product is positive, the ligand protein vector is facing away from receptor
                if dot_product > 0:
                    # Rotate 180 degrees around y-axis to make ligand protein vector face receptor
                    flip_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                    coord_manager.rotate_around_axis(flip_axis, 180.0, ligand_geometric_center)
                    
                    # Recalculate after rotation
                    current_ligand_group_center = self.calculate_center(optimized_ligand, self.ligand_group)
                    direction = receptor_group_center - current_ligand_group_center
                    direction_norm = torch.norm(direction)
                    if direction_norm < 0.1:
                        break
                    direction_normalized = direction / direction_norm
            
            # Calculate step vector (gradient descent)
            step = direction_normalized * learning_rate
            
            # Apply translation
            coord_manager.translate_coordinates(step)
            
            # Calculate new score
            new_score, _ = self.score_conformation(optimized_ligand, include_vdw=True, include_distance=True)
            
            # Check if vdw energy is too high
            vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, optimized_ligand)
            if vdw_energy > vdw_threshold:
                # If energy is too high, revert the step
                coord_manager.translate_coordinates(-step)
                break
            
            # Check if improvement is significant
            if new_score >= current_score - improvement_threshold:
                # No significant improvement, stop optimization
                break
            
            # Update current score
            current_score = new_score
        
        return optimized_ligand
    
    # ---------------------------
    # Main Docking Module
    # ---------------------------
    def dock(self, receptor_residues: List[str], ligand_residues: List[str], num_rotations: int = 36) -> Tuple[Structure, float]:
        """
        Perform docking with specified parameters.
        
        Args:
            receptor_residues (List[str]): Receptor residue IDs
            ligand_residues (List[str]): Ligand residue IDs
            num_rotations (int): Number of rotations for conformation search
            
        Returns:
            Tuple[Structure, float]: Best ligand conformation and its score
        
        Raises:
            ValueError: If receptor or ligand is not set
            ValueError: If receptor_residues or ligand_residues is empty
        """
        # Validate input parameters
        if not self.receptor:
            raise ValueError("Receptor structure is not set")
        
        if not self.ligand:
            raise ValueError("Ligand structure is not set")
        
        if not receptor_residues:
            raise ValueError("Receptor residues list is empty")
        
        if not ligand_residues:
            raise ValueError("Ligand residues list is empty")
        
        # Step 1: Convert residue IDs to atom indices
        self.logger.info("Converting residue IDs to atom indices...")
        
        # Convert residues to atom indices using shared function
        try:
            self.receptor_group = residues_to_atom_indices(self.receptor, receptor_residues, self.logger)
            self.ligand_group = residues_to_atom_indices(self.ligand, ligand_residues, self.logger)
            
            self.logger.info(f"Found {len(self.receptor_group)} atoms in receptor residue group")
            self.logger.info(f"Found {len(self.ligand_group)} atoms in ligand residue group")
        except Exception as e:
            self.logger.error(f"Error converting residues to atom indices: {e}")
            raise
        
        # Step 2: Align proteins
        self.logger.section("Aligning Proteins")
        self.align_proteins()
        
        # Step 2.5: Optimize position
        self.logger.section("Optimizing Position")
        self.optimize_position()
        
        # Step 3: Generate conformations through rotation
        self.logger.section("Generating Conformations")
        conformations = self.search_conformations(num_rotations)
        
        if not conformations:
            self.logger.error("No conformations generated")
            raise ValueError("Failed to generate conformations")
        
        # Step 3: Score all conformations
        self.logger.section("Scoring Conformations")
        scored_conformations = []
        
        for i, conf in enumerate(conformations):
            # Prioritize residue group proximity over energy by including distance with high weight
            score, detailed_scores = self.score_conformation(
                conf, 
                include_vdw=True, 
                include_distance=True, 
                distance_weight=1000.0,  # High weight to prioritize proximity
                charge_weight=1.0
            )
            scored_conformations.append((conf, score))
            
            # Calculate residue group distance for logging
            rec_group_center = self.calculate_center(self.receptor, self.receptor_group)
            lig_group_center = self.calculate_center(conf, self.ligand_group)
            group_distance = torch.norm(rec_group_center - lig_group_center).item()
            
            if i % 10 == 0 or i == len(conformations) - 1:
                self.logger.info(f"  Conformation {i+1}/{len(conformations)}: Score = {score:.2f}, Group Distance = {group_distance:.4f} Å")
                self.logger.debug(f"    Detailed scores: Electrostatic={detailed_scores['electrostatic']:.2f}, VDW={detailed_scores['vdw']:.2f}, Distance={detailed_scores['distance']:.2f}")
        
        # Step 4: Find best conformation
        scored_conformations.sort(key=lambda x: x[1])
        best_conformation, best_score = scored_conformations[0]
        
        self.logger.section("Docking Complete")
        self.logger.info(f"Best conformation score: {best_score:.2f}")
        
        return best_conformation, best_score
    
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
            # Ensure coordinates are finite
            x = float(coord[0]) if math.isfinite(coord[0]) else 0.0
            y = float(coord[1]) if math.isfinite(coord[1]) else 0.0
            z = float(coord[2]) if math.isfinite(coord[2]) else 0.0
            merged.add_atom(atom, (x, y, z))
        
        # Add TER record after receptor atoms
        merged.add_other_record("TER")
        
        # Add ligand atoms
        for i, atom in enumerate(ligand.atoms):
            coord = ligand.coordinates[i]
            # Ensure coordinates are finite
            x = float(coord[0]) if math.isfinite(coord[0]) else 0.0
            y = float(coord[1]) if math.isfinite(coord[1]) else 0.0
            z = float(coord[2]) if math.isfinite(coord[2]) else 0.0
            merged.add_atom(atom, (x, y, z))
        
        # Add final TER and END records
        merged.add_other_record("TER")
        merged.add_other_record("END")
        
        return merged