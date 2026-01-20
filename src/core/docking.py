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
        from src.core.alignment import calculate_rotation_parameters
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
        2. Initially overlap protein centers
        3. Gradually increase distance along z-axis until appropriate separation
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
        # Step 4: Gradually increase distance along z-axis
        # --------------------------
        step_size = 0.2  
        current_distance = torch.norm(self.receptor_center - self.ligand_center).item()
        self.logger.info(f"Current distance: {current_distance:.4f} Å")
        
        # Create coordinate manager for ligand
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Gradually increase distance along positive z-axis
        max_iterations = 10000  # Safety counter to prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            # Calculate translation vector along positive z-axis
            translation = torch.tensor([0.0, 0.0, step_size], device=self.device, dtype=torch.float16)
            coord_manager.translate_coordinates(translation)
            
            # Update current distance
            current_distance = torch.norm(self.receptor_center - self.ligand_center).item()
            
            # Calculate vdw energy
            vdw_energy = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            # Check if vdw energy is below threshold or distance is sufficient
            if vdw_energy == float('nan'):
                continue
            if vdw_energy < 10000:
                step_size = max(0.01,step_size*0.9)
            if vdw_energy < 0:
                self.logger.info(f"Stopping distance increase: vdw_energy = {vdw_energy:.2f}, current_distance = {current_distance:.4f} Å")
                break
            if iteration % 50 == 0:
                self.logger.info(f"Iteration {iteration}: distance = {current_distance:.4f} Å, vdw_energy = {vdw_energy:.2f}")
            iteration += 1
        
        if iteration >= max_iterations:
            self.logger.warning(f"Reached maximum iterations ({max_iterations}) without meeting stopping criteria")
        
        # --------------------------
        # Step 5: Final verification
        # --------------------------
        actual_distance = torch.norm(self.receptor_center - self.ligand_center).item()
        self.logger.info("=== Final Position ===")
        self.logger.info(f"Receptor center: ({self.receptor_center[0]:.4f}, {self.receptor_center[1]:.4f}, {self.receptor_center[2]:.4f})")
        self.logger.info(f"Ligand center: ({self.ligand_center[0]:.4f}, {self.ligand_center[1]:.4f}, {self.ligand_center[2]:.4f})")
        self.logger.info(f"Distance between centers: {actual_distance:.4f} Å")
        
        # Verify protein vector directions
        rec_vector = self.rec_group_center  # Since receptor is at origin
        lig_vector = self.lig_group_center  # Since ligand is at origin
        
        self.logger.info(f"Receptor vector direction: ({rec_vector[0]:.4f}, {rec_vector[1]:.4f}, {rec_vector[2]:.4f})")
        self.logger.info(f"Ligand vector direction: ({lig_vector[0]:.4f}, {lig_vector[1]:.4f}, {lig_vector[2]:.4f})")
    
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
            import math
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
            if score < best_score and vdw_energy < 1000.0:
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
        Simulates ligand being pulled towards receptor target group while searching along the surface,
        ensuring van der Waals energy stays below 1000 and ligand remains rigid.
        """
        if not self.receptor or not self.ligand:
            raise ValueError("Receptor or ligand structure is None")
        
        if not self.receptor_group or not self.ligand_group:
            raise ValueError("Receptor or ligand group atom indices are empty")
        
        self.logger.info("===== Optimizing Ligand Position =====")
        
        # Get receptor and ligand group centers
        receptor_center = self.calculate_center(self.receptor, self.receptor_group)
        ligand_center = self.calculate_center(self.ligand, self.ligand_group)
        
        self.logger.info(f"Initial receptor group center: ({receptor_center[0]:.4f}, {receptor_center[1]:.4f}, {receptor_center[2]:.4f})")
        self.logger.info(f"Initial ligand group center: ({ligand_center[0]:.4f}, {ligand_center[1]:.4f}, {ligand_center[2]:.4f})")
        
        # Create coordinate manager for ligand
        coord_manager = CoordinateManager(self.ligand, device=self.device)
        
        # Initial distance and energy
        initial_distance = torch.norm(receptor_center - ligand_center).item()
        initial_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        
        self.logger.info(f"Initial distance: {initial_distance:.4f} Å")
        self.logger.info(f"Initial VDW energy: {initial_vdw:.2f}")
        
        # Optimization parameters
        max_iterations = 10000
        step_size = 0.1
        min_step_size = 0.01
        vdw_threshold = 10000.0
        distance_threshold = 0.5
        
        best_distance = initial_distance
        best_position = self.ligand.coordinates.clone()
        best_vdw = initial_vdw
        
        # Counter for steps without improvement
        no_improvement_steps = 0
        
        for iteration in range(max_iterations):
            # Calculate direction vector from ligand to receptor
            current_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
            direction = receptor_center - current_ligand_center
            direction_norm = torch.norm(direction)
            
            if direction_norm < 1e-6:
                self.logger.info("Ligand is already at receptor target group center")
                break
            
            # Normalize direction vector
            direction_normalized = direction / direction_norm
            
            # Check if we need to apply reverse force
            if no_improvement_steps >= 100:
                self.logger.info(f"Applying reverse force to ligand (no improvement for {no_improvement_steps} steps)")
                # Apply reverse force - move ligand away from receptor
                reverse_step = -direction_normalized * (step_size * 5)
                coord_manager.translate_coordinates(reverse_step)
                # Reset counter
                no_improvement_steps = 0
                
                # Log reverse force application
                new_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
                new_distance = torch.norm(receptor_center - new_ligand_center).item()
                new_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                self.logger.info(f"After reverse force: Distance = {new_distance:.4f} Å, VDW = {new_vdw:.2f}")
                continue
            
            # Calculate step vector with adaptive step size
            step_vector = direction_normalized * step_size
            
            # Save current position
            current_position = self.ligand.coordinates.clone()
            
            # Move ligand in the direction of receptor
            coord_manager.translate_coordinates(step_vector)
            
            # Calculate new distance and energy
            new_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
            new_distance = torch.norm(receptor_center - new_ligand_center).item()
            new_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
            
            # Check if new position is better
            if new_vdw <= vdw_threshold:
                # Good position, update best
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_position = self.ligand.coordinates.clone()
                    best_vdw = new_vdw
                    # Reset no improvement counter
                    no_improvement_steps = 0
                else:
                    # No improvement
                    no_improvement_steps += 1
                
                # Adjust step size based on progress
                if new_distance < best_distance * 0.9:
                    step_size = min(step_size * 1.1, 0.5)
                else:
                    step_size = max(step_size * 0.9, min_step_size)
            else:
                # Bad position, revert
                self.ligand.coordinates = current_position
                # Reduce step size and try different direction
                step_size = max(step_size * 0.5, min_step_size)
                
                # Try slight lateral movement to avoid collision
                lateral_direction = torch.tensor([direction_normalized[1], -direction_normalized[0], 0.0], 
                                               device=self.device, dtype=direction_normalized.dtype)
                lateral_step = lateral_direction * step_size
                coord_manager.translate_coordinates(lateral_step)
                
                # Check lateral position
                lateral_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                if lateral_vdw > vdw_threshold:
                    # Revert if lateral also bad
                    self.ligand.coordinates = current_position
                
                # Increment no improvement counter
                no_improvement_steps += 1
            
            # Log progress every 100 iterations
            if iteration % 100 == 0:
                current_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
                current_distance = torch.norm(receptor_center - current_ligand_center).item()
                current_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
                self.logger.info(f"Iteration {iteration}: Distance = {current_distance:.4f} Å, VDW = {current_vdw:.2f}, Step size = {step_size:.4f}, No improvement steps = {no_improvement_steps}")
            
            # Check if converged
            if best_distance < distance_threshold:
                self.logger.info(f"Converged: Distance below threshold ({best_distance:.4f} Å)")
                break
        
        # Set best position
        self.ligand.coordinates = best_position
        
        # Final verification
        final_ligand_center = self.calculate_center(self.ligand, self.ligand_group)
        final_distance = torch.norm(receptor_center - final_ligand_center).item()
        final_vdw = self.energy_calculator.calculate_vdw_energy(self.receptor, self.ligand)
        
        self.logger.info("=== Final Position ===")
        self.logger.info(f"Final receptor group center: ({receptor_center[0]:.4f}, {receptor_center[1]:.4f}, {receptor_center[2]:.4f})")
        self.logger.info(f"Final ligand group center: ({final_ligand_center[0]:.4f}, {final_ligand_center[1]:.4f}, {final_ligand_center[2]:.4f})")
        self.logger.info(f"Final distance: {final_distance:.4f} Å")
        self.logger.info(f"Final VDW energy: {final_vdw:.2f}")
        
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
    
    def search_conformations(self, num_rotations: int = 360) -> List[Structure]:
        """
        Search for docking conformations by rotating ligand around z-axis.
        
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
        
        conformations = []
        rotation_step = 360.0 / num_rotations
        
        # Use z-axis as rotation axis
        rotation_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        
        # Rotation center is ligand geometric center
        rotation_center = self.ligand_center
        
        # Create copies of ligand for each rotation
        for i in range(num_rotations):
            # Create copy of base ligand with detached coordinates
            ligand_copy = self._create_ligand_copy(self.ligand)
            
            # Rotate ligand around z-axis
            angle = i * rotation_step
            CoordinateManager(ligand_copy, device=self.device).rotate_around_axis(rotation_axis, angle, rotation_center)
            
            conformations.append(ligand_copy)
        
        return conformations
    
    # ---------------------------
    # Scoring Module
    # ---------------------------
    def score_conformation(self, ligand_conformation: Structure, include_vdw: bool = True, include_distance: bool = False, distance_weight: float = 1.0, charge_weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
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
        
        # Step 2.3: Generate initial conformations
        self.logger.section("Generating Initial Conformations")
        self.generate_initial_conformations()
        
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
            score, detailed_scores = self.score_conformation(
                conf, 
                include_vdw=True, 
                include_distance=False, 
                charge_weight=1.0
            )
            scored_conformations.append((conf, score))
            
            if i % 10 == 0 or i == len(conformations) - 1:
                self.logger.info(f"  Conformation {i+1}/{len(conformations)}: Score = {score:.2f}")
                self.logger.debug(f"    Detailed scores: Electrostatic={detailed_scores['electrostatic']:.2f}, VDW={detailed_scores['vdw']:.2f}")
        
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
