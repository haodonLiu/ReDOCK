#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedicated Scan Engine for Protein-Protein Docking

This module provides a dedicated scan engine for protein-protein docking,
which generates conformations by scanning ligand positions and orientations
around receptor residue groups.
"""

from typing import List, Tuple, Dict
import torch
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
# Use relative imports to avoid circular import issues
from ..models.structure import Structure
from ..core.coordinate_manager import CoordinateManager
from ..core.energy_calculator import EnergyCalculator
from ..utils.logger import Logger


class ScanEngine:
    """
    Dedicated scan engine for protein-protein docking.
    
    This engine generates conformations by simultaneously scanning:
    1. Translation in xyz directions
    2. Rotation around xyz axes
    3. Validating and filtering conformations based on specified criteria
    """
    
    def __init__(self, logger: Logger, energy_calculator: EnergyCalculator, device: torch.device):
        """
        Initialize the scan engine.
        
        Args:
            logger (Logger): Logger instance for logging
            energy_calculator (EnergyCalculator): Energy calculator for VDW energy calculation
            device (torch.device): Device to use for calculations
        """
        self.logger = logger
        self.energy_calculator = energy_calculator
        self.device = device
    
    def calculate_center(self, structure: Structure, group_indices: List[int]) -> torch.Tensor:
        """
        Calculate the center of a residue group.
        
        Args:
            structure (Structure): Protein structure
            group_indices (List[int]): Indices of atoms in the residue group
            
        Returns:
            torch.Tensor: Center coordinates of the residue group
        """
        if not group_indices:
            raise ValueError("Group indices list is empty")
        
        # Get coordinates of the residue group atoms
        group_coords = structure.coordinates[group_indices]
        
        # Calculate center
        return torch.mean(group_coords, dim=0)
    
    def calculate_conformation_features(self, conformation: Structure, original_ligand: Structure) -> np.ndarray:
        """
        Calculate six degrees of freedom features for a conformation:
        3 for translation (x, y, z) and 3 for rotation (Euler angles α, β, γ)
        
        Args:
            conformation (Structure): Conformation to calculate features for
            original_ligand (Structure): Original ligand structure for reference
            
        Returns:
            np.ndarray: Six-dimensional feature vector [x, y, z, α, β, γ]
        """
        # 1. Calculate translation features: ligand geometric center
        current_center = conformation.calculate_geometric_center().cpu().numpy()
        original_center = original_ligand.calculate_geometric_center().cpu().numpy()
        translation = current_center - original_center
        
        # 2. Calculate rotation features: Euler angles
        # To calculate rotation, we need to align the current structure with the original
        # We'll use the Kabsch algorithm to find the rotation matrix
        # For simplicity, we'll use all atoms for alignment
        current_coords = conformation.coordinates.cpu().numpy()
        original_coords = original_ligand.coordinates.cpu().numpy()
        
        # Center both structures
        current_coords_centered = current_coords - current_center
        original_coords_centered = original_coords - original_center
        
        # Calculate covariance matrix
        covariance = np.dot(original_coords_centered.T, current_coords_centered)
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(covariance)
        
        # Calculate rotation matrix
        rotation_matrix = np.dot(Vt.T, U.T)
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(Vt.T, U.T)
        
        # Convert rotation matrix to Euler angles (ZYX convention)
        # Based on https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0.0
        
        # Convert radians to degrees for better clustering behavior
        euler_angles = np.array([x, y, z]) * 180.0 / np.pi
        
        # Combine translation and rotation features
        features = np.concatenate([translation, euler_angles])
        
        return features
    
    def cluster_conformations(self, 
                            conformations: List[Structure], 
                            original_ligand: Structure,
                            n_clusters: int = 10, 
                            random_state: int = 42) -> Dict[int, List[Structure]]:
        """
        Cluster conformations based on six degrees of freedom (xyz translation + xyz rotation).
        
        Args:
            conformations (List[Structure]): List of conformations to cluster
            original_ligand (Structure): Original ligand structure for reference
            n_clusters (int): Number of clusters to form
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[int, List[Structure]]: Dictionary mapping cluster IDs to list of conformations in each cluster
        """
        if not conformations:
            return {}
        
        self.logger.info(f"Clustering {len(conformations)} conformations into {n_clusters} clusters based on 6 degrees of freedom...")
        
        # Extract six degrees of freedom features for each conformation
        features = []
        for conf in conformations:
            conf_features = self.calculate_conformation_features(conf, original_ligand)
            features.append(conf_features)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Group conformations by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for conf, label in zip(conformations, cluster_labels):
            clusters[label].append(conf)
        
        # Remove empty clusters
        clusters = {k: v for k, v in clusters.items() if v}
        
        # Log cluster sizes
        self.logger.info(f"Clustering completed. Found {len(clusters)} non-empty clusters:")
        for cluster_id, cluster_confs in clusters.items():
            self.logger.info(f"  Cluster {cluster_id}: {len(cluster_confs)} conformations")
        
        return clusters
    
    def save_clusters(self, clusters: Dict[int, List[Structure]], receptor: Structure, output_prefix: str = "cluster"):
        """
        Save clustered conformations to PDB files.
        
        Args:
            clusters (Dict[int, List[Structure]]): Dictionary of clusters
            receptor (Structure): Receptor structure
            output_prefix (str): Prefix for output PDB filenames
        """
        for cluster_id, conformations in clusters.items():
            # Save best conformation from each cluster (first one)
            best_conf = conformations[0]
            
            # Merge receptor and ligand conformations for output
            merged_structure = receptor.copy()
            merged_structure.add_structure(best_conf)
            
            # Save to PDB file
            output_file = f"{output_prefix}_{cluster_id}.pdb"
            with open(output_file, "w") as f:
                for line in merged_structure.to_pdb_lines():
                    f.write(line + "\n")
            
            self.logger.info(f"Saved best conformation from cluster {cluster_id} to {output_file}")
            
            # Save all conformations from this cluster to a single file (optional)
            all_output_file = f"{output_prefix}_{cluster_id}_all.pdb"
            with open(all_output_file, "w") as f:
                for i, conf in enumerate(conformations):
                    merged_structure = receptor.copy()
                    merged_structure.add_structure(conf)
                    for line in merged_structure.to_pdb_lines():
                        f.write(line + "\n")
                    # Add ENDMDL record between models
                    if i < len(conformations) - 1:
                        f.write("ENDMDL\n")
            
            self.logger.info(f"Saved all {len(conformations)} conformations from cluster {cluster_id} to {all_output_file}")
    
    def _create_ligand_copy(self, ligand: Structure) -> Structure:
        """
        Create a copy of the ligand structure.
        
        Args:
            ligand (Structure): Original ligand structure
            
        Returns:
            Structure: Copy of the ligand structure
        """
        ligand_copy = Structure()
        ligand_copy.atoms = ligand.atoms.copy()
        ligand_copy.coordinates = ligand.coordinates.detach().clone()
        ligand_copy.other_records = ligand.other_records.copy()
        ligand_copy.chains = ligand.chains.copy()
        ligand_copy.residues = ligand.residues.copy()
        ligand_copy.total_charge = ligand.total_charge
        return ligand_copy
    
    def dedicated_scan(self, 
                      receptor: Structure, 
                      ligand: Structure, 
                      receptor_group: List[int], 
                      ligand_group: List[int],
                      initial_distance: float = 5.0,
                      coarse_translation_step: float = 1.0,
                      fine_translation_step: float = 0.4,
                      rotation_step: float = 60.0,
                      max_translation_range: float = 5.0,
                      coarse_vdw_threshold: float = 50000.0,
                      fine_vdw_threshold: float = 10000.0,
                      max_residue_distance: float = 10.0) -> List[Structure]:
        """
        Perform dedicated scan to generate valid conformations using two-step strategy:
        1. Coarse scan with lower VDW requirements
        2. Fine scan on promising regions with higher VDW requirements
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            receptor_group (List[int]): Indices of receptor residue group atoms
            ligand_group (List[int]): Indices of ligand residue group atoms
            initial_distance (float): Initial distance between residue groups
            coarse_translation_step (float): Step size for coarse translation scan
            fine_translation_step (float): Step size for fine translation scan
            rotation_step (float): Step size for rotation scan
            max_translation_range (float): Maximum translation range in each direction
            coarse_vdw_threshold (float): Maximum VDW energy allowed for coarse scan
            fine_vdw_threshold (float): Maximum VDW energy allowed for fine scan
            max_residue_distance (float): Maximum residue group distance allowed
            
        Returns:
            List[Structure]: List of valid conformations
        """
        self.logger.info("=== Dedicated Scan Engine ===")
        self.logger.info(f"Initial parameters:")
        self.logger.info(f"  Initial distance: {initial_distance:.2f} Å")
        self.logger.info(f"  Coarse translation step: {coarse_translation_step:.2f} Å")
        self.logger.info(f"  Fine translation step: {fine_translation_step:.2f} Å")
        self.logger.info(f"  Rotation step: {rotation_step:.2f} degrees")
        self.logger.info(f"  Max translation range: {max_translation_range:.2f} Å")
        self.logger.info(f"  Coarse VDW threshold: {coarse_vdw_threshold:.2f} kcal/mol")
        self.logger.info(f"  Fine VDW threshold: {fine_vdw_threshold:.2f} kcal/mol")
        self.logger.info(f"  Max residue distance: {max_residue_distance:.2f} Å")
        
        # Calculate receptor residue group center
        receptor_group_center = self.calculate_center(receptor, receptor_group)
        self.logger.info(f"Receptor residue group center: ({receptor_group_center[0]:.4f}, {receptor_group_center[1]:.4f}, {receptor_group_center[2]:.4f})")
        
        # Calculate ligand residue group offset from ligand geometric center
        ligand_geometric_center = ligand.calculate_geometric_center().to(self.device)
        ligand_group_center = self.calculate_center(ligand, ligand_group).to(self.device)
        ligand_group_offset = ligand_group_center - ligand_geometric_center
        self.logger.info(f"Ligand group offset: ({ligand_group_offset[0]:.4f}, {ligand_group_offset[1]:.4f}, {ligand_group_offset[2]:.4f})")
        
        # Save original ligand position
        original_position = ligand.coordinates.clone()
        
        # Ensure all coordinates are on the correct device before calculations
        receptor.coordinates = receptor.coordinates.to(self.device)
        ligand.coordinates = ligand.coordinates.to(self.device)
        receptor_group_center = receptor_group_center.to(self.device)
        ligand_group_offset = ligand_group_offset.to(self.device)
        
        all_conformations = []
        
        # Step 1: Coarse scan with lower VDW requirements
        self.logger.info("=== Step 1: Coarse Scan ===")
        self.logger.info(f"Using VDW threshold: {coarse_vdw_threshold:.2f} kcal/mol")
        self.logger.info(f"Using translation step: {coarse_translation_step:.2f} Å")
        
        # Perform coarse scan (translation + rotation simultaneously)
        coarse_conformations = self._simultaneous_scan(
            receptor=receptor,
            ligand=ligand,
            receptor_group_center=receptor_group_center,
            ligand_group_offset=ligand_group_offset,
            ligand_group=ligand_group,
            receptor_group=receptor_group,
            initial_distance=initial_distance,
            translation_step=coarse_translation_step,
            max_translation_range=max_translation_range,
            rotation_step=rotation_step,
            vdw_threshold=coarse_vdw_threshold,
            max_residue_distance=max_residue_distance,
            original_position=original_position.to(self.device),
            scan_type="coarse"
        )
        
        self.logger.info(f"Generated {len(coarse_conformations)} valid conformations from coarse scan")
        
        # Step 1.5: Cluster coarse conformations
        if coarse_conformations:
            # Create a copy of the original ligand for reference
            original_ligand_copy = self._create_ligand_copy(ligand)
            
            # Cluster coarse conformations based on 6 degrees of freedom (xyz translation + xyz rotation)
            coarse_clusters = self.cluster_conformations(
                conformations=coarse_conformations,
                original_ligand=original_ligand_copy,
                n_clusters=10,  # Default to 10 clusters
                random_state=42
            )
            
            # Save clustered conformations
            self.save_clusters(
                clusters=coarse_clusters,
                receptor=receptor,
                output_prefix="coarse_cluster"
            )
            
            # Update coarse_conformations to use only cluster representatives (best from each cluster)
            coarse_conformations = [cluster_confs[0] for cluster_confs in coarse_clusters.values()]
            self.logger.info(f"Using {len(coarse_conformations)} cluster representatives for fine scan")
        
        if not coarse_conformations:
            self.logger.warning("No valid conformations from coarse scan, trying fine scan directly")
            # If no coarse conformations found, try fine scan directly
            fine_conformations = self._simultaneous_scan(
                receptor=receptor,
                ligand=ligand,
                receptor_group_center=receptor_group_center,
                ligand_group_offset=ligand_group_offset,
                ligand_group=ligand_group,
                receptor_group=receptor_group,
                initial_distance=initial_distance,
                translation_step=fine_translation_step,
                max_translation_range=max_translation_range,
                rotation_step=rotation_step,
                vdw_threshold=fine_vdw_threshold,
                max_residue_distance=max_residue_distance,
                original_position=original_position.to(self.device),
                scan_type="fine"
            )
            
            if not fine_conformations:
                self.logger.warning("No valid conformations from fine scan, returning empty list")
                ligand.coordinates = original_position.clone()
                return []
            
            # Restore original ligand position
            ligand.coordinates = original_position.clone()
            return fine_conformations
        
        # Step 2: Fine scan on promising regions from coarse scan
        self.logger.info("=== Step 2: Fine Scan on Promising Regions ===")
        self.logger.info(f"Using VDW threshold: {fine_vdw_threshold:.2f} kcal/mol")
        self.logger.info(f"Using translation step: {fine_translation_step:.2f} Å")
        
        # For each coarse conformation, perform fine scan around it
        for i, coarse_conf in enumerate(coarse_conformations):
            self.logger.info(f"Processing coarse conformation {i+1}/{len(coarse_conformations)}")
            
            # Calculate the position of this coarse conformation
            coarse_ligand_center = coarse_conf.calculate_geometric_center().to(self.device)
            coarse_ligand_group_center = self.calculate_center(coarse_conf, ligand_group).to(self.device)
            
            # Calculate the offset from the initial position
            coarse_offset = coarse_ligand_group_center - (receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=self.device))
            
            # Perform fine scan around this position with smaller step size
            fine_conformations = self._fine_scan_around_position(
                receptor=receptor,
                ligand=ligand,
                receptor_group_center=receptor_group_center,
                ligand_group_offset=ligand_group_offset,
                ligand_group=ligand_group,
                receptor_group=receptor_group,
                center_offset=coarse_offset,
                initial_distance=initial_distance,
                fine_translation_step=fine_translation_step,
                fine_vdw_threshold=fine_vdw_threshold,
                max_residue_distance=max_residue_distance,
                original_position=original_position.to(self.device),
                rotation_step=rotation_step
            )
            
            self.logger.info(f"Generated {len(fine_conformations)} valid conformations from fine scan around coarse conformation {i+1}")
            
            # Add fine conformations to the list
            all_conformations.extend(fine_conformations)
        
        # If no fine conformations found, use coarse conformations
        if not all_conformations:
            self.logger.warning("No valid conformations from fine scan, using coarse conformations")
            all_conformations = coarse_conformations
        
        # Restore original ligand position
        ligand.coordinates = original_position.clone()
        
        self.logger.info(f"Total valid conformations generated: {len(all_conformations)}")
        return all_conformations
    
    def _simultaneous_scan(self, 
                          receptor: Structure,
                          ligand: Structure,
                          receptor_group_center: torch.Tensor,
                          ligand_group_offset: torch.Tensor,
                          ligand_group: List[int],
                          receptor_group: List[int],
                          initial_distance: float,
                          translation_step: float,
                          max_translation_range: float,
                          rotation_step: float,
                          vdw_threshold: float,
                          max_residue_distance: float,
                          original_position: torch.Tensor,
                          scan_type: str = "coarse") -> List[Structure]:
        """
        Perform simultaneous translation and rotation scan to generate conformations.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            ligand_group (List[int]): Indices of ligand residue group atoms
            receptor_group (List[int]): Indices of receptor residue group atoms
            initial_distance (float): Initial distance between residue groups
            translation_step (float): Step size for translation scan
            max_translation_range (float): Maximum translation range in each direction
            rotation_step (float): Step size for rotation scan
            vdw_threshold (float): Maximum VDW energy allowed
            max_residue_distance (float): Maximum residue group distance allowed
            original_position (torch.Tensor): Original ligand position
            scan_type (str): Type of scan, either "coarse" or "fine"
            
        Returns:
            List[Structure]: List of valid conformations
        """
        conformations = []
        
        # Generate translation grid
        num_steps = int(max_translation_range / translation_step)
        translation_values = torch.linspace(-max_translation_range, max_translation_range, num_steps * 2 + 1, device=self.device)
        total_translation_points = len(translation_values) ** 3
        
        # Generate rotation parameters based on scan type
        if scan_type == "coarse":
            # Coarse scan: 60 degrees step, 0-360 range
            rotation_angles = torch.linspace(0.0, 360.0, int(360.0 / rotation_step), device=self.device)
        else:  # fine scan
            # Fine scan: 10 degrees step, -30 to 30 range
            rotation_angles = torch.linspace(-30.0, 30.0, int(60.0 / 10.0) + 1, device=self.device)
        
        rotation_axes = [
            torch.tensor([1.0, 0.0, 0.0], device=self.device),  # x-axis
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # y-axis  
            torch.tensor([0.0, 0.0, 1.0], device=self.device)   # z-axis
        ]
        
        total_rotation_combinations = len(rotation_axes) * len(rotation_angles)
        total_points = total_translation_points * total_rotation_combinations
        
        self.logger.info(f"{scan_type.capitalize()} scan: {len(translation_values)} translation points in each dimension, {len(rotation_axes)} axes, {len(rotation_angles)} angles")
        self.logger.info(f"Total scan points: {total_points}")
        
        # Calculate initial ligand geometric center
        initial_ligand_group_center = receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=self.device)
        initial_ligand_center = initial_ligand_group_center - ligand_group_offset
        
        # Calculate initial translation
        ligand.coordinates = original_position.clone()
        initial_translation = initial_ligand_center - ligand.calculate_geometric_center().to(self.device)
        
        processed_points = 0
        valid_confs = 0
        
        # Create coordinate manager once
        coord_manager = CoordinateManager(ligand, device=self.device)
        
        # Pre-calculate receptor group center for validation
        receptor_group_center = self.calculate_center(receptor, receptor_group)
        
        # Create a generator for all translation and rotation combinations
        def scan_generator():
            for dx in translation_values:
                for dy in translation_values:
                    for dz in translation_values:
                        for axis in rotation_axes:
                            for angle in rotation_angles:
                                yield dx, dy, dz, axis, angle
        
        # Process all combinations in a single loop
        for dx, dy, dz, axis, angle in tqdm(scan_generator(), total=total_points, desc=f"{scan_type.capitalize()} scan points"):
            processed_points += 1
            
            # Skip some combinations to reduce redundant calculations
            if scan_type == "coarse":
                # Skip 0 degrees rotation for coarse scan (already checked as base conformation)
                if angle == 0.0:
                    continue
            
            # Create translation vector
            translation = torch.tensor([dx, dy, dz], device=self.device)
            
            # Reset ligand to original position
            ligand.coordinates = original_position.clone()
            
            # Apply translation
            final_translation = initial_translation + translation
            coord_manager.translate_coordinates(final_translation)
            
            # Calculate current ligand geometric center for rotation
            current_ligand_center = ligand.calculate_geometric_center().to(self.device)
            
            # Apply rotation
            coord_manager.rotate_around_axis(axis, angle.item(), current_ligand_center)
            
            # Validate conformation
            if self.validate_conformation(
                receptor=receptor,
                ligand=ligand,
                receptor_group=receptor_group,
                ligand_group=ligand_group,
                vdw_threshold=vdw_threshold,
                max_residue_distance=max_residue_distance
            ):
                # Valid conformation found
                valid_conformation = self._create_ligand_copy(ligand)
                conformations.append(valid_conformation)
                valid_confs += 1
                
                self.logger.info(f"Valid {scan_type} conformation found! Total valid: {valid_confs}")
                self.logger.info(f"  Translation: [{dx:.2f}, {dy:.2f}, {dz:.2f}] Å, Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}], Angle: {angle.item():.2f} degrees")
        
        self.logger.info(f"{scan_type.capitalize()} scan completed: {len(conformations)} valid conformations from {processed_points} processed points")
        return conformations
    
    def _fine_scan_around_position(self, 
                                 receptor: Structure,
                                 ligand: Structure,
                                 receptor_group_center: torch.Tensor,
                                 ligand_group_offset: torch.Tensor,
                                 ligand_group: List[int],
                                 receptor_group: List[int],
                                 center_offset: torch.Tensor,
                                 initial_distance: float,
                                 fine_translation_step: float,
                                 fine_vdw_threshold: float,
                                 max_residue_distance: float,
                                 original_position: torch.Tensor,
                                 rotation_step: float) -> List[Structure]:
        """
        Perform fine scan around a specific position from coarse scan.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure  
            receptor_group_center (torch.Tensor): Receptor residue group center
            ligand_group_offset (torch.Tensor): Ligand group offset from geometric center
            ligand_group (List[int]): Indices of ligand residue group atoms
            receptor_group (List[int]): Indices of receptor residue group atoms
            center_offset (torch.Tensor): Offset from initial position to scan around
            initial_distance (float): Initial distance between residue groups
            fine_translation_step (float): Step size for fine translation scan
            fine_vdw_threshold (float): Maximum VDW energy allowed for fine scan
            max_residue_distance (float): Maximum residue group distance allowed
            original_position (torch.Tensor): Original ligand position
            rotation_step (float): Step size for rotation scan
            
        Returns:
            List[Structure]: List of valid fine scan conformations
        """
        conformations = []
        
        # Define fine scan range (smaller than coarse scan)
        fine_scan_range = 2.0  # Å around the coarse position (2A以内 as requested)
        
        # Generate fine translation grid with specified step size
        num_steps = int(fine_scan_range / fine_translation_step)
        translation_values = torch.linspace(-fine_scan_range, fine_scan_range, num_steps * 2 + 1, device=self.device)
        
        # Generate rotation parameters for fine scan
        rotation_angles = torch.linspace(-30.0, 30.0, int(60.0 / 10.0) + 1, device=self.device)
        rotation_axes = [
            torch.tensor([1.0, 0.0, 0.0], device=self.device),  # x-axis
            torch.tensor([0.0, 1.0, 0.0], device=self.device),  # y-axis  
            torch.tensor([0.0, 0.0, 1.0], device=self.device)   # z-axis
        ]
        
        total_points = len(translation_values) ** 3 * len(rotation_axes) * len(rotation_angles)
        self.logger.info(f"Fine scan grid: {len(translation_values)} points in each dimension, total {total_points} points")
        
        # Calculate initial ligand geometric center for this coarse position
        initial_ligand_group_center = receptor_group_center + torch.tensor([initial_distance, 0.0, 0.0], device=self.device) + center_offset
        initial_ligand_center = initial_ligand_group_center - ligand_group_offset
        
        # Calculate initial translation
        ligand.coordinates = original_position.clone()
        initial_translation = initial_ligand_center - ligand.calculate_geometric_center().to(self.device)
        
        processed_points = 0
        valid_confs = 0
        
        # Create coordinate manager once
        coord_manager = CoordinateManager(ligand, device=self.device)
        
        # Create a generator for all fine scan combinations
        def fine_scan_generator():
            for dx in translation_values:
                for dy in translation_values:
                    for dz in translation_values:
                        for axis in rotation_axes:
                            for angle in rotation_angles:
                                yield dx, dy, dz, axis, angle
        
        # Process all combinations in a single loop
        for dx, dy, dz, axis, angle in tqdm(fine_scan_generator(), total=total_points, desc="Fine scan points"):
            processed_points += 1
            
            # Reset ligand to original position
            ligand.coordinates = original_position.clone()
            
            # Create translation vector
            fine_translation = torch.tensor([dx, dy, dz], device=self.device)
            
            # Apply translation
            final_translation = initial_translation + fine_translation
            coord_manager.translate_coordinates(final_translation)
            
            # Calculate current ligand geometric center for rotation
            current_ligand_center = ligand.calculate_geometric_center().to(self.device)
            
            # Apply rotation
            coord_manager.rotate_around_axis(axis, angle.item(), current_ligand_center)
            
            # Validate conformation
            if self.validate_conformation(
                receptor=receptor,
                ligand=ligand,
                receptor_group=receptor_group,
                ligand_group=ligand_group,
                vdw_threshold=fine_vdw_threshold,
                max_residue_distance=max_residue_distance
            ):
                # Valid conformation found
                valid_conformation = self._create_ligand_copy(ligand)
                conformations.append(valid_conformation)
                valid_confs += 1
                
                self.logger.info(f"Valid fine conformation found! Total valid: {valid_confs}")
                self.logger.info(f"  Fine translation: [{dx:.2f}, {dy:.2f}, {dz:.2f}] Å, Axis: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}], Angle: {angle.item():.2f} degrees")
        
        return conformations
    
    def validate_conformation(self, 
                            receptor: Structure,
                            ligand: Structure,
                            receptor_group: List[int],
                            ligand_group: List[int],
                            vdw_threshold: float = 10000.0,
                            max_residue_distance: float = 10.0) -> bool:
        """
        Validate if a conformation meets the specified criteria with optimized order.
        
        Args:
            receptor (Structure): Receptor protein structure
            ligand (Structure): Ligand protein structure
            receptor_group (List[int]): Indices of receptor residue group atoms
            ligand_group (List[int]): Indices of ligand residue group atoms
            vdw_threshold (float): Maximum VDW energy allowed
            max_residue_distance (float): Maximum residue group distance allowed
            
        Returns:
            bool: True if conformation is valid, False otherwise
        """
        # Calculate centers
        ligand_group_center = self.calculate_center(ligand, ligand_group)
        ligand_center = ligand.calculate_geometric_center().to(self.device)
        
        # Calculate receptor group center only once per scan
        receptor_group_center = self.calculate_center(receptor, receptor_group)
        
        # Calculate distances
        residue_group_distance = torch.norm(receptor_group_center - ligand_group_center).item()
        
        # 1. Check if residue group distance is within limit (残基组距离达标)
        if residue_group_distance > max_residue_distance:
            return False
        
        # Calculate ligand center to receptor residue group distance
        ligand_to_receptor_group = torch.norm(ligand_center - receptor_group_center).item()
        
        # 2. Check if residue group distance <= ligand center to receptor residue group distance (残基组距离小于配体中心到受体残基组距离)
        if residue_group_distance > ligand_to_receptor_group:
            return False
        
        # 3. Calculate VDW energy only if previous criteria are met (最后才计算能量)
        vdw_energy = self.energy_calculator.calculate_vdw_energy(receptor, ligand)
        
        # 4. Check if VDW energy is within threshold
        return vdw_energy < vdw_threshold


# Main function for direct execution
if __name__ == "__main__":
    import argparse
    import sys
    # Use relative imports for main function as well
    from ..utils.logger import Logger
    from ..models.force_field import ForceField
    from ..core.energy_calculator import EnergyCalculator
    from ..models.structure import Structure
    from ..io.parser import PDBParser
    from ..utils.structure_utils import residues_to_atom_indices
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Dedicated Scan Engine for Protein-Protein Docking")
    parser.add_argument("--receptor", required=True, help="Path to receptor PDB file")
    parser.add_argument("--ligand", required=True, help="Path to ligand PDB file")
    parser.add_argument("--receptor-group", required=True, help="Receptor residue group (e.g., 'A:100,A:101,A:102')")
    parser.add_argument("--ligand-group", required=True, help="Ligand residue group (e.g., 'B:200,B:201,B:202')")
    parser.add_argument("--initial-distance", type=float, default=5.0, help="Initial distance between residue groups")
    parser.add_argument("--translation-step", type=float, default=0.5, help="Translation step size in Å")
    parser.add_argument("--rotation-step", type=float, default=30.0, help="Rotation step size in degrees")
    parser.add_argument("--output", default="scan_results.pdb", help="Output PDB file path")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for calculations")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(debug=True)
    
    try:
        # Load PDB files
        PDBParser = PDBParser()
        logger.info(f"Loading receptor PDB file: {args.receptor}")
        receptor = PDBParser.parse_file(args.receptor)
        
        logger.info(f"Loading ligand PDB file: {args.ligand}")
        ligand = PDBParser.parse_file(args.ligand)
        
        # Convert residue groups to atom indices
        logger.info(f"Converting residue groups to atom indices...")
        receptor_group = residues_to_atom_indices(receptor, args.receptor_group.split(","))
        ligand_group = residues_to_atom_indices(ligand, args.ligand_group.split(","))
        
        logger.info(f"Receptor residue group: {len(receptor_group)} atoms")
        logger.info(f"Ligand residue group: {len(ligand_group)} atoms")
        
        # Initialize force field and energy calculator
        force_field = ForceField()
        device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
        energy_calculator = EnergyCalculator(force_field, device)
        
        logger.info(f"Using device: {device}")
        
        # Initialize scan engine
        scan_engine = ScanEngine(logger, energy_calculator, device)
        
        # Run the scan
        logger.info("Starting dedicated scan engine...")
        conformations = scan_engine.dedicated_scan(
            receptor=receptor,
            ligand=ligand,
            receptor_group=receptor_group,
            ligand_group=ligand_group,
            initial_distance=args.initial_distance,
            coarse_translation_step=args.translation_step,
            fine_translation_step=args.translation_step * 0.2,
            rotation_step=args.rotation_step,
            max_translation_range=5.0,
            coarse_vdw_threshold=100000.0,
            fine_vdw_threshold=20000.0,
            max_residue_distance=10.0
        )
        
        logger.info(f"Generated {len(conformations)} valid conformations")
        
        if not conformations:
            logger.warning("No valid conformations found")
            sys.exit(1)
        
        # Save the best conformation (first one in the list)
        best_conformation = conformations[0]
        logger.info(f"Saving best conformation to {args.output}")
        
        # Merge receptor and ligand conformations for output
        merged_structure = receptor.copy()
        merged_structure.add_structure(best_conformation)
        
        # Save to PDB file
        with open(args.output, "w") as f:
            for line in merged_structure.to_pdb_lines():
                f.write(line + "\n")
        
        logger.info(f"Scan completed successfully! Output saved to {args.output}")
        logger.info(f"Generated {len(conformations)} valid conformations")
        
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)