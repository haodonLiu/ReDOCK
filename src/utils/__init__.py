#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities Module

Collection of utility functions and tools for protein docking and structure analysis.
"""

from .logger import Logger
from .structure_utils import residues_to_atom_indices, calculate_structure_center
from .io_utils import PDBIO

# Backward compatibility
# PDBIO is already imported from .io_utils

from .coordinate_utils import (
    calculate_geometric_center,
    translate_coordinates,
    center_coordinates,
    create_rotation_matrix,
    rotate_coordinates,
    rotate_coordinates_euler,
    rotate_around_axis,
    calculate_rmsd,
    kabsch_algorithm,
    align_coordinates,
    generate_random_rotation,
    generate_random_translation
)
from .distance_utils import (
    calculate_atom_pair_distances,
    calculate_distance_vectors,
    find_minimum_distance,
    find_all_distances_below_cutoff,
    count_clashes,
    calculate_closest_atoms,
    calculate_radius_of_gyration,
    is_within_distance,
    batch_calculate_distances
)
from .conformation_utils import (
    generate_random_conformation,
    filter_clashing_conformations,
    calculate_conformation_rmsd,
    align_conformation_to_reference,
    cluster_conformations,
    select_representative_conformations,
    calculate_conformation_energies,
    rank_conformations_by_energy,
    analyze_conformation_diversity,
    generate_conformation_pool,
    validate_conformation
)
from .clustering import (
    kmeans_clustering_conformations,
    hierarchical_clustering,
    dbscan_clustering,
    calculate_silhouette_score,
    calculate_calinski_harabasz_score,
    calculate_davies_bouldin_score,
    evaluate_clustering,
    optimize_clustering_parameters
)

__all__ = [
    # Core utilities
    'Logger',
    'PDBIO',
    'residues_to_atom_indices',

    
    # Coordinate utilities
    'calculate_geometric_center',
    'translate_coordinates',
    'center_coordinates',
    'create_rotation_matrix',
    'rotate_coordinates',
    'rotate_coordinates_euler',
    'rotate_around_axis',
    'calculate_rmsd',
    'kabsch_algorithm',
    'align_coordinates',
    'generate_random_rotation',
    'generate_random_translation',
    
    # Distance utilities
    'calculate_atom_pair_distances',
    'calculate_distance_vectors',
    'find_minimum_distance',
    'find_all_distances_below_cutoff',
    'count_clashes',
    'calculate_closest_atoms',
    'calculate_radius_of_gyration',
    'is_within_distance',
    'batch_calculate_distances',
    
    # Conformation utilities
    'generate_random_conformation',
    'filter_clashing_conformations',
    'calculate_conformation_rmsd',
    'align_conformation_to_reference',
    'cluster_conformations',
    'select_representative_conformations',
    'calculate_conformation_energies',
    'rank_conformations_by_energy',
    'analyze_conformation_diversity',
    'generate_conformation_pool',
    'validate_conformation',
    
    # Clustering algorithms
    'kmeans_clustering_conformations',
    'hierarchical_clustering',
    'dbscan_clustering',
    'calculate_silhouette_score',
    'calculate_calinski_harabasz_score',
    'calculate_davies_bouldin_score',
    'evaluate_clustering',
    'optimize_clustering_parameters'
]
