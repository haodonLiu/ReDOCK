#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Force Field Reader Module

Handles reading and parsing of force field XML files like ff14SB.xml.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


class ForceField:
    """
    Force Field class to read and store force field parameters from XML files.
    
    Attributes:
        atom_types (Dict[str, Dict]): Mapping of atom type names to their properties
        residues (Dict[str, Dict]): Mapping of residue names to their atom information
        nonbonded_params (Dict[str, Dict]): Non-bonded parameters (sigma, epsilon)
        bond_params (Dict[Tuple[str, str], Dict]): Bond length parameters
        angle_params (Dict[Tuple[str, str, str], Dict]): Bond angle parameters
        dihedral_params (Dict[Tuple[str, str, str, str], List[Dict]]): Dihedral angle parameters
        version (str): Force field version information
    """
    def __init__(self, logger=None):
        self.atom_types: Dict[str, Dict] = {}
        self.residues: Dict[str, Dict] = {}
        self.nonbonded_params: Dict[str, Dict] = {}
        self.bond_params: Dict[Tuple[str, str], Dict] = {}
        self.angle_params: Dict[Tuple[str, str, str], Dict] = {}
        self.dihedral_params: Dict[Tuple[str, str, str, str], List[Dict]] = {}
        self.version: str = ""
        self.logger = logger
    
    def read_xml(self, file_path: str | List[str]) -> None:
        """
        Read and parse force field parameters from XML file(s).
        
        Args:
            file_path (str | List[str]): Path or list of paths to the force field XML files
        """
        # Handle multiple files
        file_paths = [file_path] if isinstance(file_path, str) else file_path
        
        for fp in file_paths:
            if self.logger is not None:
                self.logger.debug(f"Reading force field from {fp}")
            tree = ET.parse(fp)
            root = tree.getroot()
            
            # Parse atom types
            for atom_type_elem in root.findall('.//AtomTypes/Type'):
                type_name = atom_type_elem.get('name')
                if type_name:
                    self.atom_types[type_name] = {
                        'element': atom_type_elem.get('element', ''),
                        'class': atom_type_elem.get('class', ''),
                        'mass': float(atom_type_elem.get('mass', 0.0))
                    }
            
            # Parse residues
            for residue_elem in root.findall('.//Residues/Residue'):
                res_name = residue_elem.get('name')
                if res_name:
                    residue_data = {
                        'atoms': {},
                        'bonds': []
                    }
                    
                    # Parse atoms in residue
                    for atom_elem in residue_elem.findall('.//Atom'):
                        atom_name = atom_elem.get('name')
                        if atom_name:
                            residue_data['atoms'][atom_name] = {
                                'type': atom_elem.get('type', ''),
                                'charge': float(atom_elem.get('charge', 0.0))
                            }
                    
                    # Parse bonds in residue
                    for bond_elem in residue_elem.findall('.//Bond'):
                        atom1 = bond_elem.get('atomName1')
                        atom2 = bond_elem.get('atomName2')
                        if atom1 and atom2:
                            residue_data['bonds'].append((atom1, atom2))
                    
                    self.residues[res_name] = residue_data
            
            # Parse nonbonded parameters (sigma and epsilon)
            for nonbonded_elem in root.findall('.//NonbondedForce/Atom'):
                atom_class = nonbonded_elem.get('class')
                if atom_class:
                    self.nonbonded_params[atom_class] = {
                        'sigma': float(nonbonded_elem.get('sigma', 0.0)),
                        'epsilon': float(nonbonded_elem.get('epsilon', 0.0))
                    }
            
            # Parse bond parameters
            for bond_elem in root.findall('.//HarmonicBondForce/Bond'):
                atom1 = bond_elem.get('class1')
                atom2 = bond_elem.get('class2')
                if atom1 and atom2:
                    # Create sorted tuple for symmetry
                    bond_key = tuple(sorted([atom1, atom2]))
                    self.bond_params[bond_key] = {
                        'r0': float(bond_elem.get('length', 1.0)),
                        'k': float(bond_elem.get('k', 1000.0))
                    }
            
            # Parse angle parameters
            for angle_elem in root.findall('.//HarmonicAngleForce/Angle'):
                atom1 = angle_elem.get('class1')
                atom2 = angle_elem.get('class2')
                atom3 = angle_elem.get('class3')
                if atom1 and atom2 and atom3:
                    # Create tuple (keep central atom in middle)
                    angle_key = (atom1, atom2, atom3)
                    self.angle_params[angle_key] = {
                        'theta0': float(angle_elem.get('angle', 109.5)),
                        'k': float(angle_elem.get('k', 100.0))
                    }
            
            # Parse dihedral parameters
            for dihedral_elem in root.findall('.//PeriodicTorsionForce/Torsion'):
                atom1 = dihedral_elem.get('class1')
                atom2 = dihedral_elem.get('class2')
                atom3 = dihedral_elem.get('class3')
                atom4 = dihedral_elem.get('class4')
                if atom1 and atom2 and atom3 and atom4:
                    # Create tuple for dihedral
                    dihedral_key = (atom1, atom2, atom3, atom4)
                    if dihedral_key not in self.dihedral_params:
                        self.dihedral_params[dihedral_key] = []
                    
                    self.dihedral_params[dihedral_key].append({
                        'V': float(dihedral_elem.get('k', 1.0)),
                        'n': int(dihedral_elem.get('periodicity', 1)),
                        'delta': float(dihedral_elem.get('phase', 0.0))
                    })
    
    def get_atom_type(self, atom_type_name: str) -> Dict:
        """
        Get atom type properties by name.
        
        Args:
            atom_type_name (str): Name of the atom type
            
        Returns:
            Dict: Atom type properties
        """
        return self.atom_types.get(atom_type_name, {})
    
    def get_residue(self, residue_name: str) -> Dict:
        """
        Get residue information by name.
        
        Args:
            residue_name (str): Name of the residue
            
        Returns:
            Dict: Residue information
        """
        return self.residues.get(residue_name, {})
    
    def get_atom_charge(self, residue_name: str, atom_name: str) -> float:
        """
        Get atom charge from force field for a specific residue and atom.
        
        Args:
            residue_name (str): Name of the residue
            atom_name (str): Name of the atom
            
        Returns:
            float: Atom charge if found, 0.0 otherwise
        """
        residue = self.get_residue(residue_name)
        if residue and 'atoms' in residue:
            atom = residue['atoms'].get(atom_name, {})
            return atom.get('charge', 0.0)
        return 0.0
    
    def get_atom_mass(self, atom_type_name: str) -> float:
        """
        Get atom mass from force field for a specific atom type.
        
        Args:
            atom_type_name (str): Name of the atom type
            
        Returns:
            float: Atom mass if found, 0.0 otherwise
        """
        atom_type = self.get_atom_type(atom_type_name)
        return atom_type.get('mass', 0.0)
    
    def get_atom_type_for_residue(self, residue_name: str, atom_name: str) -> str:
        """
        Get atom type for a specific residue and atom name.
        
        Args:
            residue_name (str): Name of the residue
            atom_name (str): Name of the atom
            
        Returns:
            str: Atom type if found, empty string otherwise
        """
        residue = self.get_residue(residue_name)
        if residue and 'atoms' in residue:
            atom = residue['atoms'].get(atom_name, {})
            return atom.get('type', '')
        return ''
    
    def get_vdw_params(self, atom_type_name: str) -> Tuple[float, float]:
        """
        Get van der Waals parameters (sigma, epsilon) for an atom type.
        
        Args:
            atom_type_name (str): Name of the atom type
            
        Returns:
            Tuple[float, float]: sigma and epsilon parameters (in nm and kJ/mol)
        """
        # Get atom class from atom type
        atom_type = self.get_atom_type(atom_type_name)
        atom_class = atom_type.get('class', '')
        
        # Get nonbonded parameters for this class
        if atom_class in self.nonbonded_params:
            params = self.nonbonded_params[atom_class]
            return (params['sigma'], params['epsilon'])
        
        # Default values if parameters not found
        return (0.35, 0.15)  # Default sigma and epsilon values
    
    def get_bond_params(self, atom_type1: str, atom_type2: str) -> Dict[str, float]:
        """
        Get bond length parameters for a pair of atom types.
        
        Args:
            atom_type1 (str): First atom type
            atom_type2 (str): Second atom type
            
        Returns:
            Dict[str, float]: Bond length parameters (r0: equilibrium bond length, k: force constant)
        """
        # Get atom classes
        class1 = self.get_atom_type(atom_type1).get('class', '')
        class2 = self.get_atom_type(atom_type2).get('class', '')
        
        # Create sorted key for symmetry
        bond_key = tuple(sorted([class1, class2]))
        
        # Return parameters if found
        if bond_key in self.bond_params:
            return self.bond_params[bond_key]
        
        # Default values if parameters not found
        return {'r0': 1.0, 'k': 1000.0}
    
    def get_angle_params(self, atom_type1: str, atom_type2: str, atom_type3: str) -> Dict[str, float]:
        """
        Get bond angle parameters for a triplet of atom types.
        
        Args:
            atom_type1 (str): First atom type
            atom_type2 (str): Second atom type (central atom)
            atom_type3 (str): Third atom type
            
        Returns:
            Dict[str, float]: Bond angle parameters (theta0: equilibrium bond angle, k: force constant)
        """
        # Get atom classes
        class1 = self.get_atom_type(atom_type1).get('class', '')
        class2 = self.get_atom_type(atom_type2).get('class', '')
        class3 = self.get_atom_type(atom_type3).get('class', '')
        
        # Create key (keep central atom in middle)
        angle_key = (class1, class2, class3)
        
        # Return parameters if found
        if angle_key in self.angle_params:
            return self.angle_params[angle_key]
        
        # Default values if parameters not found
        return {'theta0': 109.5, 'k': 100.0}
    
    def get_dihedral_params(self, atom_type1: str, atom_type2: str, 
                           atom_type3: str, atom_type4: str) -> List[Dict[str, float]]:
        """
        Get dihedral angle parameters for a quartet of atom types.
        
        Args:
            atom_type1 (str): First atom type
            atom_type2 (str): Second atom type
            atom_type3 (str): Third atom type
            atom_type4 (str): Fourth atom type
            
        Returns:
            List[Dict[str, float]]: List of dihedral angle parameters (V: barrier height, n: periodicity, delta: phase)
        """
        # Get atom classes
        class1 = self.get_atom_type(atom_type1).get('class', '')
        class2 = self.get_atom_type(atom_type2).get('class', '')
        class3 = self.get_atom_type(atom_type3).get('class', '')
        class4 = self.get_atom_type(atom_type4).get('class', '')
        
        # Create key
        dihedral_key = (class1, class2, class3, class4)
        
        # Return parameters if found
        if dihedral_key in self.dihedral_params:
            return self.dihedral_params[dihedral_key]
        
        # Default values if parameters not found
        return [{'V': 1.0, 'n': 3, 'delta': 0.0}]
    
    def __repr__(self) -> str:
        """String representation of the force field"""
        return f"ForceField(atom_types={len(self.atom_types)}, residues={len(self.residues)}, nonbonded_params={len(self.nonbonded_params)}, bond_params={len(self.bond_params)}, angle_params={len(self.angle_params)}, dihedral_params={len(self.dihedral_params)})"
