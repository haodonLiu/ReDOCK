#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Dock Package

核心功能包，以scan_engine.py为核心，提供蛋白质结构处理、对接和分析功能。
"""

from .core.energy_calculator import EnergyCalculator
from .core.scan_engine import ScanEngine
from .utils.io_utils import PDBIO as PDBParser
from .utils.io_utils import PDBIO as PDBWriter
from .models.topology import Topology
from .models.coordinate import Coordinate
from .models.force_field import ForceField
from .utils.logger import Logger

__all__ = [
    'EnergyCalculator',
    'ScanEngine',
    'PDBParser',
    'PDBWriter',
    'Topology',
    'Coordinate',
    'ForceField',
    'Logger'
]

__version__ = '1.0.0'
__author__ = 'Protein Dock Team'
__description__ = 'Protein-protein docking software with scan engine and enhanced IO module'
