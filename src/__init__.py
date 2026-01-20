#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Processor Package

核心功能包，提供蛋白质结构处理、对接和分析功能。
"""

from .cli import main_cli
from .core.docking import Docking
from .core.energy_calculator import EnergyCalculator
from .core.statistics import StructureStatistics
from .io.parser import PDBParser
from .io.writer import PDBWriter
from .models.structure import Structure
from .models.force_field import ForceField
from .utils.logger import Logger

__all__ = [
    'main_cli',
    'Docking',
    'EnergyCalculator',
    'StructureStatistics',
    'PDBParser',
    'PDBWriter',
    'Structure',
    'ForceField',
    'Logger'
]

__version__ = '1.0.0'
__author__ = 'Protein Dock Team'
__description__ = 'Protein-protein docking software with pre-alignment and conformation search'
