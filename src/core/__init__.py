#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Module

Provides core functionality for protein-protein docking, including energy calculation and scan engine.
"""

from .energy_calculator import EnergyCalculator
from .scan_engine import ScanEngine
from .coordinate_manager import CoordinateManager

__all__ = [
    'EnergyCalculator',
    'ScanEngine',
    'CoordinateManager'
]
