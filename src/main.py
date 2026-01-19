#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Processor Main Application

Main entry point for the PDB processor application.
"""

import sys
from src.cli import main_cli


if __name__ == "__main__":
    exit_code = main_cli()
    sys.exit(exit_code)
