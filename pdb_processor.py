#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Processor Command Line Tool

命令行工具，用于处理PDB文件、预对齐蛋白质和执行对接构象搜索。
"""

import sys
from src import main_cli

if __name__ == "__main__":
    exit_code = main_cli()
    sys.exit(exit_code)
