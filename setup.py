#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for PDB Processor package
"""

from setuptools import setup, find_packages
import os

# Read requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Read README.md for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pdb-processor',
    version='1.0.0',
    author='Protein Dock Team',
    author_email='contact@proteindock.example.com',
    description='Protein-protein docking software with pre-alignment and conformation search',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/proteindock/pdb-processor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        '': ['*.py'],
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'pdb-processor = src.cli:main_cli',
            'pdb-dock = src.cli:main_cli',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
)
