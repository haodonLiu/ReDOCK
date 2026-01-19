#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDB Atom Data Model

Defines the PDBAtom class for storing atom structural information.
"""


class PDBAtom:
    """
    Atom data class for storing individual atom structural information.
    
    Attributes:
        record_type (str): Record type (ATOM or HETATM)
        atom_serial (int): Atom serial number
        atom_name (str): Atom name
        alt_loc (str): Alternate location indicator
        res_name (str): Residue name
        chain_id (str): Chain identifier
        res_seq (int): Residue sequence number
        i_code (str): Insertion code
        occupancy (float): Occupancy
        b_factor (float): Temperature factor
        element (str): Element symbol
        charge (str): Charge
    """
    def __init__(self, record_type: str, atom_serial: int, atom_name: str, alt_loc: str,
                 res_name: str, chain_id: str, res_seq: int, i_code: str, occupancy: float,
                 b_factor: float, element: str, charge: str):
        self.record_type = record_type  # 1-6: Record type
        self.atom_serial = atom_serial  # 7-11: Atom serial number
        self.atom_name = atom_name      # 13-16: Atom name
        self.alt_loc = alt_loc          # 17: Alternate location indicator
        self.res_name = res_name        # 18-20: Residue name
        self.chain_id = chain_id        # 22: Chain identifier
        self.res_seq = res_seq          # 23-26: Residue sequence number
        self.i_code = i_code            # 27: Insertion code
        self.occupancy = occupancy      # 55-60: Occupancy
        self.b_factor = b_factor        # 61-66: Temperature factor
        self.element = element          # 77-78: Element symbol
        self.charge = charge            # 79-80: Charge

    def __repr__(self) -> str:
        """String representation of the atom"""
        return f"PDBAtom({self.record_type} {self.atom_serial} {self.atom_name} {self.res_name} {self.chain_id}{self.res_seq})"
