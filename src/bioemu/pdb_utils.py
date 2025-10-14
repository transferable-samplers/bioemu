import logging
from typing import List, Set, Tuple
from Bio.PDB import PDBParser
import numpy as np

def normalize_atom_name(atom_name: str) -> str:
    """
    Normalize atom names for matching (e.g., H -> H1).

    Args:
        atom_name (str): Atom name from PDB file.

    Returns:
        str: Normalized atom name.
    """
    if atom_name == "H":  # BioEmu naming
        return "H1"
    return atom_name

def get_atom_list(pdb_file: str) -> List[Tuple[int, str, str]]:
    """
    Return a list of (resSeq, resName, atomName) for each atom in PDB order.

    Args:
        pdb_file (str): Path to PDB file.

    Returns:
        List[Tuple[int, str, str]]: List of tuples containing residue sequence number,
            residue name, and atom name.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)
    atoms = []
    for atom in structure.get_atoms():
        parent = atom.get_parent()
        resSeq = parent.get_id()[1]  # residue number
        resName = parent.get_resname().strip()
        atomName = normalize_atom_name(atom.get_name().strip())
        atoms.append((resSeq, resName, atomName))
    return atoms

def get_atom_set(pdb_file: str) -> Set[Tuple[int, str, str]]:
    """
    Return a set of (resSeq, resName, atomName) for each atom in PDB.

    Args:
        pdb_file (str): Path to PDB file.

    Returns:
        Set[Tuple[int, str, str]]: Set of tuples containing residue sequence number,
            residue name, and atom name.
    """
    return set(get_atom_list(pdb_file))

def check_atom_match(ref_pdb: str, bioemu_pdb: str) -> bool:
    """
    Check if atom sets match between reference and BioEmu PDB files.

    Args:
        ref_pdb (str): Path to reference PDB file.
        bioemu_pdb (str): Path to BioEmu PDB file.

    Returns:
        bool: True if atom sets match exactly.

    Raises:
        ValueError: If atom sets do not match.
    """
    ref_atoms = get_atom_set(ref_pdb)
    bioemu_atoms = get_atom_set(bioemu_pdb)

    if ref_atoms == bioemu_atoms:
        logging.info("✅ Atom sets match exactly.")
        return True

    missing_in_bioemu = ref_atoms - bioemu_atoms
    extra_in_bioemu = bioemu_atoms - ref_atoms

    if missing_in_bioemu:
        logging.warning("❌ Missing atoms in BioEmu PDB:")
        for atom in sorted(missing_in_bioemu):
            print("  ", atom)

    if extra_in_bioemu:
        logging.warning("❌ Extra atoms in BioEmu PDB:")
        for atom in sorted(extra_in_bioemu):
            print("  ", atom)

    raise ValueError("Atom sets do NOT match between reference and BioEmu PDBs!")

def reorder_coordinates(
    ref_pdb: str,
    bioemu_pdb: str,
    bioemu_npy: np.ndarray
) -> np.ndarray:
    """
    Reorder BioEmu coordinates to match reference PDB order.

    Args:
        ref_pdb (str): Path to reference PDB file.
        bioemu_pdb (str): Path to BioEmu PDB file.
        bioemu_npy (np.ndarray): BioEmu coordinates array of shape (N_frames, N_atoms, 3).

    Returns:
        np.ndarray: Reordered BioEmu coordinates array.

    Raises:
        ValueError: If an atom in reference PDB is not found in BioEmu PDB.
    """
    ref_atoms = get_atom_list(ref_pdb)
    bioemu_atoms = get_atom_list(bioemu_pdb)

    mapping = []
    for ref_atom in ref_atoms:
        try:
            idx = bioemu_atoms.index(ref_atom)
            mapping.append(idx)
        except ValueError:
            raise ValueError(f"Atom {ref_atom} not found in BioEmu PDB (after normalization)!")

    mapping = np.array(mapping)

    if np.array_equal(mapping, np.arange(len(mapping))):
        logging.info("ℹ️ No reordering needed: BioEmu coordinates already match reference PDB order.")
        return bioemu_npy
    else:
        logging.info("ℹ️ Reordering BioEmu coordinates to match reference PDB order.")
        return bioemu_npy[:, mapping, :]
