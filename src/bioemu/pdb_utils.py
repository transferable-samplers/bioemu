import logging

from Bio.PDB import PDBParser
import numpy as np

def normalize_atom_name(atom_name):
    """Normalize atom names for matching (e.g., H -> H1)."""
    if atom_name == "H":  # BioEmu naming
        return "H1"
    return atom_name

def get_atom_list(pdb_file):
    """Return a list of (resSeq, resName, atomName) for each atom in PDB order."""
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

def get_atom_set(pdb_file):
    """Return a set of (resSeq, resName, atomName) for each atom in PDB."""
    return set(get_atom_list(pdb_file))

def check_atom_match(ref_pdb, bioemu_pdb):
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

def reorder_coordinates(ref_pdb, bioemu_pdb, bioemu_npy):
    """Reorder BioEmu coordinates to match reference PDB order."""
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
