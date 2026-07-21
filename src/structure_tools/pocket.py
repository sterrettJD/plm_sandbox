import numpy as np
from scipy.spatial.distance import cdist

from src.structure_tools.pdb_parser import PDBData, parse_pdb_file


def find_pocket(pdb_data: PDBData, pocket_size: int = 5) -> list:
    ligand_coords = pdb_data.filter_record_type("HETATM").numpy_coords(only_atom=False)
    atom_only = pdb_data.filter_record_type("ATOM")
    atom_coords = atom_only.numpy_coords(only_atom=False)

    dist = cdist(ligand_coords, atom_coords, metric="euclidean")

    hits = dist <= pocket_size
    pocket_indexes = np.where(hits.any(axis=0))[0]

    return list(int(x) for x in pocket_indexes)


def main():
    path = "datasets/structure_project/P00520_archive-PDB/pdb1abo.ent"
    parsed = parse_pdb_file(path)
    parsed.head()

    pocket_indexes = find_pocket(parsed)
    print(f"Pocket indexes: {pocket_indexes}")

    parsed.filter_by_indexes(pocket_indexes).head()
    return


if __name__ == "__main__":
    main()
