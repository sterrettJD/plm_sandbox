import numpy as np


class PDBData:
    def __init__(
        self,
        name,
        description=None,
        record_type=None,
        element=None,
        coordinates=None,
        residue_name=None,
        residue_number=None,
        chain=None,
    ):
        self.name = name
        self.description = description
        self.record_type = record_type
        self.elements = element
        self.coordinates = coordinates
        self.residue_name = residue_name
        self.residue_number = residue_number
        self.chain = chain

    def head(self, n: int = 5) -> None:
        print(f"PDB Data for {self.name}")
        for i in range(n):
            print(f"{self.record_type[i]} {self.elements[i]}: {self.coordinates[i]}")

    def numpy_coords(self, only_atom: bool = True) -> np.ndarray:
        if only_atom:
            indexes = [record == "ATOM" for record in self.record_type]
            return np.array([c for c, m in zip(self.coordinates, indexes) if m])
        else:
            return np.array(self.coordinates)


def parse_pdb_file(path: str, ion_blocklist=["HOH"]) -> PDBData:
    """
    Reads a PDB file and parses into a PDBData object.

    Args:
        path (str): filepath
        ion_blocklist (list[str]): HETATM resnames to disclude.
            Default includes water (HOH).

    Returns: PDBData
    """
    with open(path, "r") as f:
        lines = f.readlines()

    in_model = True
    record_type = []
    element = []
    coordinates = []
    residue_name = []
    residue_number = []
    chain = []

    for line in lines:
        if line.startswith("MODEL"):
            model_num = int(line[10:14])
            in_model = True
            if model_num > 1:
                break  # stop after first model
        elif line.startswith("ENDMDL"):
            in_model = False
        elif in_model and line.startswith(("ATOM", "HETATM")):
            # skip any entries where we're in the blocklist
            if line.startswith("HETATM") and line[17:20].strip() in ion_blocklist:
                continue

            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            record_type.append(line[0:6].strip())
            element.append(line[76:78].strip())

            x, y, z = line[30:38].lstrip(), line[38:46].lstrip(), line[46:54].lstrip()
            coordinates.append((float(x), float(y), float(z)))

            residue_name.append(line[17:20].strip())
            residue_number.append(int(line[22:26]))

            chain.append(line[21])

    return PDBData(
        name=path.split("/")[-1].split(".")[-2],
        record_type=record_type,
        element=element,
        residue_name=residue_name,
        residue_number=residue_number,
        coordinates=coordinates,
        chain=chain,
    )


def main():
    path = "datasets/structure_project/P00520_archive-PDB/pdb1abo.ent"
    parsed = parse_pdb_file(path)
    parsed.head()

    print("Coords Array:")
    print(parsed.numpy_coords()[0:5])


if __name__ == "__main__":
    main()
