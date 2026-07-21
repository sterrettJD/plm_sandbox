from __future__ import annotations

import numpy as np


class PDBData:
    def __init__(
        self,
        name: str,
        description: str | None = None,
        record_type: list[str] | None = None,
        element: list[str] | None = None,
        coordinates: list[tuple[float | int]] | None = None,
        residue_name: list[str] | None = None,
        residue_number: list[int | float] | None = None,
        chain: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.record_type = record_type
        self.element = element
        self.coordinates = coordinates
        self.residue_name = residue_name
        self.residue_number = residue_number
        self.chain = chain

    def head(self, n: int = 5) -> None:
        print(f"PDB Data for {self.name}")
        print(f"{sum(r == 'ATOM' for r in self.record_type)} ATOM")
        print(f"{sum(r == 'HETATM' for r in self.record_type)} HETATM")

        for i in range(n):
            print(f"{self.record_type[i]} {self.element[i]}: {self.coordinates[i]}")

    def filter_record_type(self, type: str) -> PDBData:
        """
        Filters the object to only entries that are of record type `type`
        Args:
            type (str): the record_type to include. ATOM or HETATM

        Returns: PDBData object
        """
        accepted = {"ATOM", "HETATM"}
        if type not in accepted:
            raise ValueError(f"Please pass a supported type ({accepted})")

        indexes = [record == type for record in self.record_type]

        return PDBData(
            f"{self.name} ({type} filtered)",
            record_type=[c for c, m in zip(self.record_type, indexes) if m],
            element=[c for c, m in zip(self.element, indexes) if m],
            residue_name=[c for c, m in zip(self.residue_name, indexes) if m],
            residue_number=[c for c, m in zip(self.residue_number, indexes) if m],
            coordinates=[c for c, m in zip(self.coordinates, indexes) if m],
            chain=[c for c, m in zip(self.chain, indexes) if m],
        )

    def filter_by_indexes(self, indexes: list) -> PDBData:
        return PDBData(
            f"{self.name} (index filtered)",
            record_type=[self.record_type[i] for i in indexes],
            element=[self.element[i] for i in indexes],
            residue_name=[self.residue_name[i] for i in indexes],
            residue_number=[self.residue_number[i] for i in indexes],
            coordinates=[self.coordinates[i] for i in indexes],
            chain=[self.chain[i] for i in indexes],
        )

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

    filtered = parsed.filter_record_type("ATOM")
    filtered.head()


if __name__ == "__main__":
    main()
