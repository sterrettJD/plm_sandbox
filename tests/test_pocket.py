from src.structure_tools import pdb_parser, pocket


def test_basic_pocket():
    parsed = pdb_parser.parse_pdb_file("tests/data/pdb/pocket_test.ent")
    pocket_indexes = pocket.find_pocket(parsed)
    assert pocket_indexes == [0, 1, 2]
