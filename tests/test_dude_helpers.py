import numpy as np
from experiments.dude_validation import (assignment_rmsd, enrichment_factor, parse_pdbqt_pose, read_ism)


def test_enrichment_factor_perfect_and_random():
    y = np.array([1] * 10 + [0] * 90)
    assert enrichment_factor(y, y.astype(float), 0.1) == 10.0
    assert abs(enrichment_factor(y, np.zeros(100) + np.arange(100) % 7, 1.0) - 1.0) < 1e-9


def test_read_ism(tmp_path):
    p = tmp_path / "a.ism"; p.write_text("CCO 1 x\nCCN 2\n\nbad\n")
    assert read_ism(p) == [("CCO", "1"), ("CCN", "2")]


def test_assignment_rmsd_symmetry_and_mismatch():
    xyz = np.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0.]]); el = ["C", "C", "O"]
    assert assignment_rmsd(xyz, el, xyz[[1, 0, 2]], ["C", "C", "O"]) < 1e-9      # swapped identical atoms
    assert assignment_rmsd(xyz + 1.0, el, xyz, el) > 1.0
    assert np.isnan(assignment_rmsd(xyz, el, xyz[:2], el[:2]))


def test_parse_pdbqt_pose_skips_hydrogens_and_later_models():
    txt = ("MODEL 1\n"
           "ATOM      1  C1  LIG A   1       1.000   2.000   3.000  0.00  0.00    +0.000 A \n"
           "ATOM      2  H1  LIG A   1       1.500   2.000   3.000  0.00  0.00    +0.000 HD\n"
           "ATOM      3  O1  LIG A   1       4.000   2.000   3.000  0.00  0.00    -0.300 OA\n"
           "ENDMDL\nMODEL 2\nATOM      1  C1  LIG A   1       9.0 9.0 9.0\nENDMDL\n")
    xyz, el = parse_pdbqt_pose(txt)
    assert el == ["C", "O"] and xyz.shape == (2, 3) and xyz[0, 0] == 1.0


def test_matched_decoys_are_property_matched_and_unique():
    from experiments.dude_validation import matched_decoys
    pool = ["CCCCCC", "CCCCCCC", "c1ccccc1", "CCO", "CCCCCCCC", "c1ccc(C)cc1", "CC(C)CCCC", "[NH4+]", "CCCCCCCCC"]
    out = matched_decoys(["CCCCCCCO"], pool, 3)
    assert len(out) == 3 and len(set(out)) == 3 and "[NH4+]" not in out


def test_native_ligand_from_pdb_picks_first_copy(tmp_path):
    from experiments.dude_validation import native_ligand_from_pdb
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "HETATM    2  C1  LIG A 301       1.000   1.000   1.000  1.00  0.00           C\n"
        "HETATM    3  O1  LIG A 301       2.000   1.000   1.000  1.00  0.00           O\n"
        "HETATM    4  C1  LIG B 302       9.000   9.000   9.000  1.00  0.00           C\n")
    m = native_ligand_from_pdb(pdb, "LIG")
    assert m.GetNumAtoms() == 2


def test_parse_pdbqt_models_returns_all_poses():
    from experiments.dude_validation import parse_pdbqt_models
    one = "ATOM      1  C1  LIG A   1       {:.3f}   0.000   0.000  0.00  0.00    +0.000 A \n"
    txt = "MODEL 1\n" + one.format(1.0) + "ENDMDL\nMODEL 2\n" + one.format(5.0) + "ENDMDL\n"
    models = parse_pdbqt_models(txt)
    assert len(models) == 2 and models[0][0][0, 0] == 1.0 and models[1][0][0, 0] == 5.0
