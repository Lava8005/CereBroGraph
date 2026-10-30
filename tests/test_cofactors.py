import shutil
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from target.cofactors import append_to_receptor, extract_cofactor_lines, has_residue, protein_lines


def _pdb(tmp_path):
    m = Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO")); AllChem.EmbedMolecule(m, randomSeed=3)
    blk = Chem.MolToPDBBlock(m).replace("UNL", "FAD").replace("ATOM  ", "HETATM")
    prot = "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\nTER\n"
    p = tmp_path / "x.pdb"; p.write_text(prot + blk)
    return p


def test_extract_and_protein_split(tmp_path):
    p = _pdb(tmp_path)
    cof = extract_cofactor_lines(p, ["FAD"])
    assert cof and all(l[17:20] == "FAD" for l in cof)
    prot = protein_lines(p, exclude_resnames=["FAD"])
    assert any(l.startswith("ATOM") for l in prot) and not any("FAD" in l for l in prot)


def test_cofactor_atom_records_are_found_too(tmp_path):
    p = tmp_path / "d.pdb"   # DUD-E stores FAD as ATOM records
    p.write_text("ATOM      1  C1  FAD A 501       1.000   1.000   1.000  1.00  0.00           C\n")
    assert len(extract_cofactor_lines(p, ["fad"])) == 1


@pytest.mark.skipif(shutil.which("obabel") is None, reason="needs Open Babel")
def test_cofactor_roundtrip_into_receptor(tmp_path):
    from target.cofactors import cofactor_to_pdbqt_lines
    p = _pdb(tmp_path)
    lines = cofactor_to_pdbqt_lines(extract_cofactor_lines(p, ["FAD"]))
    assert lines
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("ATOM      1  N   ALA A   1      11.104   6.134  -6.504  0.00  0.00    +0.000 N \nEND\n")
    n = append_to_receptor(rec, lines)
    assert n == len(lines) and has_residue(rec, ["FAD"])
    serials = [int(l[6:11]) for l in rec.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert serials == sorted(set(serials))


def test_two_protein_copies_keep_the_one_near_the_ligand(tmp_path):
    def atom(i, num, x):
        return f"ATOM  {i:5d}  CA  ALA A{num:4d}    {x:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C"
    lines = [atom(1, 1, 0.0), atom(2, 2, 3.0), atom(3, 3, 6.0),          # copy 1 near the origin
             atom(4, 1, 100.0), atom(5, 2, 103.0), atom(6, 3, 106.0)]    # copy 2 far away, numbering restarts
    p = tmp_path / "two.pdb"; p.write_text("\n".join(lines) + "\n")
    near_first = protein_lines(p, ligand_center=(3.0, 0.0, 0.0))
    near_second = protein_lines(p, ligand_center=(103.0, 0.0, 0.0))
    xs = lambda ls: [float(l[30:38]) for l in ls if l.startswith("ATOM")]
    assert xs(near_first) == [0.0, 3.0, 6.0] and xs(near_second) == [100.0, 103.0, 106.0]
    assert len(xs(protein_lines(p))) == 6    # without a ligand centre nothing is dropped
