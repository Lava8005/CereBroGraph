"""Keep essential cofactors (e.g. FAD in MAO-B) in the docking receptor.

Meeko's receptor preparation has no template for FAD and silently deletes it, but FAD forms part of the
MAO-B substrate cavity, so docking without it lets ligands sit where the flavin should be. We convert the
cofactor with Open Babel and append its atoms to the Meeko receptor PDBQT as rigid receptor atoms.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

RECORDS = ("ATOM", "HETATM")


def extract_cofactor_lines(pdb: Path, resnames: Iterable[str]) -> List[str]:
    """ATOM/HETATM lines of the given residue names (first altloc only, first chain copy of each residue)."""
    wanted = {r.upper() for r in resnames}
    out, chain = [], None
    for line in Path(pdb).read_text().splitlines():
        if line.startswith(RECORDS) and line[17:20].strip().upper() in wanted:
            if line[16] not in (" ", "A"):
                continue
            c = line[21]
            chain = chain or c
            if c == chain:
                out.append(line)
    return out


def _split_copies(atom_lines: Sequence[str]) -> List[List[str]]:
    """Split ATOM lines into separate protein copies: a new copy starts when residue numbering restarts (or on TER)."""
    copies, cur, last = [], [], None
    for l in atom_lines:
        if l.startswith("TER"):
            if cur:
                copies.append(cur); cur, last = [], None
            continue
        try:
            num = int(l[22:26])
        except ValueError:
            cur.append(l); continue
        if last is not None and num < last and cur:   # numbering jumped back: another copy of the chain
            copies.append(cur); cur = []
        cur.append(l); last = num
    if cur:
        copies.append(cur)
    return copies


def _copy_near(copies: List[List[str]], center, radius: float = 10.0) -> List[str]:
    """The copy with most atoms within `radius` Angstrom of `center` (the one that owns the binding site)."""
    cx, cy, cz = center
    def n_close(cp):
        c = 0
        for l in cp:
            try:
                x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
            except ValueError:
                continue
            c += ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= radius ** 2
        return c
    return max(copies, key=n_close)


def protein_lines(pdb: Path, exclude_resnames: Iterable[str] = (), ligand_center=None) -> List[str]:
    """ATOM lines (first altloc) minus the cofactors handled separately, plus TER/END records.

    Some structures (e.g. DUD-E AChE) hold two copies of the protein with the same chain and residue numbers,
    which Meeko rejects as "interrupted residues". If `ligand_center` is given, only the copy that contains the
    binding site is kept.
    """
    skip = {r.upper() for r in exclude_resnames}
    atoms = []
    for line in Path(pdb).read_text().splitlines():
        if line.startswith("ATOM") and line[16] in (" ", "A") and line[17:20].strip().upper() not in skip:
            atoms.append(line)
        elif line.startswith("TER"):
            atoms.append("TER")
    if ligand_center is not None:
        copies = _split_copies(atoms)
        if len(copies) > 1:
            atoms = _copy_near(copies, ligand_center)
    return [l for l in atoms if l != "TER"] + ["TER", "END"]


def cofactor_to_pdbqt_lines(lines: Sequence[str], obabel: str = None) -> List[str]:
    """Convert cofactor PDB lines to rigid PDBQT ATOM lines with Open Babel (adds hydrogens, Gasteiger charges)."""
    obabel = obabel or shutil.which("obabel") or shutil.which("obabel.exe")
    if obabel is None:
        raise RuntimeError("Open Babel (obabel) is required to keep cofactors such as FAD but was not found on PATH.")
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "cof.pdb", Path(td) / "cof.pdbqt"
        src.write_text("\n".join(lines) + "\nEND\n")
        r = subprocess.run([obabel, str(src), "-O", str(dst), "-h", "-xr"], capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError(f"Open Babel failed on the cofactor: {r.stderr[-800:]}")
        return [l for l in dst.read_text().splitlines() if l.startswith(RECORDS)]


def append_to_receptor(receptor_pdbqt: Path, cofactor_lines: Sequence[str]) -> int:
    """Append cofactor atoms (renumbered) to a receptor PDBQT. Returns the number of atoms appended."""
    path = Path(receptor_pdbqt)
    text = path.read_text().splitlines()
    last = 0
    for l in text:
        if l.startswith(RECORDS):
            try:
                last = max(last, int(l[6:11]))
            except ValueError:
                pass
    add = []
    for i, l in enumerate(cofactor_lines, start=1):
        add.append(l[:6] + f"{last + i:5d}" + l[11:])
    body = [l for l in text if l.strip() != "END"]
    path.write_text("\n".join(body + add) + "\n")
    return len(add)


def has_residue(pdbqt: Path, resnames: Iterable[str]) -> bool:
    wanted = {r.upper() for r in resnames}
    for l in Path(pdbqt).read_text().splitlines():
        if l.startswith(RECORDS) and l[17:20].strip().upper() in wanted:
            return True
    return False
