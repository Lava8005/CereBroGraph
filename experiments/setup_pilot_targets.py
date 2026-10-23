"""Prepare the two receptors that can be obtained without access to RCSB.

The DOCKSTRING benchmark package (PyPI: dockstring) bundles prepared PDBQT receptors for
ACHE and MAOB together with box definitions. This script extracts them into
data/pilot_targets/ in the same format that target/prepare_target.py produces
(<name>_target.pdbqt + <name>_grid.json + target_index.json).

NOTE: 5-HT2A is not part of DOCKSTRING, so the pilot uses two targets only. The PDB codes
of the DOCKSTRING receptors were not verified here and may differ from 4EY7 / 2V5Z.
"""
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pilot_targets"
TARGETS = {"ACHE": "Alzheimers_AChE", "MAOB": "Parkinsons_MAOB"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", "-d", str(tmp), "dockstring"], check=True)
    wheel = next(tmp.glob("dockstring-*.whl"))
    index = []
    with zipfile.ZipFile(wheel) as z:
        for name, disease in TARGETS.items():
            pdbqt = OUT / f"{name}_target.pdbqt"
            pdbqt.write_bytes(z.read(f"dockstring/resources/targets/{name}_target.pdbqt"))
            conf = z.read(f"dockstring/resources/targets/{name}_conf.txt").decode()
            g = {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([-\d.]+)", conf)}
            grid = OUT / f"{name}_grid.json"
            grid.write_text(json.dumps(g, indent=4))
            index.append({"disease": disease, "pdb_id": name, "pdbqt": str(pdbqt.resolve()), "grid": str(grid.resolve())})
        # Vina 1.1.2 binary shipped with DOCKSTRING (Linux)
        binary = OUT / "vina"
        binary.write_bytes(z.read("dockstring/resources/bin/vina_linux"))
        binary.chmod(0o755)
    (OUT / "target_index.json").write_text(json.dumps(index, indent=4))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
