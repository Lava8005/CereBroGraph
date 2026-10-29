"""Production docking oracle for the surrogate loop: real AutoDock Vina, parallel and cached on disk.

Every docked molecule is appended to a cache CSV immediately, so an interrupted run (Colab
disconnect, laptop sleep) resumes without re-docking anything.
"""
import csv
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

import numpy as np

from physics_engine.dock_vina import (DEFAULT_EXHAUSTIVENESS, DEFAULT_TIMEOUT_S, dock_precomputed_target,
                                      prepare_ligand_pdbqt)


class VinaOracle:
    def __init__(self, target_index: Path, cache_csv: Path, vina_exe: str = "vina", max_workers: int = 2,
                 exhaustiveness: int = DEFAULT_EXHAUSTIVENESS, timeout: int = DEFAULT_TIMEOUT_S,
                 work_dir: Path = Path("data/physics_engine/tmp_ligands")):
        self.targets = json.loads(Path(target_index).read_text())
        self.names = [t.get("disease", "Unknown") for t in self.targets]
        self.cache_csv, self.vina_exe, self.workers = Path(cache_csv), vina_exe, max_workers
        self.exh, self.timeout, self.work_dir = exhaustiveness, timeout, Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.cache = {}
        if self.cache_csv.exists():
            with open(self.cache_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.cache[row["smiles"]] = [float(row[n]) if row[n] not in ("", "nan") else np.nan for n in self.names]
        else:
            with open(self.cache_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["smiles"] + self.names)
        self.docking_calls = 0

    def _dock_one(self, smiles: str) -> List[float]:
        pdbqt = prepare_ligand_pdbqt(smiles)
        if not pdbqt:
            return [np.nan] * len(self.targets)
        fd, path = tempfile.mkstemp(suffix=".pdbqt", dir=self.work_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(pdbqt)
            scores = []
            for t in self.targets:
                _, s = dock_precomputed_target(self.vina_exe, path, t, self.exh, self.timeout)
                scores.append(s if s < 0 else np.nan)  # 0.0 means failure
            return scores
        finally:
            if os.path.exists(path):
                os.remove(path)

    def __call__(self, smiles: Sequence[str]) -> np.ndarray:
        todo = [s for s in dict.fromkeys(smiles) if s not in self.cache]
        if todo:
            def work(s):
                sc = self._dock_one(s)
                with self._lock:
                    self.cache[s] = sc
                    self.docking_calls += 1
                    with open(self.cache_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([s] + ["" if np.isnan(x) else x for x in sc])
                return sc
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(work, todo))
            logging.info(f"Docked {len(todo)} new molecules ({len(self.cache)} cached in total).")
        return np.array([self.cache[s] for s in smiles], dtype=float)
