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
        self._failed_at = {}   # smiles -> timeout (s) under which the failure was recorded
        header = ["smiles"] + self.names + ["timeout_s"]
        if self.cache_csv.exists():
            with open(self.cache_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                sc = [float(row[n]) if row[n] not in ("", "nan") else np.nan for n in self.names]
                self.cache[row["smiles"]] = sc
                self._failed_at[row["smiles"]] = float(row.get("timeout_s") or 0)
            if rows and "timeout_s" not in rows[0]:   # upgrade an old cache file in place
                with open(self.cache_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f); w.writerow(header)
                    for smi, sc in self.cache.items():
                        w.writerow([smi] + ["" if np.isnan(x) else x for x in sc] + [0])
        else:
            with open(self.cache_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)
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
        # A molecule is re-docked if it was never docked, or if a target failed under a SHORTER timeout than now.
        todo = [s for s in dict.fromkeys(smiles)
                if s not in self.cache or (np.isnan(self.cache[s]).any() and self._failed_at.get(s, 0) < self.timeout)]
        if todo:
            def work(s):
                sc = self._dock_one(s)
                with self._lock:
                    self.cache[s] = sc
                    self._failed_at[s] = self.timeout
                    self.docking_calls += 1
                    with open(self.cache_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([s] + ["" if np.isnan(x) else x for x in sc] + [self.timeout])
                return sc
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(work, todo))
            logging.info(f"Docked {len(todo)} new molecules ({len(self.cache)} cached in total).")
        return np.array([self.cache[s] for s in smiles], dtype=float)
