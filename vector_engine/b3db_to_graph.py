"""SMILES to PyTorch Geometric Graph Converter & Scaffold-Split Dataset Loader.

Translates 1D chemical string representations into 2D graph tensors and
orchestrates deterministic Bemis-Murcko scaffold-split DataLoaders for the
Blood-Brain Barrier (B3DB) dataset while quarantining FDA validation controls.
"""

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit is required. Install via: pip install rdkit")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Canonical SMILES of FDA positive/negative controls quarantined from training
# to prevent data leakage during Validation 2 (Retrospective Control Spike).
_RAW_FDA_CONTROLS = [
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",  # Donepezil (AChE +)
    "CC(Nc1ccc(OCc2ccccc2F)cc1)C(N)=O",               # Safinamide (MAO-B +)
    "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",  # Risperidone (5-HT2A +)
    "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1", # Pimavanserin (5-HT2A +)
    "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",   # Loperamide (BBB -)
    "CN(C)C(=O)C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",  # Loperamide variant in B3DB
]


def _build_canonical_control_set() -> Set[str]:
    """Canonicalize FDA control SMILES strings for exact lookup."""
    controls: Set[str] = set()
    for smi in _RAW_FDA_CONTROLS:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            controls.add(Chem.MolToSmiles(mol, canonical=True))
    return controls


FDA_CONTROL_SMILES: Set[str] = _build_canonical_control_set()


def extract_murcko_scaffold(smiles: str) -> str:
    """Extract the canonical Bemis-Murcko scaffold SMILES string for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string into a PyTorch Geometric Data object.

    Extracts 5 topological node features using modern RDKit valence APIs.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        node_features: List[List[float]] = []
        for atom in mol.GetAtoms():
            implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetTotalNumHs()),
                float(implicit_valence),
                float(atom.GetIsAromatic()),
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices: List[List[int]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        scaffold_smiles = extract_murcko_scaffold(canonical_smiles)

        data = Data(x=x, edge_index=edge_index)
        data.smiles = canonical_smiles
        data.scaffold = scaffold_smiles
        return data

    except Exception as e:
        logging.debug(f"Failed to process SMILES '{smiles}': {e}")
        return None


def scaffold_split_graphs(
    graphs: List[Data],
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[List[Data], List[Data]]:
    """Partition molecular graphs into train and validation sets by Bemis-Murcko scaffold.

    Guarantees that no two molecules sharing a core ring scaffold appear in both
    partitions, preventing congeneric series memorization.
    """
    if val_split <= 0.0 or len(graphs) <= 1:
        return graphs, []

    scaffold_groups: Dict[str, List[Data]] = defaultdict(list)
    for g in graphs:
        scaf = getattr(g, "scaffold", "")
        scaffold_groups[scaf].append(g)

    # Sort scaffold buckets deterministically, then shuffle with fixed seed
    # for balanced class representation while preserving strict scaffold disjointness.
    buckets = list(scaffold_groups.items())
    buckets.sort(key=lambda item: (len(item[1]), item[0]), reverse=True)

    rng = random.Random(random_seed)
    # Keep giant buckets (>25% of dataset) in train; deterministically shuffle the rest
    n_total = len(graphs)
    target_val_size = max(1, int(round(n_total * val_split)))

    large_buckets = [b for b in buckets if len(b[1]) > target_val_size]
    small_buckets = [b for b in buckets if len(b[1]) <= target_val_size]
    rng.shuffle(small_buckets)

    train_graphs: List[Data] = []
    val_graphs: List[Data] = []

    for _, group in large_buckets:
        train_graphs.extend(group)

    for _, group in small_buckets:
        if len(val_graphs) + len(group) <= target_val_size:
            val_graphs.extend(group)
        else:
            train_graphs.extend(group)

    # Fallback if all buckets were larger than target_val_size
    if not val_graphs and len(buckets) > 1:
        _, smallest_group = buckets[-1]
        val_graphs.extend(smallest_group)
        train_graphs = [g for g in graphs if g not in val_graphs]

    return train_graphs, val_graphs


def _resolve_bbb_label(
    raw_label: str,
    raw_logbb: Optional[str],
    logbb_cutoff: float = -0.1,
) -> float:
    """Resolve binary BBB permeability label.

    If a quantitative logBB value is present in B3DB, applies a physiological
    threshold (logBB >= -0.1) so P-gp efflux substrates with negative brain-to-blood
    partitioning (e.g., Loperamide at logBB = -0.25) are accurately labeled BBB- (0.0).
    """
    if raw_logbb is not None and str(raw_logbb).strip() != "":
        try:
            logbb_val = float(str(raw_logbb).strip())
            return 1.0 if logbb_val >= logbb_cutoff else 0.0
        except ValueError:
            pass

    if isinstance(raw_label, str):
        return 1.0 if ("BBB+" in raw_label or raw_label.strip() == "1") else 0.0
    return float(raw_label)


def get_b3db_dataloaders(
    csv_path: Path = Path("data/training/B3DB_classification.tsv"),
    batch_size: int = 128,
    val_split: float = 0.2,
    random_seed: int = 42,
    split_mode: str = "scaffold",
    exclude_controls: bool = False,
    logbb_cutoff: float = -0.1,
) -> Tuple[DataLoader, DataLoader]:
    """Parse the B3DB dataset, execute RDKit graph conversions, and yield DataLoaders.

    Args:
        csv_path: Path to the B3DB TSV or CSV file.
        batch_size: Mini-batch size for PyG DataLoader.
        val_split: Fraction of dataset reserved for validation (0.0 returns all in train).
        random_seed: Deterministic seed for splitting (recorded as 42 for Phase 2).
        split_mode: 'scaffold' (Bemis-Murcko, default) or 'stratified'.
        exclude_controls: If True, removes FDA validation controls to prevent leakage.
        logbb_cutoff: Quantitative logBB threshold when logBB column is present.
    """
    if not isinstance(csv_path, Path):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training dataset missing at {csv_path}. "
            "Please ensure the B3DB classification dataset is placed in data/training/."
        )

    logging.info(
        f"Loading B3DB dataset from {csv_path} (split_mode={split_mode}, seed={random_seed})..."
    )

    dataset_graphs: List[Data] = []
    dropped_count = 0
    quarantined_count = 0

    delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Empty or headerless dataset file: {csv_path}")

        label_col = next(
            (col for col in reader.fieldnames if col in ["BBB+/BBB-", "p_np", "label", "BBB"]),
            None,
        )
        if not label_col:
            raise ValueError(
                f"Could not identify a target label column in {csv_path}. Found: {reader.fieldnames}"
            )

        has_logbb = "logBB" in reader.fieldnames

        for row in reader:
            raw_smiles = row.get("SMILES", row.get("smiles", row.get("compound_smiles")))
            raw_label = row.get(label_col)

            if not raw_smiles or raw_label is None or str(raw_label).strip() == "":
                continue

            graph = smiles_to_graph(raw_smiles)
            if graph is None:
                dropped_count += 1
                continue

            if exclude_controls and graph.smiles in FDA_CONTROL_SMILES:
                quarantined_count += 1
                continue

            y_val = _resolve_bbb_label(
                raw_label=str(raw_label),
                raw_logbb=row.get("logBB") if has_logbb else None,
                logbb_cutoff=logbb_cutoff,
            )
            graph.y = torch.tensor([y_val], dtype=torch.float)
            dataset_graphs.append(graph)

    total_valid = len(dataset_graphs)
    if total_valid == 0:
        raise RuntimeError("Dataset processing failed: 0 valid graphs were generated.")

    logging.info(
        f"Generated {total_valid} valid graphs | Dropped {dropped_count} invalid SMILES | "
        f"Quarantined {quarantined_count} FDA controls."
    )

    if val_split <= 0.0:
        train_graphs = dataset_graphs
        val_graphs = []
    elif split_mode == "scaffold":
        train_graphs, val_graphs = scaffold_split_graphs(
            dataset_graphs, val_split=val_split, random_seed=random_seed
        )
        unique_train_scaffolds = len({g.scaffold for g in train_graphs})
        unique_val_scaffolds = len({g.scaffold for g in val_graphs})
        logging.info(
            f"Scaffold Split (seed={random_seed}): "
            f"{len(train_graphs)} Train ({unique_train_scaffolds} scaffolds) | "
            f"{len(val_graphs)} Val ({unique_val_scaffolds} scaffolds)"
        )
    else:
        labels = [g.y.item() for g in dataset_graphs]
        try:
            train_graphs, val_graphs = train_test_split(
                dataset_graphs,
                test_size=val_split,
                random_state=random_seed,
                stratify=labels,
            )
        except ValueError:
            train_graphs, val_graphs = train_test_split(
                dataset_graphs, test_size=val_split, random_state=random_seed
            )
        logging.info(f"Stratified Split: {len(train_graphs)} Train | {len(val_graphs)} Validation")

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_graphs if val_graphs else train_graphs[:1],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader