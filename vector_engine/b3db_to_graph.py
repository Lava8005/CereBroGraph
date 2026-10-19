"""Molecular Graph Featurizer for CerebroGraph.

Converts SMILES strings into PyTorch Geometric Data objects.
Implements rich one-hot node/edge encodings and extracts global descriptors (TPSA, MW)
for advanced GNN processing. Bypasses Murcko scaffold calculation during inference
to maximize CPU parse-rate and feed the GPU efficiently.
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# -------------------------------------------------------------------------
# Encoding Helpers
# -------------------------------------------------------------------------

def _one_hot_encoding(value, choices):
    """Create a one-hot vector with a catch-all 'Unknown' bin at the end."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding

def _atom_featurizer(atom: Chem.Atom) -> list:
    """Extract 27-dimensional rich node features."""
    # 1. Atom Symbol (10 dims)
    symbols = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    symbol_feat = _one_hot_encoding(atom.GetSymbol(), symbols)
    
    # 2. Degree (7 dims)
    degree_feat = _one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    
    # 3. Hybridization (6 dims)
    hyb_choices = [
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2, 
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D, 
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hyb_feat = _one_hot_encoding(atom.GetHybridization(), hyb_choices)
    
    # 4. Electronic & Topological Properties (4 dims)
    charge = [atom.GetFormalCharge()]
    aromatic = [1.0 if atom.GetIsAromatic() else 0.0]
    in_ring = [1.0 if atom.IsInRing() else 0.0]
    mass = [atom.GetMass() / 100.0] # Scaled to prevent exploding gradients
    
    return symbol_feat + degree_feat + hyb_feat + charge + aromatic + in_ring + mass

def _bond_featurizer(bond: Chem.Bond) -> list:
    """Extract 7-dimensional explicit edge features."""
    # 1. Bond Type (5 dims)
    bt_choices = [
        Chem.rdchem.BondType.SINGLE, 
        Chem.rdchem.BondType.DOUBLE, 
        Chem.rdchem.BondType.TRIPLE, 
        Chem.rdchem.BondType.AROMATIC
    ]
    bt_feat = _one_hot_encoding(bond.GetBondType(), bt_choices)
    
    # 2. Topological Properties (2 dims)
    conjugation = [1.0 if bond.GetIsConjugated() else 0.0]
    ring = [1.0 if bond.IsInRing() else 0.0]
    
    return bt_feat + conjugation + ring

# Dimension exporters for downstream GNN dynamic sizing
def get_node_dim() -> int:
    return 27

def get_edge_dim() -> int:
    return 7

# -------------------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------------------

def smiles_to_graph(smiles: str, is_inference: bool = False) -> Data:
    """Convert a SMILES string to a mathematically rich PyTorch Geometric graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. Topological Bypass (Speed Optimization #4)
    # The Murcko scaffold is strictly required for dataset splitting during training.
    # Computing it during inference is wasted CPU time.
    scaffold_smiles = ""
    if not is_inference:
        try:
            scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold_smiles = ""

    # 2. Global Descriptors (BBB Prediction Boosters)
    # TPSA and MW are highly correlated with blood-brain barrier permeability
    try:
        tpsa = Descriptors.TPSA(mol) / 100.0   # Scaled
        mw = Descriptors.MolWt(mol) / 500.0    # Scaled
    except Exception:
        tpsa, mw = 0.0, 0.0
    global_features = torch.tensor([[tpsa, mw]], dtype=torch.float32)

    # 3. Node Featurization
    x_list = []
    for atom in mol.GetAtoms():
        x_list.append(_atom_featurizer(atom))
    x = torch.tensor(x_list, dtype=torch.float32)

    # 4. Edge Featurization (Bidirectional for Undirected Graph)
    edge_indices = []
    edge_attrs = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_feat = _bond_featurizer(bond)
        
        # PyG expects directed edges, so we add both directions for undirected chemical bonds
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_feat, b_feat]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        # Failsafe for single-atom molecules (e.g., salts/ions)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, get_edge_dim()), dtype=torch.float32)

    # 5. Assemble PyG Data Object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)
    data.smiles = smiles
    
    if not is_inference:
        data.scaffold = scaffold_smiles
        
    return data