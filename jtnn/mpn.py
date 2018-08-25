import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .chemutils import get_mol
from networkx import Graph, DiGraph, line_graph, convert_node_labels_to_integers
from dgl import DGLGraph, line_graph, batch, unbatch
from functools import partial

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def mol2graph(mol_batch):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    scope = []
    total_atoms = 0

    for smiles in mol_batch:
        mol = get_mol(smiles)
        #mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append( atom_features(atom) )
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds) 
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)
        
        scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b

    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2

    return fatoms, fbonds, agraph, bgraph, scope

def mol2dgl(smiles_batch):
    n_nodes = 0
    graph_list = []
    for smiles in smiles_batch:
        atom_feature_list = []
        bond_feature_list = []
        bond_source_feature_list = []
        graph = DGLGraph()
        mol = get_mol(smiles)
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx())
            atom_feature_list.append(atom_features(atom))
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            features = bond_features(bond)
            graph.add_edge(begin_idx, end_idx)
            bond_feature_list.append(features)
            bond_source_feature_list.append(atom_feature_list[begin_idx])
            # set up the reverse direction
            graph.add_edge(end_idx, begin_idx)
            bond_feature_list.append(features)
            bond_source_feature_list.append(atom_feature_list[end_idx])

        graph.set_n_repr({'features': torch.stack(atom_feature_list)})
        graph.set_e_repr(
                {'features': torch.stack(bond_feature_list),
                 'source_features': torch.stack(bond_source_feature_list)}
                )
        graph_list.append(graph)

    return graph_list


class MPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, mol_graph):
        fatoms,fbonds,agraph,bgraph,scope = mol_graph
        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)
        self.agraph = agraph

        binput = self.W_i(fbonds)
        self.binput = binput
        message = nn.ReLU()(binput)
        self.message0 = message

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            self.nei_message = nei_message
            nei_message = self.W_h(nei_message)
            message = nn.ReLU()(binput + nei_message)
            self.message = message

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        self.atom_incoming_message = nei_message
        self.fatoms = fatoms
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        self.atom_hiddens = atom_hiddens
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs


def mpn_loopy_bp_msg(src, edge):
    return src['msg']


def mpn_loopy_bp_reduce(node, msgs):
    return {'msg': torch.sum(msgs, 1)}


class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, node, accum):
        msg_input = node['msg_input']
        msg_delta = self.W_h(accum['msg']) if accum is not None else 0
        msg = F.relu(msg_input + msg_delta)
        return {'msg': msg}


def mpn_gather_msg(src, edge):
    return edge['msg']


def mpn_gather_reduce(node, msgs):
    # TODO: this looks a bit unnatural
    n_nodes = node['features'].shape[0]
    return {'msg': torch.sum(msgs, 1)}


class GatherUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, node, accum):
        m = (torch.zeros(*node['features'].shape[:-1], self.hidden_size)
             .to(node['features'])
             if accum is None
             else accum['msg'])
        return {
            'h': F.relu(self.W_o(torch.cat([node['features'], m], 1))),
            'm': m,
        }


class DGLMPN(nn.Module):
    def __init__(self, hidden_size, depth):
        super(DGLMPN, self).__init__()

        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)

        self.loopy_bp_updater = LoopyBPUpdate(hidden_size)
        self.gather_updater = GatherUpdate(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, mol_graph_list):
        mol_graph = batch(mol_graph_list)
        mol_line_graph = line_graph(mol_graph, no_backtracking=True)

        bond_features = mol_line_graph.get_n_repr()['features']
        source_features = mol_line_graph.get_n_repr()['source_features']

        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        self.msg_input = msg_input
        mol_line_graph.set_n_repr({'msg_input': msg_input})
        mol_line_graph.set_n_repr({'msg': F.relu(msg_input)})

        for i in range(self.depth - 1):
            mol_line_graph.update_all(mpn_loopy_bp_msg, mpn_loopy_bp_reduce, self.loopy_bp_updater, True)

        self.message = mol_line_graph.get_n_repr()['msg']

        mol_graph.update_all(mpn_gather_msg, mpn_gather_reduce, self.gather_updater, True)

        self.accum = mol_graph.get_n_repr()['m']
        self.edge_list = mol_graph.edge_list
        self.new_edge_list = mol_graph.new_edge_list
        self.new_edges = mol_graph.new_edges

        mol_graph_list = unbatch(mol_graph)
        g_repr = torch.stack([g.get_n_repr()['h'].mean(0) for g in mol_graph_list], 0)

        return g_repr
