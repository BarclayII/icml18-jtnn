import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from .nnutils import *
from .chemutils import get_mol
from networkx import Graph, DiGraph, line_graph, convert_node_labels_to_integers
from dgl import DGLGraph, line_graph

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

def mol2dgl_ideal(smiles_batch):
    '''
    Get a batched DGLGraph object from the given batch of SMILES
    '''
    graphs = []
    for smiles in smiles_batch:
        mol = get_mol(smiles)
        graph = Graph()
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), features=atom_features(atom))
        for bond in mol.GetBonds():
            graph.add_edge(
                    bond.GetBeginAtom().GetIdx(),
                    bond.GetEndAtom().GetIdx(),
                    features=bond_features(bond),
                    )
        graphs.append(graph)

    graph = DiGraph(nx.disjoint_union_all(graph))
    # A small caveat:
    # When an undirected graph is converted to a directed graph, the edge
    # contents are duplicated by reference:
    # >>> G = Graph()
    # >>> G.add_nodes_from([0, 1])
    # >>> G.add_edge(0, 1, l=[1,2,3,4,5])
    # >>> H = DiGraph(G)
    # >>> H[0][1]['l'].append(6)    # H[0][1]['l'] becomes [1,2,3,4,5,6]
    # >>> H[1][0]['l']
    # [1, 2, 3, 4, 5, 6]
    #
    # That essentially means, if I converted an undirected graph into a
    # (directed) DGLGraph with preset edge tensors, I would like to have
    # G[u][v] and G[v][u] share the preset ones (and only those ones).
    # In this example, I think we are safe, because the edge features are
    # inputs (hence not changed/replaced throughout the whole computation).

    # create a line graph to do loopy belief propagation
    lgraph = line_graph(graph)
    lgraph_edges = list(lgraph.edges)
    for e in lgraph_edges:
        (u1, v1), (u2, v2) = e
        if u1 == v2 and u2 == v1:
            lgraph.remove_edge(e)
    for u, v in lgraph.nodes:
        lgraph.nodes[u, v]['node_features'] = graph.nodes[u]['features']
        lgraph.nodes[u, v]['edge_features'] = graph[u][v]['features']
    lgraph = convert_node_labels_to_integers(lgraph, label_attribute='edge')
    return DGLGraph(graph), DGLGraph(lgraph)

def mol2dgl(smiles_batch):
    graph = DGLGraph()
    atom_feature_list = []
    bond_feature_list = []
    bond_source_feature_list = []
    n_nodes = 0
    for smiles in smiles_batch:
        mol = get_mol(smiles)
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx() + n_nodes)
            atom_feature_list.append(atom_features(atom))
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx() + n_nodes
            end_idx = bond.GetEndAtom().GetIdx() + n_nodes
            features = bond_features(bond)
            graph.add_edge(begin_idx, end_idx)
            bond_feature_list.append(features)
            bond_source_feature_list.append(atom_feature_list[begin_idx])
            # set up the reverse direction
            graph.add_edge(end_idx, begin_idx)
            bond_feature_list.append(features)
            bond_source_feature_list.append(atom_feature_list[end_idx])

        n_nodes += mol.GetNumAtoms()

    graph.set_n_repr({'features': torch.stack(atom_feature_list)})
    graph.set_e_repr(
            {'features': torch.stack(bond_feature_list),
             'source_features': torch.stack(bond_source_feature_list)},
            )

    lgraph = line_graph(graph)
    for (u1, v1), (u2, v2) in lgraph.edge_list:
        if u1 == v2 and u2 == v1:
            lgraph.remove_edge((u1, v1), (u2, v2))

    return graph, lgraph


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

        binput = self.W_i(fbonds)
        message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = nn.ReLU()(binput + nei_message)

        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs

