import torch
import torch.nn as nn
from collections import deque
from .mol_tree import Vocab, MolTree
from .nnutils import create_var, GRU
import itertools
import networkx as nx
from dgl import batch, unbatch

MAX_NB = 8

class JTNNEncoder(nn.Module):

    def __init__(self, vocab, hidden_size, embedding=None):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        
        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, root_batch):
        orders = []
        for root in root_batch:
            order = get_prop_order(root)
            orders.append(order)
        
        h = {}
        max_depth = max([len(x) for x in orders])
        padding = create_var(torch.zeros(self.hidden_size), False)

        for t in range(max_depth):
            prop_list = []
            for order in orders:
                if t < len(order):
                    prop_list.extend(order[t])

            cur_x = []
            cur_h_nei = []
            for node_x,node_y in prop_list:
                x,y = node_x.idx,node_y.idx
                cur_x.append(node_x.wid)

                h_nei = []
                for node_z in node_x.neighbors:
                    z = node_z.idx
                    if z == y: continue
                    h_nei.append(h[(z,x)])

                pad_len = MAX_NB - len(h_nei)
                h_nei.extend([padding] * pad_len)
                cur_h_nei.extend(h_nei)

            cur_x = create_var(torch.LongTensor(cur_x))
            cur_x = self.embedding(cur_x)
            cur_h_nei = torch.cat(cur_h_nei, dim=0).view(-1,MAX_NB,self.hidden_size)

            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
            for i,m in enumerate(prop_list):
                x,y = m[0].idx,m[1].idx
                h[(x,y)] = new_h[i]

        root_vecs = node_aggregate(root_batch, h, self.embedding, self.W)

        return h, root_vecs

"""
Helper functions
"""

def get_prop_order(root):
    queue = deque([root])
    visited = set([root.idx])
    root.depth = 0
    order1,order2 = [],[]
    while len(queue) > 0:
        x = queue.popleft()
        for y in x.neighbors:
            if y.idx not in visited:
                queue.append(y)
                visited.add(y.idx)
                y.depth = x.depth + 1
                if y.depth > len(order1):
                    order1.append([])
                    order2.append([])
                order1[y.depth-1].append( (x,y) )
                order2[y.depth-1].append( (y,x) )
    order = order2[::-1] + order1
    return order

def node_aggregate(nodes, h, embedding, W):
    x_idx = []
    h_nei = []
    hidden_size = embedding.embedding_dim
    padding = create_var(torch.zeros(hidden_size), False)

    for node_x in nodes:
        x_idx.append(node_x.wid)
        nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
        pad_len = MAX_NB - len(nei)
        nei.extend([padding] * pad_len)
        h_nei.extend(nei)
    
    h_nei = torch.cat(h_nei, dim=0).view(-1,MAX_NB,hidden_size)
    sum_h_nei = h_nei.sum(dim=1)
    x_vec = create_var(torch.LongTensor(x_idx))
    x_vec = embedding(x_vec)
    node_vec = torch.cat([x_vec, sum_h_nei], dim=1)
    return nn.ReLU()(W(node_vec))


class DGLJTNNEncoderMessageModule(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, dst, edge):
        z = torch.sigmoid(self.W_z(torch.cat([src['x'], src['s']], 1)))
        m = torch.tanh(self.W_h(torch.cat([src['x'], src['rm']], 1)))
        m = (1 - z) * src['s'] + z * m
        r_1 = self.W_r(dst['x'])
        r_2 = self.U_r(m)
        r = torch.sigmoid(r_1 + r_2)
        return {'z': z, 'm': m, 'r': r}


def enc_tree_msg(src, edge):
    return {'z': edge['z'], 'm': edge['m'], 'r': edge['r']}


def enc_tree_reduce(node, msgs):
    s = msgs['m'].sum(1)
    rm = (msgs['m'] * msgs['r']).sum(1)
    return {'s': s, 'rm': rm}


def enc_tree_update(node, accum):
    return {'s': accum['s'], 'rm': accum['rm']}


def enc_tree_final_msg(src, edge):
    return {'m': edge['m']}


def enc_tree_final_reduce(node, msgs):
    return {'m': msgs['m'].sum(1)}


class DGLJTNNEncoderFinalUpdateModule(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.W = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, node, accum):
        x = torch.cat([node['x'], accum['m']], 1)
        return {'h': torch.relu(self.W(x))}


class DGLJTNNEncoder(nn.Module):
    def __init__(self, vocab, hidden_size, embedding=None):
        super(DGLJTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab

        if embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        else:
            self.embedding = embedding

        self.enc_tree_msg = DGLJTNNEncoderMessageModule(self.hidden_size)
        self.enc_tree_final_update = \
                DGLJTNNEncoderFinalUpdateModule(self.hidden_size)

    def forward(self, mol_trees):
        mol_tree_batch = batch(mol_trees)
        # root nodes are already 0 for each subgraph, so we can directly
        # take the node_offset attribute to get the root node IDs
        root_ids = mol_tree_batch.node_offset[:-1]
        n_nodes = len(mol_tree_batch.nodes)
        n_edges = len(mol_tree_batch.edges)

        wid = mol_tree_batch.get_n_repr()['wid']
        x = self.embedding(wid)
        mol_tree_batch.set_n_repr({'x': x})

        # bottom-up phase
        mol_tree_batch.set_n_repr({
            's': torch.zeros(n_nodes, self.hidden_size),
            'm': torch.zeros(n_nodes, self.hidden_size),
            'r': torch.zeros(n_nodes, self.hidden_size),
            'rm': torch.zeros(n_nodes, self.hidden_size),
            'h': torch.zeros(n_nodes, self.hidden_size),
            })
        mol_tree_batch.set_e_repr({
            'z': torch.zeros(n_edges, self.hidden_size),
            'm': torch.zeros(n_edges, self.hidden_size),
            'r': torch.zeros(n_edges, self.hidden_size),
            })

        for u, v in level_order(mol_tree_batch, root_ids, reverse=True):
            mol_tree_batch.update_edge(u, v, self.enc_tree_msg, batchable=True)
            mol_tree_batch.update_by_edge(
                    u, v, enc_tree_msg, enc_tree_reduce, enc_tree_update,
                    batchable=True)

        # move the message field and reinitialize everything and perform
        # top-down phase
        mol_tree_batch.set_e_repr({
            'm1': mol_tree_batch.get_e_repr()['m'],
            'z': torch.zeros(n_edges, self.hidden_size),
            'm': torch.zeros(n_edges, self.hidden_size),
            'r': torch.zeros(n_edges, self.hidden_size),
            })
        mol_tree_batch.set_n_repr({
            's': torch.zeros(n_nodes, self.hidden_size),
            'm': torch.zeros(n_nodes, self.hidden_size),
            'r': torch.zeros(n_nodes, self.hidden_size),
            'rm': torch.zeros(n_nodes, self.hidden_size),
            'h': torch.zeros(n_nodes, self.hidden_size),
            })

        for u, v in level_order(mol_tree_batch, root_ids, reverse=False):
            mol_tree_batch.update_edge(u, v, self.enc_tree_msg, batchable=True)
            mol_tree_batch.update_by_edge(
                    u, v, enc_tree_msg, enc_tree_reduce, enc_tree_update,
                    batchable=True)

        # Combine messages from both phases.  Since the edges in two phases
        # are disjoint, we can simply add them together.
        mol_tree_batch.set_e_repr({
            'm': mol_tree_batch.get_e_repr()['m'] + mol_tree_batch.get_e_repr()['m1']
            })

        # compute tree vector by aggregating on root node
        mol_tree_batch.update_to(
                root_ids, enc_tree_final_msg, enc_tree_final_reduce,
                self.enc_tree_final_update, batchable=True)

        msgs = {e: m for e, m in zip(
            mol_tree_batch.edge_list, mol_tree_batch.get_e_repr()['m'])}

        return msgs, mol_tree_batch.get_n_repr(root_ids)['h']


def level_order(forest, roots, reverse=False):
    '''
    Given the forest and the list of root nodes,
    returns iterator of list of edges ordered by depth
    '''
    edge_list = []
    node_depth = {}

    edge_list.append([])

    for root in roots:
        node_depth[root] = 0
        for u, v in nx.bfs_edges(forest, root):
            node_depth[v] = node_depth[u] + 1
            if len(edge_list) == node_depth[u]:
                edge_list.append([])
            edge_list[node_depth[u]].append((u, v))

    if reverse:
        for edges in reversed(edge_list):
            u, v = zip(*edges)
            yield v, u
    else:
        for edges in edge_list:
            u, v = zip(*edges)
            yield u, v
