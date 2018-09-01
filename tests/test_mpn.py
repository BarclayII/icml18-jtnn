from jtnn.mpn import mol2dgl, DGLMPN, MPN, mol2graph
from jtnn.jtnn_enc import JTNNEncoder, DGLJTNNEncoder
from jtnn.jtnn_dec import JTNNDecoder, DGLJTNNDecoder
from jtnn.mol_tree import MolTree, Vocab
from jtnn.mol_tree_nx import DGLMolTree
from jtnn.jtnn_vae import set_batch_nodeID, dgl_set_batch_nodeID
import torch
from torch import nn
import matplotlib.pyplot as plt
from dgl import batch

torch.manual_seed(1)

smiles_batch = ['C1=CC=CC=C1C2=CC=CC=C2', 'C1=CC=CC=C1C(=O)O', 'Cc1ccccc1C(=O)C2CCCC2']

def allclose(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-7)

def isclose(a, b):
    return torch.isclose(a, b, rtol=1e-4, atol=1e-7)


def test_mpn():
    gl = mol2dgl(smiles_batch)
    dglmpn = DGLMPN(10, 4)
    mpn = MPN(10, 4)
    mpn.W_i = dglmpn.W_i
    mpn.W_o = dglmpn.gather_updater.W_o
    mpn.W_h = dglmpn.loopy_bp_updater.W_h

    glb = batch(gl)

    result = dglmpn.forward(gl)
    mol_vec = mpn(mol2graph(smiles_batch))

    assert allclose(result, mol_vec)


def test_treeenc():
    mol_batch = [MolTree(smiles) for smiles in smiles_batch]
    for mol_tree in mol_batch:
        mol_tree.recover()
        mol_tree.assemble()

    vocab = [x.strip('\r\n ') for x in open('data/vocab.txt')]
    vocab = Vocab(vocab)

    set_batch_nodeID(mol_batch, vocab)

    emb = nn.Embedding(vocab.size(), 10)
    jtnn = JTNNEncoder(vocab, 10, emb)

    root_batch = [mol_tree.nodes[0] for mol_tree in mol_batch]
    tree_mess, tree_vec = jtnn(root_batch)

    nx_mol_batch = [DGLMolTree(smiles) for smiles in smiles_batch]
    for nx_mol_tree in nx_mol_batch:
        nx_mol_tree.recover()
        nx_mol_tree.assemble()
    dgl_set_batch_nodeID(nx_mol_batch, vocab)

    dgljtnn = DGLJTNNEncoder(vocab, 10, emb)
    dgljtnn.enc_tree_update.W_z = jtnn.W_z
    dgljtnn.enc_tree_update.W_h = jtnn.W_h
    dgljtnn.enc_tree_update.W_r = jtnn.W_r
    dgljtnn.enc_tree_update.U_r = jtnn.U_r
    dgljtnn.enc_tree_gather_update.W = jtnn.W

    # TODO: preserve edge messages
    dgl_tree_mess, dgl_tree_vec = dgljtnn(nx_mol_batch)

    assert len(dgl_tree_mess) == len(tree_mess)
    fail = False
    for e in tree_mess:
        if not allclose(tree_mess[e], dgl_tree_mess[e][0]):
            fail = True
            print(e, tree_mess[e], dgl_tree_mess[e][0])
    assert not fail

    assert allclose(dgl_tree_vec, tree_vec)


def test_treedec():
    mol_batch = [MolTree(smiles) for smiles in smiles_batch]
    for mol_tree in mol_batch:
        mol_tree.recover()
        mol_tree.assemble()

    tree_vec = torch.randn(len(mol_batch), 10)
    vocab = [x.strip('\r\n ') for x in open('data/vocab.txt')]
    vocab = Vocab(vocab)

    set_batch_nodeID(mol_batch, vocab)

    nx_mol_batch = [DGLMolTree(smiles) for smiles in smiles_batch]
    for nx_mol_tree in nx_mol_batch:
        nx_mol_tree.recover()
        nx_mol_tree.assemble()
    dgl_set_batch_nodeID(nx_mol_batch, vocab)

    emb = nn.Embedding(vocab.size(), 10)
    dgljtnn = DGLJTNNDecoder(vocab, 10, 10, emb)
    dgl_q_loss, dgl_p_loss, dgl_q_acc, dgl_p_acc = dgljtnn(nx_mol_batch, tree_vec)

    jtnn = JTNNDecoder(vocab, 10, 10, emb)
    jtnn.W = dgljtnn.W
    jtnn.U = dgljtnn.U
    jtnn.W_o = dgljtnn.W_o
    jtnn.U_s = dgljtnn.U_s
    jtnn.W_z = dgljtnn.dec_tree_edge_update.W_z
    jtnn.W_r = dgljtnn.dec_tree_edge_update.W_r
    jtnn.U_r = dgljtnn.dec_tree_edge_update.U_r
    jtnn.W_h = dgljtnn.dec_tree_edge_update.W_h
    q_loss, p_loss, q_acc, p_acc = jtnn(mol_batch, tree_vec)

    assert isclose(p_loss, dgl_p_loss)
    assert isclose(q_loss, dgl_q_loss)
    assert isclose(p_acc, dgl_p_acc)
    assert isclose(q_acc, dgl_q_acc)


if __name__ == '__main__':
    test_mpn()
    test_treeenc()
    test_treedec()
