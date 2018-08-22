from jtnn.mpn import mol2dgl, DGLMPN, MPN, mol2graph
import torch
import matplotlib.pyplot as plt
from dgl import batch

smiles_batch = ['C1=CC=CC=C1C2=CC=CC=C2', 'C1=CC=CC=C1C(=O)O', 'c1ccccc1c2ccccc2']
gl = mol2dgl(smiles_batch)
dglmpn = DGLMPN(10, 1)
mpn = MPN(10, 1)
mpn.W_i = dglmpn.W_i
mpn.W_o = dglmpn.gather_updater.W_o
mpn.W_h = dglmpn.loopy_bp_updater.W_h

glb = batch(gl)


result = dglmpn.forward(gl)
mol_vec = mpn(mol2graph(smiles_batch))

fail = False
for i, (u, v) in enumerate(dglmpn.new_edges):
    if (i + 1) not in mpn.agraph[v]:
        fail = True
        print(mpn.agraph[v], (u, v), i)
assert not fail

print(result)
print(mol_vec)
print(torch.allclose(dglmpn.message, mpn.message0[1:]))
print(torch.allclose(dglmpn.accum, mpn.atom_incoming_message))