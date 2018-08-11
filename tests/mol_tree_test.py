
import sys
from jtnn.mol_tree_nx import NXMolTree
import rdkit
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

cset = set()
for line in sys.stdin:
    smiles = line.split()[0]
    mol = NXMolTree(smiles)
    for c in mol.nodes:
        cset.add(mol.nodes[c]['smiles'])
for x in cset:
    print(x)
