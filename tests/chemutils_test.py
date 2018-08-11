import sys
from jtnn.mol_tree_nx import NXMolTree
import rdkit
from jtnn.chemutils import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1","O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2", "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3", "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1", 'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1', "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]
mol_tree = NXMolTree("C")
assert len(mol_tree.nodes) > 0

def draw():
    import rdkit.Chem as Chem
    import rdkit.Chem.Draw as Draw
    for i, s in enumerate(smiles):
        Draw.MolToFile(Chem.MolFromSmiles(s), '%d.png' % i)

def tree_test():
    for s in smiles:
        s = s.split()[0]
        tree = NXMolTree(s)
        print('-------------------------------------------')
        print(s)
        for node in tree.nodes:
            print(tree.nodes[node]['smiles'], [tree.nodes[x]['smiles'] for x in tree[node]])

def decode_test():
    wrong = 0
    for tot,s in enumerate(smiles):
        s = s.split()[0]
        tree = NXMolTree(s)
        tree.recover()

        cur_mol = copy_edit_mol(tree.nodes[0]['mol'])
        global_amap = [{}] + [{} for node in tree.nodes]
        global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

        dfs_assemble_nx(tree, cur_mol, global_amap, [], 0, None)

        cur_mol = cur_mol.GetMol()
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        set_atommap(cur_mol)
        dec_smiles = Chem.MolToSmiles(cur_mol)

        gold_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s))
        if gold_smiles != dec_smiles:
            print(gold_smiles, dec_smiles)
            wrong += 1
        print(wrong, tot + 1)

def enum_test():
    for s in smiles:
        s = s.split()[0]
        tree = NXMolTree(s)
        tree.recover()
        tree.assemble()
        for i in tree.nodes:
            node = tree.nodes[i]
            if node['label'] not in node['cands']:
                print(tree.smiles)
                print(node['smiles'], [tree.nodes[x]['smiles'] for x in tree[i]])
                print(node['label'], len(node['cands']))

def count():
    cnt,n = 0,0
    for s in smiles:
        s = s.split()[0]
        tree = NXMolTree(s)
        tree.recover()
        tree.assemble()
        for i in tree.nodes:
            cnt += len(tree.nodes[i]['cands'])
        n += len(tree.nodes)
        print(cnt * 1.0 / n)

draw()
tree_test()
decode_test()
enum_test()
count()
