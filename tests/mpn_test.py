from jtnn.mpn import mol2dgl, DGLMPN

gl = mol2dgl(['C1=CC=CC=C1C2=CC=CC=C2', 'C1=CC=CC=C1C(=O)O', 'c1ccccc1c2ccccc2'])
mpn = DGLMPN(10, 4)
result = mpn.forward(gl)

print(result)
