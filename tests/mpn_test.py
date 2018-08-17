from jtnn.mpn import mol2dgl

g, lg = mol2dgl(['C1=CC=CC=C1C2=CC=CC=C2', 'C1=CC=CC=C1C(=O)O'])
erepr = g.get_e_repr()['features']
lnrepr = lg.get_n_repr()['features']
assert erepr.equal(lnrepr)
