import numpy as np
from sets import Set
from astropy.table import Table

def restrict_by_name(table1, table2):
    """
    find stars with same name in two table.

    Input
    ------
    table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0

    table2: astropy.table
        ontains name,m,x,y,xe,ye


    Output
    ------
    table1_trim, table2_trim
    """

    name1 = table1['name']
    name2 = table2['name']
    Name = Set(name1) & Set(name2)

    idx1 = np.array([list(name1).index(i) for i in Name])
    idx2 = np.array([list(name2).index(i) for i in Name])

    return table1[idx1], table2[idx2], len(idx1)
