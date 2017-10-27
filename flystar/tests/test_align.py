from flystar import align
from flystar import starlists
from flystar import startables
import pdb

def test_mosaic_lists():
    """
    Cross-match and align 4 starlists.
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    star_table, trans_table = mosaic_lists(list_files, ref_index=0, transform=transforms.PolyTransform, transform_args={'order': 2})

    return
    
    
