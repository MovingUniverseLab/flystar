from flystar import align
from flystar import starlists
from flystar import startables
from flystar import transforms
import pdb

def test_mosaic_lists():
    """
    Cross-match and align 4 starlists.
    """
    list_files = ['A.lis', 'B.lis', 'C.lis', 'D.lis']
    lists = [starlists.StarList.from_lis_file(lf) for lf in list_files]

    star_table, trans_table = align.mosaic_lists(lists, ref_index=0, iters=5,
                                                 trans_class=transforms.PolyTransform,
                                                 trans_args={'order': 2})

    return
    
    
