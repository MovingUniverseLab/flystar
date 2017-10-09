from flystar import stitch_method2

#Given files A.lis, B.lis, C.lis, and D.lis, let's say you chose A.lis as an initial reference frame.

name_initial_ref='A'
all_starlists=['A','B','C','D']
stitch_method2.stitch(all_starlists,name_initial_ref,N_iter=5)