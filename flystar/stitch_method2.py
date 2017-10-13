from flystar import starlists,plots,match,align,analysis, transforms
import numpy as np
from astropy.table import vstack, Table
import pandas as pd

def align_starlists(starlist, ref, transModel=transforms.PolyTransform, order=2, N_loop=2,
          dr_tol=1.0, briteN=None, weights='both'):
    """
    Transforming a starlist(label.dat) into a reference frame.

    Parameters:
    -----------
    starlist: Table
        Starlist we would like to transform into the reference frame, eg:label.dat

    ref: Table
        Starlist that defines the reference frame.

    transModel: transformation class (default: transforms.polyTransform)
        Defines which transformation model to use. Both the four-parameter and
        polynomial transformations are supported

    order: int (default=1)
        Order of the polynomial transformation. Only used for polynomial transform

    N_loop: int (default=2)
        How many times to iterate on the transformation calculation. Ideally,
        each iteration adds more stars and thus a better transform, to some
        limit.

    dr_tol: float(default=1.0)
        the distance tolerance for matching two stars in align.transform_and_match

    briteN: int (default=100)
        the number of stars used in blind matching

    weights: string (default='both')
        if weights=='both', we use both position error in transformed starlist and
           reference starlist as uncertanty. And weights is the reciprocal of this uncertanty.
        if weights=='starlist', we only use postion error in transformed starlist.
        if weights=='reference', we only use position error in reference starlist.
        if weights==None, we don't use weights.

    """
    
    #--------------------------------------------------
    # Initial transformation with brightest briteN stars
    #--------------------------------------------------
    # define stars that are used to calculate initial transformation.
    #pdb.set_trace()
    idx_ini_ref, idx_ini_starlist, briteN_ini = starlists.restrict_by_name(ref, starlist)

    # Perform a blind trangle-matching of the brightest briteN stars
    # and calculate initial transform
    ref_ini = ref[idx_ini_ref]
    starlist_ini = starlist[idx_ini_starlist]

    if briteN == None:
        briteN = briteN_ini
    #trans = align.initial_align(starlist_ini, ref_ini, briteN=briteN,
    #        transformModel=transModel, order=order)
    #pdb.set_trace()
    trans = transModel(starlist_ini['x'], starlist_ini['y'], ref_ini['x'], ref_ini['y'],order=order, weights=None)

    # apply the initial transform to label.dat
    # this is used for future weights calculation
    starlist_trans_ini = align.transform_from_object(starlist, trans)

    #------------------------------------------------------------------------
    # Use transformation to match starlists, then recalculate transformation.
    #------------------------------------------------------------------------
    # Iterate on this as many times as desired
    for i in range(N_loop):
        # apply the transformation to label.dat and
        # matched the transformed label with starlist.
        idx_starlist, idx_ref = align.transform_and_match(starlist, ref, trans, dr_tol=dr_tol)

        # use the matched stars to calculate new transformation
        ref_match = ref[idx_ref]
        starlist_match = starlist[idx_starlist]
        starlist_ini_match = starlist_trans_ini[idx_starlist]

        trans, N_trans = align.find_transform(starlist_match, starlist_ini_match, ref_match,
                            transModel=transModel, order=order, weights = weights)


    #---------------------------------------------
    # Write final transform in java align format
    #---------------------------------------------
    delta_m = np.mean(ref_match['m'] - starlist_match['m'])

    starlist_trans = align.transform_from_object(starlist, trans)
    starlist_trans_match = starlist_trans[idx_starlist]
    return idx_starlist, idx_ref, starlist_trans, trans

def weighted_mean(df,x,xe,frames_in_use):
	# value = x or y
	# error = xe or ye
	# all_frames = e.g. ['A', 'B', 'C', ...]

	

	cols_x=["{0}_{1}".format(x,f) for f in frames_in_use] # columns for x_* e.g. ['x_A', 'x_B', 'x_C', ....]
	cols_xe=["{0}_{1}".format(xe,f) for f in frames_in_use] # columns for xe_* e.g. ['xe_A', 'xe_B', 'xe_C', ....]
	array_x=np.array(df[cols_x]) # array that contains x_A, x_B, ...
	array_xe=np.array(df[cols_xe]) # array that contains xe_A, xe_B, ...
	array_w=1./np.array(df[cols_xe])**2. # array that contains w_A, w_B, ...
	x_master=[]
	xe_master=[]
	rows_to_drop=[]
	for i in range(len(array_x)):
		mask=~np.isnan(array_x[i])
		if mask.sum()>1: # i.e. for stars that have at least one match, calculate weighted mean values and standard deviations.
			x_hat=(array_w[i][mask]*array_x[i][mask]).sum()/(array_w[i][mask]).sum() # x_hat=sum(w*x)/sum(w)
			sigma_x=np.sqrt((array_w[i][mask]*(array_x[i][mask]-x_hat)**2.).sum()/array_w[i][mask].sum()) # sigma=sqrt( sum(w*(x-x_hat)^2)/sum(w) )
			x_master.append(x_hat)
			xe_master.append(sigma_x)
		elif mask.sum()==1:	# i.e. for stars that have no match, just append their original values and errors.
			x_master.append(array_x[i][mask][0])
			xe_master.append(array_xe[i][mask][0])
		else:
			rows_to_drop.append(i)
			
	df=df.drop(rows_to_drop)
	df[x]=np.array(x_master)
	df[xe]=np.array(xe_master)
	

	return df

def normal_mean(df,x,frames_in_use):

	cols_x=["{0}_{1}".format(x,f) for f in frames_in_use]
	df[x]=df[cols_x].mean(axis=1)
	
	return df


def stitch(all_starlists, name_initial_ref, N_iter=5, corr_thresh=0.8,  outMaster='./master.lis'):
	
	# all_starslist: the list of the names of all starlists e.g. ['A', 'B', 'C', ... ]
	# name_initial_ref: the name of the reference that you use in the very first match.
	# corr_thresh : threshold for correlation values.
	# N_iter: number of iterations.
	# outMaster: the name/location of master.lis

	input_starslists=(all_starlists*N_iter)
	if name_initial_ref in input_starslists:
		input_starslists.remove(name_initial_ref)

	for name_starlist in input_starslists:
		
		starlist=starlists.read_starlist('{0}.lis'.format(name_starlist))
		if 'ref' not in locals():
			ref=starlists.read_starlist('{0}.lis'.format(name_initial_ref))
			

		#------------ Choose good stars to use for a trans object --------------------

		starlist_for_align=starlist[(starlist['corr']>corr_thresh)]
		ref_for_align=ref[(ref['corr']>corr_thresh)]

		# Select the very first 11 columns (i.e. the master reference) consistent with those of the starlist.
		# Table -> dataframe -> Table, which lets us avoid the following error: 'MaskedColumn' object has no attribute '_mask'
		
		ref_for_align=ref_for_align.to_pandas()
	
		ref_for_align=Table.from_pandas(ref_for_align[starlist_for_align.colnames])

		_,_,_,trans=align_starlists(starlist_for_align,ref_for_align,order=2,dr_tol=1,N_loop=15)

	
		#------------ Transform the whole starlist using the trans object and match with the reference -------------

		starlist_transformed=align.transform_from_object(starlist,trans)
		idx_starlist_transformed_matched, idx_ref_matched, dr, dm=match.match(starlist_transformed['x'],starlist_transformed['y'],starlist_transformed['m'],ref['x'],ref['y'],ref['m'],dr_tol=1)

		#------------ Find the indices of unmatched stars ------------------------

		idx_starlist_transformed_unmatched=[x for x in range(len(starlist)) if x not in idx_starlist_transformed_matched]
		idx_ref_unmatched=[x for x in range(len(ref)) if x not in idx_ref_matched]

		#-------------Convert the astropy talbes into dataframes ---------------------
		df_ref=ref.to_pandas()
		df_starlist_transformed=starlist_transformed.to_pandas()
	
		#-------------Columns 11-21 contain the measurments for the initial reference--------------

		colnames=starlist.colnames
		if name_initial_ref: # i.e., if this is the first match.
			for col in colnames:
				df_ref.insert(len(df_ref.columns),'{0}_{1}'.format(col,name_initial_ref),np.nan)
				df_ref.loc[:,'{0}_{1}'.format(col,name_initial_ref)]= np.array(df_ref.loc[:,col])
			name_initial_ref=False # i.e. the first match is done.

		#-------------- Add a column for the transformed starlist and insert its values to the reference ------------

		if 'x_{0}'.format(name_starlist) in ref.colnames:
			for col in colnames:
				df_ref['{0}_{1}'.format(col,name_starlist)]=np.nan
				df_ref.loc[idx_ref_matched,'{0}_{1}'.format(col,name_starlist)]= np.array(df_starlist_transformed.loc[idx_starlist_transformed_matched,col])
	
		else:

			for col in colnames:
		
				df_ref.insert(len(df_ref.columns),'{0}_{1}'.format(col,name_starlist),np.nan)
				df_ref.loc[idx_ref_matched,'{0}_{1}'.format(col,name_starlist)]= np.array(df_starlist_transformed.loc[idx_starlist_transformed_matched,col])

			#-------------- Append the unmatched stars in the starlist to the reference------------

		columns=df_ref.columns
		df_starlist_transformed.columns=["{0}_{1}".format(x,name_starlist) for x in colnames]

		df_comb=pd.concat([df_ref,df_starlist_transformed.loc[idx_starlist_transformed_unmatched]],ignore_index=True)
		df_comb=df_comb[columns]

		#-------------- Figure out which frames are currently included in the master frame -----------

		frames_in_use=sorted(set([column[-1] for column in columns if (column[-1] in all_starlists)]))
		
		#-------------- Average the measurements -------------
		for col in colnames:
			if (col!='name') and (col!='x') and (col!='y') and (col!='xe') and (col!='ye'):
				df_comb=normal_mean(df_comb,col,frames_in_use)

		df_comb=weighted_mean(df_comb,'x','xe',frames_in_use)
		df_comb=weighted_mean(df_comb,'y','ye',frames_in_use)

		#-------------- Deal with the names of stars----------
		Nanstar=df_comb['name'].isnull()
		df_comb.loc[Nanstar,'name']=df_comb.loc[Nanstar,'name_{0}'.format(name_starlist)].copy()
		star=df_comb['name'].str.startswith('star').copy()
		twostar=df_comb['name'].str.startswith('2star').copy()
		df_comb.loc[star,'name']=['star_'+str(x) for x in df_comb[star].index]
		df_comb.loc[twostar,'name']=['2star_'+str(x) for x in df_comb[twostar].index]

		#-------------- Convert the final dataframe back into an astropy table ------

		ref=Table.from_pandas(df_comb)
		
	ref.write(outMaster,format='ascii.commented_header', header_start=-1, overwrite=True)

	return
