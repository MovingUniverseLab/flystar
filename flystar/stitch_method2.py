from flystar import examples,starlists,plots,match,align
import matplotlib.pylab as plt
import numpy as np
from astropy.table import vstack, Table
import pandas as pd


def weighted_mean(df,x,xe,all_frames):
	# value = x or y
	# error = xe or ye
	# all_frames = e.g. ['A', 'B', 'C', ...]

	cols_x=["{0}_{1}".format(x,f) for f in all_frames] # columns for x_* e.g. ['x_A', 'x_B', 'x_C', ....]
	cols_xe=["{0}_{1}".format(xe,f) for f in all_frames] # columns for xe_* e.g. ['xe_A', 'xe_B', 'xe_C', ....]
	array_x=np.array(df[cols_x]) # array that contains x_A, x_B, ...
	array_w=1./np.array(df[cols_xe])**2. # array that contains w_A, w_B, ...
	x_master=[]
	xe_master=[]
	for i in range(len(array_x)):
		mask=~np.isnan(array_x[i])
		if mask.sum()>1: # i.e. for stars that have at least one match, calculate weighted mean values and standard deviations.
			x_hat=(array_w[i][mask]*array_x[i][mask]).sum()/(array_w[i][mask]).sum() # x_hat=sum(w*x)/sum(w)
			sigma_x=np.sqrt((array_w[i][mask]*(array_x[i][mask]-x_hat)**2.).sum()/array_w[i][mask].sum()) # sigma=sqrt( sum(w*(x-x_hat)^2)/sum(w) )
			x_master.append(x_hat)
			xe_master.append(sigma_x)
		else:	# i.e. for stars that have no match, just append their original values and errors.
			x_master.append(array_x[i][mask][0])
			xe_master.append(array_e[i][mask][0])
	
	df[x]=np.array(x_master)
	df[xe]=np.array(xe_master)

	return df

def normal_mean(df,x,all_frames):

	cols_x=["{0}_{1}".format(x,f) for f in all_frames]
	df[x]=df[cols_x].mean(axis=1)
	
	return df


def stitch(name_starlist,name_ref, corr_thresh_starlist=0.8, corr_thresh_ref=0.8):

	# name_starslist: the name of starlist
	# name_ref : the name of reference, either the initial reference or master reference.
	# name_initial_ref: the name of the reference that you use in the very first match.
	# corr_thresh : threshold for correlation values.
	# x/y_min/max: in order to remove false positives at the boundary.

	starlist=starlists.read_starlist('{0}.lis'.format(name_starlist))
	ref=starlists.read_starlist('{0}.lis'.format(name_ref))

	#------------ Choose good stars to use for a trans object --------------------

	starlist_for_align=starlist[(starlist['corr']>corr_thresh_starlist)]
	ref_for_align=ref[(ref['corr']>corr_thresh_ref)]

	# Select the very first 11 columns (i.e. the master reference) consistent with those of the starlist.
	ref_for_align=ref_for_align[:][starlist_for_align.colnames] 
	# Table -> dataframe -> Table, which lets us avoid the following error: 'MaskedColumn' object has no attribute '_mask'
	ref_for_align=Table.from_pandas(ref_for_align.to_pandas()) 					  

	_,_,_,trans=examples.align_starlists(starlist_for_align,ref_for_align,order=2,dr_tol=1,N_loop=15,outFile='/u/dkim/Documents/outTrans.txt')

	#------------ Transform the whole starlist using the trans object and match with the reference -------------

	starlist_transformed=align.transform_from_object(starlist,trans)
	idx_starlist_transformed_matched, idx_ref_matched, dr, dm=match.match(starlist_transformed['x'],starlist_transformed['y'],starlist_transformed['m'],ref['x'],ref['y'],ref['m'],dr_tol=1)

	#------------ Find the indices of unmatched stars ------------------------

	idx_starlist_transformed_unmatched=[x for x in range(len(starlist)) if x not in idx_starlist_transformed_matched]
	idx_ref_unmatched=[x for x in range(len(ref)) if x not in idx_ref_matched]

	#-------------Convet the astropy talbes into dataframes ---------------------
	df_ref=ref.to_pandas()
	df_starlist_transformed=starlist_transformed.to_pandas()
	#df_ref.dropna(axis=0, how="any", inplace=True) # In case that you want to use only stars in common among the dithered exposures.

	#-------------Columns 11-21 contain the measurments for the initial reference--------------

	colnames=starlist.colnames
	if name_ref==name_initial_ref:
		for col in colnames:
			df_ref.insert(len(df_ref.columns),'{0}_{1}'.format(col,name_ref),np.nan)
			df_ref.loc[:,'{0}_{1}'.format(col,name_ref)]= np.array(df_ref.loc[:,col])

	#-------------- Add a column for the transformed starlist and insert its values to the reference ------------ 

	for col in colnames:
		df_ref.insert(len(df_ref.columns),'{0}_{1}'.format(col,name_starlist),np.nan)
		df_ref.loc[idx_ref_matched,'{0}_{1}'.format(col,name_starlist)]= np.array(df_starlist_transformed.loc[idx_starlist_transformed_matched,col])

	#-------------- Append the unmatched stars in the starlist to the reference------------

	columns=df_ref.columns
	df_starlist_transformed.columns=["{0}_{1}".format(x,name_starlist) for x in colnames]

	df_comb=pd.concat([df_ref,df_starlist_transformed.loc[idx_starlist_transformed_unmatched]],ignore_index=True)
	df_comb=df_comb[columns]

	#-------------- Name the new reference frame -----------

	name_new_ref=name_ref+name_starlist # e.g. 'A' + 'B' = 'AB'

	#-------------- Average the measurements -------------
	for col in colnames:
		if (col!='name') and (col!='x') and (col!='y') and (col!='xe') and (col!='ye'):
			df_comb=normal_mean(df_comb,col,name_new_ref)

	df_comb=weighted_mean(df_comb,'x','xe',name_new_ref)
	df_comb=weighted_mean(df_comb,'y','ye',name_new_ref)

	#-------------- Deal with the names of stars----------
	Nanstar=df_comb['name'].isnull()
	df_comb.loc[Nanstar,'name']=df_comb.loc[Nanstar,'name_{0}'.format(name_starlist)].copy()
	star=df_comb['name'].str.startswith('star').copy()
	twostar=df_comb['name'].str.startswith('2star').copy()
	df_comb.loc[star,'name']=['star_'+str(x) for x in df_comb[star].index]
	df_comb.loc[twostar,'name']=['2star_'+str(x) for x in df_comb[twostar].index]

	#-------------- Convert the final dataframe back into an astropy table ------
	new_ref=Table.from_pandas(df_comb)
	new_ref.write('{0}.lis'.format(name_new_ref),format='ascii.commented_header', header_start=-1, overwrite=True)

	#print new_ref

	return name_new_ref


#------test-----------


name_new_ref='B'
Names_all=['A','B','C']

Names_all.remove(name_new_ref)

for name_starlist in Names_all:

	name_new_ref=stitch(name_starlist,name_new_ref)
