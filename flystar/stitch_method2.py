from flystar import examples,starlists,plots,match,align
import matplotlib.pylab as plt
import numpy as np
from astropy.table import vstack, Table
import pandas as pd


def weighted_mean(df,value,error,all_frames):
	# value = x or y
	# error = xe or ye
	# all_frames = e.g. ['A', 'B', 'C', ...]

	col_values=["{0}_{1}".format(value,x) for x in all_frames] # e.g. ['x_A', 'x_B', 'x_C', ....]
	col_errors=["{0}_{1}".format(error,x) for x in all_frames] # e.g. ['xe_A', 'xe_B', 'xe_C', ....]
	array_values=np.array(df[col_values])
	array_errors=np.array(df[col_errors])
	means=[]
	stds=[]
	for i in range(len(array_values)):
		mask=~np.isnan(array_values[i])
		if mask.sum()>1: # i.e. for stars that have at least one match, calculate weighted mean values and standard deviations.
			mean=(array_values[i][mask]/array_errors[i][mask]**2.).sum()/(1./array_errors[i][mask]**2.).sum()
			means.append(mean)
			std=np.sqrt(((array_values[i][mask]-mean)**2./array_errors[i][mask]**2.)).sum()/(1./array_errors[i][mask]**2.).sum()
			stds.append(std)
		else:	# i.e. for stars that have no match, just append their original values and errors.
			means.append(array_values[i][mask][0])
			stds.append(array_errors[i][mask][0])
	
	df[value]=np.array(means)
	df[error]=np.array(stds)

	return df

def normal_mean(df,value,all_frames):

	col_values=["{0}_{1}".format(value,x) for x in all_frames]
	df[value]=df[col_values].mean(axis=1)
	
	return df


def stitch(name_starlist,name_ref,name_initial_ref, corr_thresh_starlist=0.8, corr_thresh_ref=0.8, xmin=10, xmax=1040, ymin=10, ymax=1040):

	# name_starslist: the name of starlist
	# name_ref : the name of reference, either the initial reference or master reference.
	# name_initial_ref: the name of the reference that you use in the very first match.
	# corr_thresh : threshold for correlation values.
	# x/y_min/max: in order to remove false positives at the boundary.

	starlist=starlists.read_starlist('mag2014_12_nirc2_BN_Mosaic_{0}_kp_rms.lis'.format(name_starlist))
	ref=starlists.read_starlist('mag2014_12_nirc2_BN_Mosaic_{0}_kp_rms.lis'.format(name_ref))

	#----------- Remove false positives at the boundary of the chip -------------

	starlist=starlist[(starlist['x']>xmin) & (starlist['x']<xmax) & (starlist['y']>ymin) & (starlist['y']<ymax) & (starlist['snr']>3)]

	# In the case of reference, this process is needed only for the initial reference. Master reference should have already passed this process.
	if name_ref==name_initial_ref:
		ref=ref[(ref['x']>xmin) & (ref['x']<xmax) & (ref['y']>ymin) & (ref['y']<ymax) & (ref['snr']>3)]

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
	new_ref.write('mag2014_12_nirc2_BN_Mosaic_{0}_kp_rms.lis'.format(name_new_ref),format='ascii.commented_header', header_start=-1, overwrite=True)

	#print new_ref

	return name_new_ref


#------test-----------

name_initial_ref='B'
name_new_ref='B'
Names_all=['A','B','C']

Names_all.remove(name_initial_ref)

for name_starlist in Names_all:

	name_new_ref=stitch(name_starlist,name_new_ref,name_initial_ref)
