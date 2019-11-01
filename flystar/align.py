import numpy as np
from flystar import match
from flystar import transforms
from flystar import plots
from flystar.starlists import StarList
from flystar.startables import StarTable
from astropy.table import Table, Column, vstack
import datetime
import copy
import os
import pdb
import warnings
from astropy.utils.exceptions import AstropyUserWarning

# Keep a list of columns that are "aggregated" motion model terms.
motion_model_col_names = ['x0', 'x0e', 'y0', 'y0e',
                          'vx', 'vxe', 'vy', 'vye',
                          'ax', 'axe', 'ay', 'aye',
                          't0', 'use_in_trans']

class MosaicSelfRef(object):
    def __init__(self, list_of_starlists, ref_index=0, iters=2,
                 dr_tol=[1, 1], dm_tol=[2, 1],
                 outlier_tol=[None, None],
                 trans_args=[{'order': 2}, {'order': 2}],
                 mag_trans=True, mag_lim=None, weights=None,
                 trans_input=None, trans_class=transforms.PolyTransform,
                 use_vel=False, 
                 init_guess_mode='miracle', iter_callback=None,
                 verbose=True):

        """
        Required Parameters
        ----------
        list_of_starlists : array of StarList objects
            An array or list of flystar.starlists.StarList objects (which are Astropy Tables).
            There should be one for each starlist and they must contain 'x', 'y', and 'm' columns.

            Note that there is an optional weights column called 'w'. If this column exists
            in any of the lists, it will be queried to determine if an individual star can be
            used to derive the transformations between starlists. This is the most flexible way
            to allow you to determine, as a function of time and star, which ones are good enough 
            in the transformation. Note that just because it can be used (i.e. w_in=1), 
            doesn't meant that it will be used. The mag limits and outliers still take precedence. 
            Note also that the weights that go into the transformation are 

                star_list['w'] * ref_list['w'] * weight_from_keyword (see the weights parameter)

            for those stars not trimmed out by the other criteria. 


        Optional Parameters
        ----------
        ref_index : int
            The index of the reference epoch. (default = 0). Note that this is the reference
            list only for the first iteration. Subsequent iterations will utilize the sigma-clipped 
            mean of the positions from all the starlists. 

        iters : int
            The number of iterations used in the matching and transformation.  TO DO: INNER/OUTER? 

        dr_tol : list or array
            The delta-radius (dr) tolerance for matching in units of the reference coordinate system.
            This is a list of dr values, one for each iteration of matching/transformation.

        dm_tol : list or array
            The delta-magnitude (dm) tolerance for matching in units of the reference coordinate system.
            This is a list of dm values, one for each iteration of matching/transformation. 

        mag_trans : boolean
            If true, this will also calculate and (temporarily) apply a zeropoint offset to 
            magnitudes in each list to bring them into a common magnitude system. This is 
            essential for matching (with finite dm_tol) starlists of different filters or 
            starlists that are not photometrically calibrated. Note that the final_table columns 
            of 'm', 'm0', and 'm0e' will contain the transformed magnitudes while the 
            final_table column 'm_orig' will contain the original un-transformed magnitudes. 
            If mag_trans = False, then no such zeropoint offset it applied at any point. 

        mag_lim : array
            If different from None, it indicates the minimum and maximum magnitude
            on the catalogs for finding the transformations

        weights : str
            Either None (def), 'both,var', 'list,var', or 'ref,var' depending on whether you want
            to weight by the positional uncertainties (variances) in the individual starlists, or also with
            the uncertainties in the reference frame itself.  Note weighting only works when there
            are positional uncertainties availabe. Other options include 'both,std', 'list,std', 'list,var'.

        trans_input : array or list of transform objects
            def = None. If not None, then this should contain an array or list of transform
            objects that will be used as the initial guess in the alignment and matching. 

        trans_class : transforms.Transform2D object (or subclass)
            The transform class that will be used to when deriving the optimal
            transformation parameters between each list and the reference list. 

        trans_args : dictionary
            A dictionary (or a list of dictionaries) containing any extra keywords that are needed 
            in the transformation object. For instance, "order". Note that if a list is passed in, 
            then the transformation argument (i.e. order) will be changed for every iteration in
            iters.

        use_vel : boolean
            If velocities are present in the reference list and use_vel == True, then during
            each iteration of the alignment, the reference list will be propogated in time
            using the velocity information. So all transformations will be derived w.r.t. 
            the propogated positions. See also update_vel.

        iter_callback : None or function
            A function to call (that accepts a StarTable object and an iteration number)
            at the end of every iteration. This can be used for plotting or printing state. 
        """

        self.star_lists = list_of_starlists
        self.ref_index = ref_index
        self.iters = iters
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.outlier_tol = outlier_tol
        self.trans_args = trans_args
        self.mag_trans = mag_trans
        self.mag_lim = mag_lim
        self.weights = weights
        self.trans_input = trans_input
        self.trans_class = trans_class
        self.use_vel = use_vel
        self.init_guess_mode = init_guess_mode
        self.iter_callback = iter_callback
        self.verbose = verbose
              
        self.N_lists = len(self.star_lists)

        # Hard-coded values:
        self.use_ref_new = True
        self.update_ref_orig = True

        ##########
        # Error checking for parameters.
        ##########
        self.fix_iterable_conditions()  # fix dr_tol, dm_tol, outlier_tol, mag_lim to be iterable.
        check_iter_tolerances(self.iters, self.dr_tol, self.dm_tol, self.outlier_tol)
        check_trans_input(self.star_lists, self.trans_input, self.mag_trans)

        ##########
        # Setup transformation lists and arguments.
        #     - Keep a list of the transformation objects for each epochs.
        #     - Load up previous transformations, if they exist.
        #     - Keep a list of trans_args, one for each starlist. If only a single
        #       is passed in, replicate for all star lists, all loop iterations.
        ##########
        self.setup_trans_info()

        return

    def fix_iterable_conditions(self):
        if not np.iterable(self.dr_tol):
            self.dr_tol = np.repeat(self.dr_tol, self.iters)
        assert len(self.dr_tol) == self.iters

        if not np.iterable(self.dm_tol):
            self.dm_tol = np.repeat(self.dm_tol, self.iters)
        assert len(self.dm_tol) == self.iters

        if not np.iterable(self.outlier_tol):
            self.outlier_tol = np.repeat(self.outlier_tol, self.iters)
        assert len(self.outlier_tol) == self.iters

        if self.mag_lim == None:
            self.mag_lim = np.repeat([[None, None]], len(self.star_lists), axis=0)
        else:
            if (len(self.mag_lim) == 2):
                self.mag_lim = np.repeat([self.mag_lim], len(self.star_lists), axis=0)
                
        assert len(self.mag_lim) == len(self.star_lists)

        return
        
    
    def fit(self):
        """
        Using the current parameter settings, match and transform all the lists
        to a reference position. Note in the first pass, the reference position
        is just the specified input reference starlist. In subsequent iterations,
        this is updated. 

        The ultimate outcome is the creation of self.ref_table. This reference
        table will contain "averaged" quantites as well as a big 2D array of all
        the matched original and transformed quantities. 

        Averaged columns on ref_table:
        x0
        y0
        m0
        x0e
        y0e
        m0e
        vx  (only if use_vel=True)
        vy  (only if use_vel=True)
        vxe (only if use_vel=True)
        vye (only if use_vel=True)

        """
        ##########
        # Setup a reference table to store data. It will contain:
        #    x0, y0, m0 -- the running average of positions: 1D
        #    x, y, m, (opt. errors) -- the transformed positions for the lists: 2D
        #    x_orig, y_orig, m_orig, (opt. errors) -- the transformed errors for the lists: 2D
        #    w, w_orig (optiona) -- the input and output weights of stars in transform: 2D
        ##########
        self.ref_table = self.setup_ref_table_from_starlist(self.star_lists[self.ref_index])

        # Save the reference index to the meta data on the reference list.
        self.ref_table.meta['ref_list'] = self.ref_index

        ##########
        #
        # Repeat transform + match of all the starlists several times.
        #
        ##########
        for nn in range(self.iters):
            
            # If we are on subsequent iterations, remove matching results from the 
            # prior iteration. This leaves aggregated (1D) columns alone.
            if nn > 0:
                self.reset_ref_values()

            if self.verbose:
                print(" ")
                print("**********")
                print("**********")
                print('Starting iter {0:d} with ref_table shape:'.format(nn), self.ref_table['x'].shape)
                print("**********")
                print("**********")

            # ALL the action is in here. Match and transform the stack of starlists.
            # This updates trans objects and the ref_table. 
            self.match_and_transform(self.mag_lim[self.ref_index],
                                     self.dr_tol[nn], self.dm_tol[nn], self.outlier_tol[nn],
                                     self.trans_args[nn])

            # Clean up the reference table
            # Find where stars are detected.
            self.ref_table.detections()

            ### Drop all stars that have 0 detections.
            idx = np.where(self.ref_table['n_detect'] == 0)[0]
            print('  *** Getting rid of {0:d} out of {1:d} junk sources'.format(len(idx), len(self.ref_table)))
            self.ref_table.remove_rows(idx)

            # Fit velocities again
            self.update_ref_table_aggregates()

            if self.iter_callback != None:
                self.iter_callback(self.ref_table, nn)
            

        ##########
        #
        # Re-do all matching given final transformations.
        #        No trimming this time.
        #        First rest the reference table 2D values. 
        ##########
        self.reset_ref_values(exclude=['used_in_trans'])

        if self.verbose:
            print("**********")
            print("Final Matching")
            print("**********")

        self.match_lists(self.dr_tol[-1], self.dm_tol[-1])
        self.update_ref_table_aggregates()

        
        ##########
        # Clean up output table.
        # 
        ##########
        # Find where stars are detected.
        if self.verbose:
            print('')
            print('   Preparing the reference table...')
            
        self.ref_table.detections()

        ### Drop all stars that have 0 detections.
        idx = np.where(self.ref_table['n_detect'] == 0)[0]
        print('  *** Getting rid of {0:d} out of {1:d} junk sources'.format(len(idx), len(self.ref_table)))
        self.ref_table.remove_rows(idx)

        if self.iter_callback != None:
            self.iter_callback(self.ref_table, nn)
        
        return

    def match_and_transform(self, ref_mag_lim, dr_tol, dm_tol, outlier_tol, trans_args):
        """
        Given some reference list of positions, loop through all the starlists
        transform and match them.
        """
        for ii in range(len(self.star_lists)):
            if self.verbose:
                msg  = '   Matching catalog {0} / {1} with {2:d} stars'
                msg2 = '      {0:8s} < {1:0.3f}'
                print(" ")
                print("   **********")
                print(msg.format((ii + 1), len(self.star_lists), len(self.star_lists[ii])))
                print(msg2.format('dr', dr_tol))
                print(msg2.format('|dm|', dm_tol))
                print('      outlier tol: ', outlier_tol)
                print('          mag_lim: ', self.mag_lim[ii])
                print("   **********")

            star_list = self.star_lists[ii]
            ref_list = self.get_ref_list_from_table(star_list['t'][0])
            trans = self.trans_list[ii]

            # Trim a COPY of the reference and star lists based on magnitude.
            #       ref_list gets marked via the "use_in_trans" flag.
            #       star_list_T is actually trimmed but not yet transformed.
            self.apply_mag_lim_via_use_in_trans(ref_list, ref_mag_lim)
            star_list_T = apply_mag_lim(star_list, self.mag_lim[ii])

            ### Initial match and transform: 1st order (if we haven't already).
            if trans == None:
                trans = trans_initial_guess_new(ref_list, star_list_T, self.trans_args[0],
                                                mode=self.init_guess_mode,
                                                verbose=self.verbose)

            # Apply the XY transformation to a new copy of the starlist.
            star_list_T.transform_xym(trans)

            # Match stars between the transformed, trimmed lists.
            idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                             ref_list['x'], ref_list['y'], ref_list['m'],
                                             dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)

            if self.verbose:
                print( '  Match 1: Found ', len(idx1), ' matches out of ', len(star_list_T),
                       '. If match count is low, check dr_tol, dm_tol.' )


            # Outlier rejection
            if outlier_tol != None:
                keepers =  self.outlier_rejection_indices(star_list_T[idx1], ref_list[idx2],
                                                          outlier_tol)
                if self.verbose:
                    print( '  Rejected ', len(idx1) - len(keepers), ' outliers.' )
                    
                idx1 = idx1[keepers]
                idx2 = idx2[keepers]

            # Only use stars specified by "use_in_trans" column.
            if 'use_in_trans' in ref_list.colnames:
                keepers = np.where(ref_list[idx2]['use_in_trans'] == True)[0]
                
                if self.verbose:
                    print( '  Rejected ', len(idx1) - len(keepers), ' with use_in_trans=False.' )
                    
                idx1 = idx1[keepers]
                idx2 = idx2[keepers]


            # Determine weights in the fit.
            weight = self.get_weights_for_lists(ref_list[idx2], star_list_T[idx1])

            # Derive the best-fit transformation parameters. 
            if self.verbose:
                print( '  Using ', len(idx1), ' stars in transformation.' )
            trans = self.trans_class.derive_transform(star_list['x'][idx1], star_list['y'][idx1], 
                                                      ref_list['x'][idx2], ref_list['y'][idx2],
                                                      trans_args['order'],
                                                      m=star_list['m'][idx1], mref=ref_list['m'][idx2],
                                                      weights=weight)


            # Save the final transformation.
            self.trans_list[ii] = trans

            # Apply the XY transformation to a new copy of the starlist and
            # do one final match between the two (now transformed) lists.
            star_list_T = copy.deepcopy(star_list)
            star_list_T.transform_xym(trans)

            fmt = '{n:13s} {xl:9.5f} {xr:9.5f} {yl:9.5f} {yr:9.5f} {ml:6.2f} {mr:6.2f} '
            fmt += '{dx:7.2f} {dy:7.2f} {dm:6.2f} {xo:9.5f} {yo:9.5f} {mo:6.2f}'
            for foo in range(len(idx1)):
                star_s = star_list[idx1[foo]]
                star_r = ref_list[idx2[foo]]
                star_t = star_list_T[idx1[foo]]
                print(fmt.format(n=star_s['name'], xl=star_t['x'], xr=star_r['x'],
                                     yl=star_t['y'], yr=star_r['y'],
                                     ml=star_t['m'], mr=star_r['m'],
                                     dx=(star_t['x'] - star_r['x']) * 1e3, 
                                     dy=(star_t['y'] - star_r['y']) * 1e3,
                                     dm=(star_t['m'] - star_r['m']),
                                     xo=star_s['x'], yo=star_s['y'], mo=star_s['m']))

            idx_lis, idx_ref, dr, dm = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                                   ref_list['x'], ref_list['y'], ref_list['m'],
                                                   dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)

            if self.verbose:
                print( '  Match 2: After trans, found ', len(idx_lis), ' matches out of ', len(star_list_T),
                       '. If match count is low, check dr_tol, dm_tol.' )

            ### Update the observed (but transformed) values in the reference table.
            self.update_ref_table_from_list(star_list, star_list_T, ii, idx_ref, idx_lis, idx2)
            
            ### Update the "average" values to be used as the reference frame for the next list.
            if self.update_ref_orig != 'periter':
                self.update_ref_table_aggregates()

            # Print out some metrics
            if self.verbose:
                msg1 = '    {0:2s} (mean and std) for {1:10s}: {2:8.5f} +/- {3:8.5f}'
                print('    Residuals: ')
                print(msg1.format('dr', 'all stars', dr.mean(), dr.std()))
                print(msg1.format('dm', 'all stars', dm.mean(), dm.std()))

                # Calculate the residuals just for those used in the transformation
                used = np.where(self.ref_table['used_in_trans'][:, ii] == True)[0]
                used_good = used[ np.where(np.isin(used, idx_ref) == True)[0] ]
                dr_u = np.hypot(self.ref_table['x'][used_good, ii] - ref_list['x'][used_good],
                                self.ref_table['y'][used_good, ii] - ref_list['y'][used_good])
                dm_u = np.abs(self.ref_table['m'][used_good, ii] - ref_list['m'][used_good])
                print(msg1.format('dr', 'trans stars', dr_u.mean(), dr_u.std()))
                print(msg1.format('dm', 'trans stars', dm_u.mean(), dm_u.std()))
                print('    Dropped {0:d} matches after transform.'.format(len(used) - len(used_good)))
                
        return
    
    def setup_trans_info(self):
        """ Setup transformation info into a usable format.

        trans_input : list or None
        trans_args : dict or None
        N_lists : int
        iters : int
        """
        trans_input = self.trans_input
        trans_args = self.trans_args
        N_lists = len(self.star_lists)
        iters = self.iters
        
        trans_list = [None for ii in range(N_lists)]
        if trans_input != None:
            trans_list = [trans_input[ii] for ii in range(N_lists)]

        # Keep a list of trans_args, one for each starlist. If only
        # a single is passed in, replicate for all star lists, all loop iterations.
        if type(trans_args) == dict:
            tmp = trans_args
            trans_args = [tmp for ii in range(iters)]

        self.trans_list = trans_list
        self.trans_args = trans_args
        
        return

    def setup_ref_table_from_starlist(self, star_list):
        """ 
        Start with the reference list.... this will change and grow
        over time, so make a copy that we will keep updating.
        The reference table will contain one columne for every named
        array in the original reference star list.
        """
        col_arrays = {}
        for col_name in star_list.colnames:
            if col_name == 'name':
                # The "name" column will be 1D; but we will also add a "name_in_list" column.
                col_arrays['name'] = star_list[col_name].data
                new_col_name = "name_in_list"
            else:
                new_col_name = col_name

            # Make every column's 2D arrays except "name" and those
            # columns used for the motion model.
            if col_name in motion_model_col_names:
                col_arrays[new_col_name] = star_list[col_name].data
            else:
                new_col_data = np.array([star_list[col_name].data]).T
                col_arrays[new_col_name] = new_col_data

        # Use the columns from the ref list to make the ref_table.
        ref_table = StarTable(**col_arrays)

        # Make new columns to hold original values. These will be copies
        # of the old columns and will only include x, y, m, xe, ye, me.
        # The columns we have already created will hold transformed values. 
        trans_col_names = ['x', 'y', 'm', 'xe', 'ye', 'me', 'w']
        for tt in range(len(trans_col_names)):
            old_name = trans_col_names[tt]

            if old_name in ref_table.colnames:
                new_col = ref_table[old_name].copy()
                new_col.name = old_name + '_orig'
                ref_table.add_column(new_col)

        # Average the x, y, and m columns (although this is just a copy) and store in
        # x0, y0, m0. This will be what we use to align with. We will keep
        # updating the average with every new starlist.
        ref_table.combine_lists('x')
        ref_table.combine_lists('y')
        ref_table.combine_lists('m', ismag=True)

        # Make sure we have a column to indicate whether each star
        # CAN BE USED in the transformation. This will be 1D
        if 'use_in_trans' not in ref_table.colnames:
            new_col = Column(np.ones(len(ref_table), dtype=bool), name='use_in_trans')
            ref_table.add_column(new_col)

        # Make sure we have a column to indicate whether each star
        # IS USED in the transformation. This will be 2D
        if 'used_in_trans' not in ref_table.colnames:
            new_col = Column(np.zeros([len(ref_table),1], dtype=bool), name='used_in_trans')
            ref_table.add_column(new_col)
            
        # Keep track of whether this is an original reference star.
        col_ref_orig = Column(np.ones(len(ref_table), dtype=bool), name='ref_orig')
        ref_table.add_column(col_ref_orig)

        # Now reset the original values to invalids... they will be filled in
        # at later times. Preserve content only in the columns: name, x0, y0, m0 (and 0e).
        # Note that these are all the 1D columsn.
        for col_name in ref_table.colnames:
            if len(ref_table[col_name].data.shape) == 2:      # Find the 2D columns
                ref_table._set_invalid_list_values(col_name, -1)    

        return ref_table

    def apply_mag_lim_via_use_in_trans(self, ref_list, ref_mag_lim):
        """Set the use_in_trans flag to False for any star in the 
        star list that falls beyond the magnitude limits. 

        This should really only be applied to reference star lists.
        """
        if ((ref_mag_lim is not None) and (ref_mag_lim[0] is not None)):
            # Support 'm0' (primary) or 'm' column name.
            if 'm0' in ref_list.colnames:
                mcol = 'm0'
            else:
                mcol = 'm'

            no_use = np.where((ref_list[mcol] < ref_mag_lim[0]) |
                              (ref_list[mcol] >= ref_mag_lim[1]))

            ref_list['use_in_trans'][no_use]  = False
            
        return

    def outlier_rejection_indices(self, star_list, ref_list, outlier_tol, verbose=True):
        """
        Determine the outliers based on the residual positions between two different
        starlists and some threshold (in sigma). Return the indices of the stars 
        to keep (that shouldn't be rejected as outliers). 

        Note that we assume that the star_list and ref_list are already transformed and
        matched. 

        Parameters
        ----------
        star_list : StarList
            starlist with 'x', 'y'

        ref_list : StarList
            starlist with 'x0', 'y0'

        outlier_tol : float
            Number of sigma inside which we keep stars and outside of which we 
            reject stars as outliers. 

        Optional Parameters
        --------------------
        verbose : boolean

        Returns
        ----------
        keepers : nd.array
            The indicies of the stars to keep. 
        """
        # Optionally propogate the reference positions forward in time.
        xref = ref_list['x']
        yref = ref_list['y']

        # Residuals
        x_resid_on_old_trans = star_list['x'] - xref
        y_resid_on_old_trans = star_list['y'] - yref
        resid_on_old_trans = np.hypot(x_resid_on_old_trans, y_resid_on_old_trans)

        threshold = outlier_tol * resid_on_old_trans.std()
        keepers = np.where(resid_on_old_trans < threshold)[0]

        if verbose:
            msg = '  Outlier Rejection: Keeping {0:d} of {1:d}'
            print(msg.format(len(keepers), len(resid_on_old_trans)))

        return keepers

    def update_ref_table_from_list(self, star_list, star_list_T, ii, idx_ref, idx_lis, idx_ref_in_trans):
        """
        Inputs
        ----------
        star_list : StarList
            The original star list. 

        star_list_T : StarList
            The original star list now transformed into the reference coordinate system.

        ii : int
            The index of this epoch/starlist in the final table.

        idx_ref : np.array dtype=int
            The indices of the matched targets in the reference list/table.

        idx_lis : np.array dtype=int
            The indices of the matched targets in the origin starlist (epoch).

        idx_ref_in_trans : np.array dtype=int
            The indices in the reference table (self.ref_table). 
        """
        ### Update the reference table for matched stars.
        #   Add the matched stars to the reference table.
        #   For every epoch except the reference, we need to add a starlist.
        if ((self.ref_table['x'].shape[1] != len(self.star_lists)) and
            (ii != self.ref_index) and
            (ii >= self.ref_table['x'].shape[1])):
            
            self.ref_table.add_starlist()
                
        copy_over_values(self.ref_table, star_list, star_list_T, ii, idx_ref, idx_lis)
        self.ref_table['used_in_trans'][idx_ref_in_trans, ii] = True

        ### Add the unmatched stars and grow the size of the reference table.
        self.ref_table, idx_lis_new, idx_ref_new = add_rows_for_new_stars(self.ref_table, star_list, idx_lis)
        if len(idx_ref_new) > 0:
            print('    Adding {0:d} new stars to the reference table.'.format(len(idx_ref_new)))
            copy_over_values(self.ref_table, star_list, star_list_T, ii, idx_ref_new, idx_lis_new)

            # Copy the single-epoch values to the aggregate (only for new stars).
            self.ref_table['x0'][idx_ref_new] = star_list_T['x'][idx_lis_new]
            self.ref_table['y0'][idx_ref_new] = star_list_T['y'][idx_lis_new]
            self.ref_table['m0'][idx_ref_new] = star_list_T['m'][idx_lis_new]
            
            self.ref_table['name'] = update_old_and_new_names(self.ref_table, ii, idx_ref_new)

            if self.use_ref_new == True:
                self.ref_table['use_in_trans'][idx_ref_new] = True
            else:
                self.ref_table['use_in_trans'][idx_ref_new] = False
                
        return
        
    
    def update_ref_table_aggregates(self):
        """
        Average positions or fit velocities.
        Average magnitudes.

        Update the use_in_trans values as needed.

        Updates aggregate columns in self.ref_table in place.
        """
        # Keep track of the original reference values.
        # In certain cases, we will NOT update these.
        if not self.update_ref_orig:
            ref_orig_idx = np.where(self.ref_table['ref_orig'] == True)[0]
            x0_orig = self.ref_table['x0'][ref_orig_idx]
            y0_orig = self.ref_table['y0'][ref_orig_idx]
            m0_orig = self.ref_table['m0'][ref_orig_idx]
            x0e_orig = self.ref_table['x0e'][ref_orig_idx]
            y0e_orig = self.ref_table['y0e'][ref_orig_idx]
            m0e_orig = self.ref_table['m0e'][ref_orig_idx]

            if self.use_vel:
                vx_orig = self.ref_table['vx'][ref_orig_idx]
                vy_orig = self.ref_table['vy'][ref_orig_idx]
                vxe_orig = self.ref_table['vxe'][ref_orig_idx]
                vye_orig = self.ref_table['vye'][ref_orig_idx]
                t0_orig = self.ref_table['t0'][ref_orig_idx]
                
        
        if self.use_vel:
            # Combine positions with a velocity fit.
            self.ref_table.fit_velocities(verbose=self.verbose)
    
            # Combine (transformed) magnitudes
            if 'me' in self.ref_table.colnames:
                weights_col = None
            else:
                weights_col = 'me'
                
            self.ref_table.combine_lists('m', weights_col=weights_col, ismag=True)
        else:
            weighted_xy = ('xe' in self.ref_table.colnames) and ('ye' in self.ref_table.colnames)
            weighted_m = ('me' in self.ref_table.colnames)
    
            self.ref_table.combine_lists_xym(weighted_xy=weighted_xy, weighted_m=weighted_m)

        # Replace the originals if we are supposed to keep them fixed.
        if not self.update_ref_orig:
            self.ref_table['x0'][ref_orig_idx] = x0_orig
            self.ref_table['y0'][ref_orig_idx] = y0_orig
            self.ref_table['m0'][ref_orig_idx] = m0_orig
            self.ref_table['x0e'][ref_orig_idx] = x0e_orig
            self.ref_table['y0e'][ref_orig_idx] = y0e_orig
            self.ref_table['m0e'][ref_orig_idx] = m0e_orig

            if self.use_vel:
                self.ref_table['vx'][ref_orig_idx] = vx_orig
                self.ref_table['vy'][ref_orig_idx] = vy_orig
                self.ref_table['vxe'][ref_orig_idx] = vxe_orig
                self.ref_table['vye'][ref_orig_idx] = vye_orig
                self.ref_table['t0'][ref_orig_idx] = t0_orig
                
        return

    def get_weights_for_lists(self, ref_list, star_list):
        if 'xe' in ref_list.colnames:
            var_xref = ref_list['xe']**2
            var_yref = ref_list['ye']**2
        else:
            var_xref = 0.0
            var_yref = 0.0
            
        if 'xe' in star_list.colnames:
            var_xlis = star_list['xe']**2
            var_ylis = star_list['ye']**2
        else:
            var_xlis = 0.0
            var_ylis = 0.0

        if self.weights != None:
            if self.weights == 'both,var':
                weight = 1.0 / (var_xref + var_xlis + var_yref + var_ylis)
            if self.weights == 'both,std':
                weight = 1.0 / np.sqrt(var_xref + var_xlis + var_yref + var_ylis)
            if self.weights == 'ref,var':
                weight = 1.0 / (var_xref + var_yref)
            if self.weights == 'ref,std':
                weight = 1.0 / np.sqrt(var_xref + var_yref)
            if self.weights == 'list,var':
                weight = 1.0 / (var_xlis, var_ylis)
            if self.weights == 'list,std':
                weight = 1.0 / np.sqrt(var_xlis, var_ylis)
        else:
            weight = None

        # One last check to make sure we had weights at all.
        # Technically, this is mis-use; but lets handle it anyhow.
        if ('xe' not in ref_list.colnames) and ('ye' not in star_list.colnames):
            weight = None

        return weight

    
    def match_lists(self, dr_tol, dm_tol):
        """
        Using the existing trans objects, match all the starlists to the
        reference starlist (self.ref_table), propogated to the appropriate epoch. 

        No trimming of stars.
        No new transformations derived. 

        The resulting matched values will be used to update self.ref_table
        """
        for ii in range(self.N_lists):
            # Apply the XY transformation to a new copy of the starlist and
            # do one final match between the two (now transformed) lists.
            star_list_T = copy.deepcopy(self.star_lists[ii])
            star_list_T.transform_xym(self.trans_list[ii])
            
            xref, yref = get_pos_at_time(star_list_T['t'][0], self.ref_table)  # optional velocity propogation.
            mref = self.ref_table['m0']

            idx_lis, idx_ref, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                                   xref, yref, mref,
                                                   dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)
            if self.verbose:
                print('Matched {0:d} out of {1:d} stars in list {2:d}'.format(len(idx_lis), len(star_list_T), ii))

            copy_over_values(self.ref_table, self.star_lists[ii], star_list_T, ii, idx_ref, idx_lis)

        return

    def get_ref_list_from_table(self, epoch):
        """
        Convert the averaged quantites in self.ref_table into a StarList object
        appropriate for the specified epoch. 

        Columns in resulting reference list will include:
            name
            x
            y
            m
            xe (optional)
            ye (optional)
            me (optional)
            use_in_trans (optional)
        """
        # Reference stars will be named. 
        name = self.ref_table['name']

        if self.use_vel and ('vx' in self.ref_table.colnames):
            # First check if we should use velocities and if they exist.
            dt = epoch - self.ref_table['t0']
            x = self.ref_table['x0'] + (self.ref_table['vx'] * dt)
            y = self.ref_table['y0'] + (self.ref_table['vy'] * dt)
            
            xe = np.hypot(self.ref_table['x0e'], self.ref_table['vxe']*dt)
            ye = np.hypot(self.ref_table['y0e'], self.ref_table['vye']*dt)
        else:
            # No velocities... just used average positions.
            x = self.ref_table['x0']
            y = self.ref_table['y0']
            
            if 'x0e' in self.ref_table.colnames:
                xe = self.ref_table['x0e']
                ye = self.ref_table['y0e']
            else:
                xe = None
                ye = None

        m = self.ref_table['m0']
        
        if 'm0e' in self.ref_table.colnames:
            me = self.ref_table['m0e']
        else:
            me = None

        # Check if we have any information about which stars to
        # use in the transformation.
        use_in_trans = np.ones(len(x), dtype=bool)
        if 'use_in_trans' in self.ref_table.colnames:
            use_in_trans = self.ref_table['use_in_trans']

        # Make starlist
        ref_list = StarList(name=name, x=x, y=y, m=m)
        ref_list.add_column(use_in_trans, name='use_in_trans')

        # Check if we should add errors.
        if (xe is not None) and (ye is not None):
            ref_list['xe'] = xe
            ref_list['ye'] = ye

        if me is not None:
            ref_list['me'] = me

        return ref_list

    def reset_ref_values(self, exclude=None):
        """
        Reset all the 2D arrays in the reference table. This is the action
        we take at the beginning of each new iteration. We don't preserve matching
        results from the prior iterations. 
        """
        # All 2D columns should be reset.
        for col_name in self.ref_table.colnames:
            if (exclude != None) and (col_name in exclude):
                continue
            
            if len(self.ref_table[col_name].data.shape) == 2:      # Find the 2D columns
                # Loop through epochs for this array.
                for cc in range(self.ref_table[col_name].shape[1]):
                    self.ref_table._set_invalid_list_values(col_name, cc)

        return
    


class MosaicToRef(MosaicSelfRef):
    def __init__(self, ref_list, list_of_starlists, iters=2,
                 dr_tol=[1, 1], dm_tol=[2, 1],
                 outlier_tol=[None, None],
                 trans_args=[{'order': 2}, {'order': 2}],
                 mag_trans=True, mag_lim=None, ref_mag_lim=None,
                 weights=None,
                 trans_input=None, trans_class=transforms.PolyTransform,
                 use_ref_new=False,
                 use_vel=False, update_ref_orig=False,
                 init_guess_mode='miracle',
                 iter_callback=None,
                 verbose=True):

        """
        Required Parameters
        ----------
        ref_list : StarList object
            Can optionally have velocities. All starlists will be aligned to this one. 

        list_of_starlists : array of StarList objects
            An array or list of flystar.starlists.StarList objects (which are Astropy Tables).
            There should be one for each starlist and they must contain 'x', 'y', and 'm' columns.

            Note that there is an optional weights column called 'w'. If this column exists
            in any of the lists, it will be queried to determine if an individual star can be
            used to derive the transformations between starlists. This is the most flexible way
            to allow you to determine, as a function of time and star, which ones are good enough 
            in the transformation. Note that just because it can be used (i.e. w_in=1), 
            doesn't meant that it will be used. The mag limits and outliers still take precedence. 
            Note also that the weights that go into the transformation are 

                star_list['w'] * ref_list['w'] * weight_from_keyword (see the weights parameter)

            for those stars not trimmed out by the other criteria. 


        Optional Parameters
        ----------
        iters : int
            The number of iterations used in the matching and transformation.  TO DO: INNER/OUTER? 

        dr_tol : list or array
            The delta-radius (dr) tolerance for matching in units of the reference coordinate system.
            This is a list of dr values, one for each iteration of matching/transformation.

        dm_tol : list or array
            The delta-magnitude (dm) tolerance for matching in units of the reference coordinate system.
            This is a list of dm values, one for each iteration of matching/transformation. 

        mag_trans : boolean
            If true, this will also calculate and (temporarily) apply a zeropoint offset to 
            magnitudes in each list to bring them into a common magnitude system. This is 
            essential for matching (with finite dm_tol) starlists of different filters or 
            starlists that are not photometrically calibrated. Note that the final_table columns 
            of 'm', 'm0', and 'm0e' will contain the transformed magnitudes while the 
            final_table column 'm_orig' will contain the original un-transformed magnitudes. 
            If mag_trans = False, then no such zeropoint offset it applied at any point. 

        mag_lim : array
            If different from None, it indicates the minimum and maximum magnitude
            on the catalogs for finding the transformations

        ref_mag_lim : array
            If different from None, it indicates the minimum and maximum magnitude
            on the reference catalog for finding the transformations.

        weights : str
            Either None (def), 'both,var', 'list,var', or 'ref,var' depending on whether you want
            to weight by the positional uncertainties (variances) in the individual starlists, or also with
            the uncertainties in the reference frame itself.  Note weighting only works when there
            are positional uncertainties availabe. Other options include 'both,std', 'list,std', 'list,var'.

        trans_input : array or list of transform objects
            def = None. If not None, then this should contain an array or list of transform
            objects that will be used as the initial guess in the alignment and matching. 

        trans_class : transforms.Transform2D object (or subclass)
            The transform class that will be used to when deriving the optimal
            transformation parameters between each list and the reference list. 

        trans_args : dictionary
            A dictionary (or a list of dictionaries) containing any extra keywords that are needed 
            in the transformation object. For instance, "order". Note that if a list is passed in, 
            then the transformation argument (i.e. order) will be changed for every iteration in
            iters.

        update_ref_orig : boolean or str
            Should we update the reference values (position, velocity, t0) after each starlist
            is transformed in each iteration? 

                False if you want to get into an absolute reference frame and are using Gaia data. 
                True if you want to use the reference list as more of an initial guess.
                'periter' if you want to align all the starlists, then calculate the velocity.

            Note that this only impacts the stars that are in the original reference list... the
            newly identified stars that end up in ref_table will always be updated; but not always
            used for transformation fitting.

        use_ref_new : boolean
            Each pass, new stars are matched and added to the ref_table. However, we don't 
            necessarily want to use these in the reference frame in subsequent passes. 
            If True, then the new stars will be used in later passes/iterations.
            If False, then the new stars will be carried, but not used in the transformation.
            We determine which stars to use through setting a boolean use_in_trans flag. 

        use_vel : boolean
            If velocities are present in the reference list and use_vel == True, then during
            each iteration of the alignment, the reference list will be propogated in time
            using the velocity information. So all transformations will be derived w.r.t. 
            the propogated positions. See also update_vel.

        iter_callback : None or function
            A function to call (that accepts a StarTable object and an iteration number)
            at the end of every iteration. This can be used for plotting or printing state. 

        """

        super().__init__(list_of_starlists, ref_index=-1, iters=iters,
                         dr_tol=dr_tol, dm_tol=dm_tol,
                         outlier_tol=outlier_tol, trans_args=trans_args,
                         mag_trans=mag_trans, mag_lim=mag_lim, weights=weights,
                         trans_input=trans_input, trans_class=trans_class,
                         use_vel=use_vel, init_guess_mode=init_guess_mode,
                         iter_callback=iter_callback,
                         verbose=verbose)
        
        self.ref_list = copy.deepcopy(ref_list)
        self.ref_mag_lim = ref_mag_lim
        self.update_ref_orig = update_ref_orig
        self.use_ref_new = use_ref_new

        # Do some temporary clean up of the reference list.
        if ('x' not in self.ref_list.colnames) and ('x0' in self.ref_list.colnames):
            self.ref_list['x'] = self.ref_list['x0']
            self.ref_list['y'] = self.ref_list['y0']
        if ('xe' not in self.ref_list.colnames) and ('x0e' in self.ref_list.colnames):
            self.ref_list['xe'] = self.ref_list['x0e']
            self.ref_list['ye'] = self.ref_list['y0e']
        if ('m' not in self.ref_list.colnames) and ('m0' in self.ref_list.colnames):
            self.ref_list['m'] = self.ref_list['m0']
        if ('me' not in self.ref_list.colnames) and ('m0e' in self.ref_list.colnames):
            self.ref_list['me'] = self.ref_list['m0e']
        if ('t' not in self.ref_list.colnames) and ('t0' in self.ref_list.colnames):
            self.ref_list['t'] = self.ref_list['t0']

        return

    
    def fit(self):
        """
        Using the current parameter settings, match and transform all the lists
        to a reference position. Note in the first pass, the reference position
        is just the specified input reference starlist. In subsequent iterations,
        this is (optionally) updated. 

        The ultimate outcome is the creation of self.ref_table. This reference
        table will contain "averaged" quantites as well as a big 2D array of all
        the matched original and transformed quantities. 

        Averaged columns on ref_table:
        x0
        y0
        m0
        x0e
        y0e
        m0e
        vx  (only if use_vel=True)
        vy  (only if use_vel=True)
        vxe (only if use_vel=True)
        vye (only if use_vel=True)

        """
        ##########
        # Setup a reference table to store data. It will contain:
        #    x0, y0, m0 -- the running average of positions: 1D
        #    x, y, m, (opt. errors) -- the transformed positions for the lists: 2D
        #    x_orig, y_orig, m_orig, (opt. errors) -- the transformed errors for the lists: 2D
        #    w, w_orig (optiona) -- the input and output weights of stars in transform: 2D
        ##########
        self.ref_table = self.setup_ref_table_from_starlist(self.ref_list)
        
        # copy over velocities if they exist in the reference list
        if 'vx' in self.ref_list.colnames:
            self.ref_table['vx'] = self.ref_list['vx']
            self.ref_table['vy'] = self.ref_list['vy']
            self.ref_table['t0'] = self.ref_list['t0']
        if 'vxe' in self.ref_list.colnames:
            self.ref_table['vxe'] = self.ref_list['vxe']
            self.ref_table['vye'] = self.ref_list['vye']


        ##########
        #
        # Repeat transform + match of all the starlists several times.
        #
        ##########
        for nn in range(self.iters):
            
            # If we are on subsequent iterations, remove matching results from the 
            # prior iteration. This leaves aggregated (1D) columns alone.
            if nn > 0:
                self.reset_ref_values()

            if self.verbose:
                print(" ")
                print("**********")
                print("**********")
                print('Starting iter {0:d} with ref_table shape:'.format(nn), self.ref_table['x'].shape)
                print("**********")
                print("**********")
                
            # ALL the action is in here. Match and transform the stack of starlists.
            # This updates trans objects and the ref_table.
            self.match_and_transform(self.ref_mag_lim,
                                     self.dr_tol[nn], self.dm_tol[nn], self.outlier_tol[nn],
                                     self.trans_args[nn])

            # Clean up the reference table
            # Find where stars are detected.
            self.ref_table.detections()

            ### Drop all stars that have 0 detections.
            idx = np.where((self.ref_table['n_detect'] == 0) & (self.ref_table['ref_orig'] == False))[0]
            print('  *** Getting rid of {0:d} out of {1:d} junk sources'.format(len(idx), len(self.ref_table)))
            self.ref_table.remove_rows(idx)

            if self.iter_callback != None:
                self.iter_callback(self.ref_table, nn)
            

        ##########
        #
        # Re-do all matching given final transformations.
        #        No trimming this time.
        #        First rest the reference table 2D values. 
        ##########
        self.reset_ref_values(exclude=['used_in_trans'])

        if self.verbose:
            print("**********")
            print("Final Matching")
            print("**********")

        self.match_lists(self.dr_tol[-1], self.dm_tol[-1])
        self.update_ref_table_aggregates()
                

        ##########
        # Clean up output table.
        # 
        ##########
        # Find where stars are detected.
        if self.verbose:
            print('')
            print('   Preparing the reference table...')
            
        self.ref_table.detections()

        ### Drop all stars that have 0 detections.
        idx = np.where(self.ref_table['n_detect'] == 0)[0]
        print('  *** Getting rid of {0:d} out of {1:d} junk sources'.format(len(idx), len(self.ref_table)))
        self.ref_table.remove_rows(idx)

        if self.iter_callback != None:
            self.iter_callback(self.ref_table, nn)

        return



########################################
# Old way of doing things... we still use
# some utilities down here. 
########################################    
    
def mosaic_lists(list_of_starlists, ref_index=0, iters=2,
                 dr_tol=[1, 1], dm_tol=[2, 1],
                 outlier_tol=[None, None],
                 trans_args=[{'order': 2}, {'order': 2}],
                 mag_trans=True, mag_lim=None, weights=None,
                 trans_input=None, trans_class=transforms.PolyTransform,
                 update_mag_offset=True, update_ref_per_iter=True,
                 use_vel=False, update_vel=True,
                 init_guess_mode='miracle', init_guess_n_req=2,
                 ref_epoch_mean=True, verbose=True):
    """
    Required Parameters
    ----------
    list_of_starlists : array of StarList objects
        An array or list of flystar.starlists.StarList objects (which are Astropy Tables).
        There should be one for each starlist and they must contain 'x', 'y', and 'm' columns.

        Note that there is an optional weights column called 'w'. If this column exists
        in any of the lists, it will be queried to determine if an individual star can be
        used to derive the transformations between starlists. This is the most flexible way
        to allow you to determine, as a function of time and star, which ones are good enough 
        in the transformation. Note that just because it can be used (i.e. w_in=1), 
        doesn't meant that it will be used. The mag limits and outliers still take precedence. 
        Note also that the weights that go into the transformation are 
        
            star_list['w'] * ref_list['w'] * weight_from_keyword (see the weights parameter)

        for those stars not trimmed out by the other criteria. 


    Optional Parameters
    ----------
    ref_index : int
        The index of the reference epoch. (default = 0). Note that this is the reference
        list only for the first iteration. Subsequent iterations will utilize the sigma-clipped 
        mean of the positions from all the starlists. 

    iters : int
        The number of iterations used in the matching and transformation.  TO DO: INNER/OUTER? 

    dr_tol : list or array
        The delta-radius (dr) tolerance for matching in units of the reference coordinate system.
        This is a list of dr values, one for each iteration of matching/transformation.
    
    dm_tol : list or array
        The delta-magnitude (dm) tolerance for matching in units of the reference coordinate system.
        This is a list of dm values, one for each iteration of matching/transformation. 

    mag_trans : boolean
        If true, this will also calculate and (temporarily) apply a zeropoint offset to 
        magnitudes in each list to bring them into a common magnitude system. This is 
        essential for matching (with finite dm_tol) starlists of different filters or 
        starlists that are not photometrically calibrated. Note that the final_table columns 
        of 'm', 'm0', and 'm0e' will contain the transformed magnitudes while the 
        final_table column 'm_orig' will contain the original un-transformed magnitudes. 
        If mag_trans = False, then no such zeropoint offset it applied at any point. 
    
    mag_lim : array
        If different from None, it indicates the minimum and maximum magnitude
        on the catalogs for finding the transformations

    weights : str
        Either None (def), 'both,var', 'list,var', or 'ref,var' depending on whether you want
        to weight by the positional uncertainties (variances) in the individual starlists, or also with
        the uncertainties in the reference frame itself.  Note weighting only works when there
        are positional uncertainties availabe. Other options include 'both,std', 'list,std', 'list,var'.
    
    trans_input : array or list of transform objects
        def = None. If not None, then this should contain an array or list of transform
        objects that will be used as the initial guess in the alignment and matching. 

    trans_class : transforms.Transform2D object (or subclass)
        The transform class that will be used to when deriving the optimal
        transformation parameters between each list and the reference list. 

    trans_args : dictionary
        A dictionary (or a list of dictionaries) containing any extra keywords that are needed 
        in the transformation object. For instance, "order". Note that if a list is passed in, 
        then the transformation argument (i.e. order) will be changed for every iteration in
        iters.
    
    update_mag_offset : boolean
        Update the magnitude offset every time a new transformation is found.
        A 3-sigma clipped mean is used

    update_ref_per_iter : boolean
        Update the reference list positions for each new starlist that is transformed. In 
        other words, if you are aligning lists A, B, C, and D, then the first loop has
        ref=A and the second loop has ref=(A+B)/2, etc. by default. Set update_ref_per_iter=False
        to keep referenece list fixed for every new list transform.

    use_vel : boolean
        If velocities are present in the reference list and use_vel == True, then during
        each iteration of the alignment, the reference list will be propogated in time
        using the velocity information. So all transformations will be derived w.r.t. 
        the propogated positions. See also update_vel.

    update_vel : boolean
        Fit velocities to the positions vs. time after the first iteration. In subsequent
        iterations, the velocities from the first iteration will be used if use_vel=True. 
        If use_vel=False and update_vel=True, then velocities will still be determined; but
        not used in subsequent iterations of the fitting. (why would you do this? If you 
        have trustworthy absolute proper motions from an external source). 
    
    ref_epoch_mean : boolean
        Include the reference catalog to calculate the last xym combination

    """
    ###
    ### QUESTION: Do we want to add support for preserving (not updating) the reference list.
    ### We would only update the reference at the end (outside of mosaic_lists).
    ###

    ##########
    # Error checking for parameters.
    ##########
    check_iter_tolerances(iters, dr_tol, dm_tol, outlier_tol)
    if mag_lim == None:
        mag_lim = np.repeat([[None, None]], len(list_of_starlists), axis=0)
        assert len(mag_lim) == len(list_of_starlists)

    
    check_trans_input(list_of_starlists, trans_input, mag_trans)
    
    star_lists = list_of_starlists    # Shorthand
    N_lists = len(star_lists)

    ##########
    # Setup a reference table to store data. It will contain:
    #    x0, y0, m0 -- the running average of positions: 1D
    #    x, y, m, (opt. errors) -- the transformed positions for the lists: 2D
    #    x_orig, y_orig, m_orig, (opt. errors) -- the transformed errors for the lists: 2D
    #    w, w_orig (optiona) -- the input and output weights of stars in transform: 2D
    ##########
    ref_table = setup_ref_table_from_starlist(star_lists[ref_index])
    
    # Save the reference index to the meta data on the reference list.
    ref_table.meta['ref_list'] = ref_index

    
    ##########
    # Setup transformation lists and arguments.
    #     - Keep a list of the transformation objects for each epochs.
    #     - Load up previous transformations, if they exist.
    #     - Keep a list of trans_args, one for each starlist. If only a single
    #       is passed in, replicate for all star lists, all loop iterations.
    ##########
    trans_list, trans_args = setup_trans_info(trans_input, trans_args, N_lists, iters)

    
    ##########
    #
    # Loop through starlists and align them to the reference list, one at a time.
    #
    ##########

    ### Repeat transform + match several times.
    for nn in range(iters):

        # If we are on subsequent iterations, remove matching
        # results from the prior iteration. 
        if nn > 0:
            reset_ref_values(ref_table)
            
        ### Get the (updated) reference list
        #       ref_list - not trimmed (only used for final matching).
        if update_ref_per_iter:
            ref_list = StarList(ref_table['name', 'x0', 'y0', 'm0', 'x0e', 'y0e', 'm0e'])
        else:
            ref_list = copy_and_rename_for_ref(star_lists[ref_index])

        if verbose:
            print(" ")
            print("**********")
            print("**********")
            print('Starting iter {0:d} with ref_table shape:'.format(nn), ref_table['x'].shape)
            print("**********")
            print("**********")

        ## In this iteration, loop through the starlists.
        for ii in range(len(star_lists)):
            if verbose:
                msg = '   Matching catalog {0} / {1} in iteration {2} with {3:d} stars'
                print(" ")
                print("   **********")
                print(msg.format((ii + 1), len(star_lists), nn, len(star_lists[ii])))
                print("   **********")

            star_list = star_lists[ii]
            trans = trans_list[ii]


            # Trim a COPY of the reference and star lists based on magnitude.
            #       ref_list_T  - trimmed (used for all matching/transformation derivations)
            #       star_list_T - trimmed but not yet transformed.
            ref_list_T  = apply_mag_lim(ref_list, mag_lim[ref_index])
            star_list_T = apply_mag_lim(star_list, mag_lim[ii])       

            ### Initial match and transform: 1st order (if we haven't already).
            if trans == None:
                trans = trans_initial_guess(ref_list_T, star_list_T, trans_args[0],
                                            mode=init_guess_mode, n_req_match=init_guess_n_req,
                                            verbose=verbose)

            mag_offset = trans.mag_offset

            # Apply the XY transformation to a new copy of the starlist.
            star_list_T.transform_xym(trans)

            # Match stars between the transformed, trimmed lists.
            xref, yref = get_pos_at_time(star_list_T['t'][0], ref_list_T)  # optional velocity propogation.
            mref = ref_list_T['m0']
                                             
            idx1, idx2, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                             xref, yref, mref,
                                             dr_tol=dr_tol[nn], dm_tol=dm_tol[nn], verbose=verbose)

            if verbose:
                print( 'In Loop ', nn, ' found ', len(idx1), ' matches' )

            # Outlier rejection
            if outlier_tol[nn] != None:
                keepers = outlier_rejection_indices(star_list_T[idx1], ref_list_T[idx2],
                                                        outlier_tol[nn])
                idx1 = idx1[keepers]
                idx2 = idx2[keepers]


            # Determine weights in the fit.
            weight = get_weighting_scheme(weights, ref_list_T[idx2], star_list_T[idx1])

            # Derive the transformation parameters for this list. 
            xref, yref = get_pos_at_time(star_list_T['t'][0], ref_list_T[idx2])  # optional velocity propogation.
            mref = ref_list_T['m0'][idx2]
            
            trans = trans_class.derive_transform(star_list['x'][idx1], star_list['y'][idx1], 
                                                 xref, yref, 
                                                **(trans_args[nn]),
                                                m=star_list_T['m'][idx1], mref=mref,
                                                weights=weight)


            if ~update_mag_offset:
                trans.mag_offset = mag_offset

            # Save the final transformation.
            trans_list[ii] = trans

            # Apply the XY transformation to a new copy of the starlist and
            # do one final match between the two (now transformed) lists.
            star_list_T = copy.deepcopy(star_list)
            star_list_T.transform_xym(trans)
            
            xref, yref = get_pos_at_time(star_list_T['t'][0], ref_list)  # optional velocity propogation.
            mref = ref_list_T['m0']
            
            idx_lis, idx_ref, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                                   xref, yref, mref,
                                                   dr_tol=dr_tol[nn], dm_tol=dm_tol[nn], verbose=verbose)

            ### Update the reference table for matched stars.
            # Add the matched stars to the reference table.
            # For every epoch except the reference, we need to add a starlist.
            if (nn == 0) and (ii != ref_index):
                ref_table.add_starlist()
            copy_over_values(ref_table, star_list, star_list_T, ii, idx_ref, idx_lis)
        
            ### Add the unmatched stars and grow the size of the reference table.
            ref_table, idx_lis_new, idx_ref_new = add_rows_for_new_stars(ref_table, star_list, idx_lis)

            if len(idx_ref_new) > 0:
                copy_over_values(ref_table, star_list, star_list_T, ii, idx_ref_new, idx_lis_new)
                ref_table['name'] = update_old_and_new_names(ref_table, ii, idx_ref_new)
        
            ### Update the "average" values to be used as the reference frame for the next list.
            weighted_xy = ('xe' in ref_table.colnames) and ('ye' in ref_table.colnames)
            weighted_m = ('me' in ref_table.colnames)
            
            ### Remove the reference epoch from the mean
            if not ref_epoch_mean:
                ref_table.combine_lists_xym(weighted_xy=weighted_xy, weighted_m=weighted_m,
                                    mask_lists=[ref_index])
            else:
                ref_table.combine_lists_xym(weighted_xy=weighted_xy, weighted_m=weighted_m)

            
    ### Re-do all matching given final transformations.
    #        No trimming this time.
    #        First rest the reference table 2D values. 
    reset_ref_values(ref_table)

    if verbose:
        print("**********")
        print("Final Matching")
        print("**********")
        
    for ii in range(len(star_lists)):
        # Apply the XY transformation to a new copy of the starlist and
        # do one final match between the two (now transformed) lists.
        star_list_T = copy.deepcopy(star_lists[ii])
        star_list_T.transform_xym(trans_list[ii])
        
        idx_lis, idx_ref, dm, dr = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                               ref_table['x0'], ref_table['y0'], ref_table['m0'],
                                               dr_tol=dr_tol[-1], dm_tol=dm_tol[-1], verbose=verbose)
        if verbose:
            print('Matched {0:d} out of {1:d} stars in list {2:d}'.format(len(idx_lis), len(star_list_T), ii))
            
        copy_over_values(ref_table, star_lists[ii], star_list_T, ii, idx_ref, idx_lis)

    
    ### Find where stars are detected
    if verbose:
        print('')
        print('   Preparing the reference table...')
    ref_table.detections()

    ### Drop all stars that have 0 detections.
    idx = np.where(ref_table['n_detect'] != 0)[0]
    ref_table = ref_table[idx]
    
    return ref_table, trans_list

def setup_ref_table_from_starlist(star_list):
    """ 
    Start with the reference list.... this will change and grow
    over time, so make a copy that we will keep updating.
    The reference table will contain one columne for every named
    array in the original reference star list.
    """
    col_arrays = {}
    for col_name in star_list.colnames:
        if col_name == 'name':
            # The "name" column will be 1D; but we will also add a "name_in_list" column.
            col_arrays['name'] = star_list[col_name].data
            new_col_name = "name_in_list"
        else:
            new_col_name = col_name
            
        # Make every column's 2D arrays except "name" and those
        # columns used for the motion model.
        if col_name in motion_model_col_names:
            col_arrays[new_col_name] = star_list[col_name].data
        else:
            new_col_data = np.array([star_list[col_name].data]).T
            col_arrays[new_col_name] = new_col_data

    # Use the columns from the ref list to make the ref_table.
    ref_table = StarTable(**col_arrays)

    # Make new columns to hold original values. These will be copies
    # of the old columns and will only include x, y, m, xe, ye, me.
    # The columns we have already created will hold transformed values. 
    trans_col_names = ['x', 'y', 'm', 'xe', 'ye', 'me', 'w']
    for tt in range(len(trans_col_names)):
        old_name = trans_col_names[tt]

        if old_name in ref_table.colnames:
            new_col = ref_table[old_name].copy()
            new_col.name = old_name + '_orig'
            ref_table.add_column(new_col)

    # Average the x, y, and m columns (although this is just a copy) and store in
    # x0, y0, m0. This will be what we use to align with. We will keep
    # updating the average with every new starlist.
    ref_table.combine_lists('x')
    ref_table.combine_lists('y')
    ref_table.combine_lists('m', ismag=True)

    if 'use_in_trans' not in ref_table.colnames:
        new_col = Column(np.ones(len(ref_table), dtype=bool), name='use_in_trans')
        ref_table.add_column(new_col)

    # Now reset the original values to invalids... they will be filled in
    # at later times. Preserve content only in the columns: name, x0, y0, m0 (and 0e).
    # Note that these are all the 1D columsn.
    for col_name in ref_table.colnames:
        if len(ref_table[col_name].data.shape) == 2:      # Find the 2D columns
            ref_table._set_invalid_list_values(col_name, -1)    

    return ref_table

def copy_over_values(ref_table, star_list, star_list_T, idx_epoch, idx_ref, idx_lis):
    """
    Copy values from an individual starlist (both untransformed and transformed)
    into the reference table we carry around and that is the final output product.
    Copy only those values for stars that match.

    Copy all columns that are in both ref_table and star_list_T. 
    Copy all columns that are also in star_list but copy them into <col>_orig.

    Parameters
    ----------
    ref_table : StarTable
        The table we will be copying values into. Note the columns with the appropriate
        names and dimensions must already exist. 
    star_list : StarList
        The astropy table to copy values from. These should be untransformed (orig) values.
    star_list_T : StarList
        The astropy table to copy values from. These should be transformed values.
    idx_ref : list or array
        The indices into the ref_table where values are copied to.
    idx_lis : list or array
        The indices into the star_list or star_list_T where values are copied from.
    """
    for col_name in ref_table.colnames:
        if col_name in star_list_T.colnames:
            if col_name == 'name':
                ref_table['name_in_list'][idx_ref, idx_epoch] = star_list_T[col_name][list(idx_lis)]
            else:
                ref_table[col_name][idx_ref, idx_epoch] = star_list_T[col_name][list(idx_lis)]

            orig_col_name = col_name + '_orig'
            if orig_col_name in ref_table.colnames:
                ref_table[orig_col_name][idx_ref, idx_epoch] = star_list[col_name][list(idx_lis)]

    return

def reset_ref_values(ref_table):
    """
    Reset all the 2D arrays in the reference table. This is the action
    we take at the beginning of each new iteration. We don't preserve matching
    results from the prior iterations. 
    """
    # All 2D columns should be reset.
    for col_name in ref_table.colnames:
        if len(ref_table[col_name].data.shape) == 2:      # Find the 2D columns
            # Loop through epochs for this array.
            for cc in range(ref_table[col_name].shape[1]):
                ref_table._set_invalid_list_values(col_name, cc)
                
    return

def add_rows_for_new_stars(ref_table, star_list, idx_lis):
    """
    For each star that is in star_list and NOT in idx_list, make a 
    new row in the reference table. The values will be empty (None, NAN, etc.). 

    Parameters
    ----------
    ref_table : StarTable
        The reference table that the rows will be added to.

    star_list : StarList
        The starlist that will be used to estimate how many new stars there are.

    idx_lis : array or list
        The indices of the non-new stars (those that matched already). The complement
        of this array will be used as the new stars.

    Returns
    ----------
    ref_table : StarTable
        The reference table with rows added into.
    idx_lis_new : list
        The list of indices into the star_list object for the "new" stars.
    idx_ref_new : list
        The list of indices into the ref_table object for the "new" stars. 

    """
    last_star_idx = len(ref_table)

    idx_lis_orig = np.arange(len(star_list))
    idx_lis_new = np.array(list(set(idx_lis_orig) - set(idx_lis)))

    if len(idx_lis_new) > 0:
        col_arrays = {}

        for col_name in ref_table.colnames:
            new_col_name = col_name

            if ref_table[col_name].dtype == np.dtype('float'):
                new_col_empty = np.nan
            elif ref_table[col_name].dtype == np.dtype('int'):
                new_col_empty = -1
            elif ref_table[col_name].dtype == np.dtype('bool'):
                new_col_empty = False
            else:
                new_col_empty = None
            
            if len(ref_table[col_name].shape) == 1:
                new_col_shape = len(idx_lis_new)
            else:
                new_col_shape = [len(idx_lis_new), ref_table[col_name].shape[1]]

            new_col_data = Column(data=np.tile(new_col_empty, new_col_shape),
                                  name=col_name, dtype=ref_table[col_name].dtype)
            col_arrays[new_col_name] = new_col_data

        ref_table_new = StarTable(**col_arrays)
        ref_table_nstars = ref_table.meta['n_stars'] + ref_table_new.meta['n_stars']
        ref_table.meta['n_stars'] = ref_table_nstars
        ref_table_new.meta['n_stars'] = ref_table_nstars
        ref_table_new.meta['ref_list'] = ref_table.meta['ref_list']
        ref_table = vstack([ref_table, ref_table_new])

    idx_ref_new = np.arange(last_star_idx, len(ref_table))
    
    return ref_table, idx_lis_new, idx_ref_new


def run_align_iter(catalog, trans_order=1, poly_deg=1, ref_mag_lim=19, ref_radius_lim=300):
    # Load up data with matched stars.
    d = Table.read(catalog)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine the reference epoch
    ref = d.meta['L_REF']

    # Figure out the number of free parameters for the specified
    # poly2d order.
    poly2d = models.Polynomial2D(trans_order)
    N_par_trans_per_epoch = 2.0 * poly2d.get_num_coeff(2)  # one poly2d for each dimension (X, Y)
    N_par_trans = N_par_trans_per_epoch * N_epochs

    ##########
    # First iteration -- align everything to REF epoch with zero velocities. 
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 1')
    ee_ref = d.meta['L_REF']

    target_name = 'OB120169'

    trans1, used1 = calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim)

    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0)

    calc_mag_avg_all_stars(d)
    
    tdx = np.where((d['name_0'] == 'OB120169') | (d['name_0'] == 'OB120169_L'))[0]
    print(d[tdx]['name_0', 't0', 'mag', 'x0', 'vx', 'x0e', 'vxe', 'chi2x', 'y0', 'vy', 'y0e', 'vye', 'chi2y', 'dof'])

    ##########
    # Second iteration -- align everything to reference positions derived from iteration 1
    ##########
    print('ALIGN_EPOCHS: run_align_iter() -- PASS 2')
    target_name = 'OB120169'

    trans2, used2 = calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim)
    
    ##########
    # Derive the velocity of each stars using the round 1 transforms. 
    ##########
    calc_polyfit_all_stars(d, poly_deg, init_fig_idx=4)

    ##########
    # Save output
    ##########
    d.write(catalog.replace('.fits', '_aln.fits'), overwrite=True)
    
    return

def calc_transform_ref_epoch(d, target_name, ee_ref, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Reference values
    t_ref = d['t_{0:d}'.format(ee_ref)]
    m_ref = d['m_{0:d}'.format(ee_ref)]
    x_ref = d['x_{0:d}'.format(ee_ref)]
    y_ref = d['y_{0:d}'.format(ee_ref)]
    xe_ref = d['xe_{0:d}'.format(ee_ref)]
    ye_ref = d['ye_{0:d}'.format(ee_ref)]
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    # Loop through and align each epoch to the reference epoch.
    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Figure out the set of stars detected in both epochs.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both epochs AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref + ye_ref) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref[idx][rdx], y_ref[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)

        trans.append(trans_tmp)


        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)

        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref, y_ref, idx, rdx, 'Epoch: ' + str(ee))
            
    used = np.array(used)
    
    return trans, used


def calc_transform_ref_poly(d, target_name, poly_deg, ref_mag_lim, ref_radius_lim):
    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # output array
    trans = []
    used = []

    # Find the target
    tdx = np.where(d['name_0'] == 'OB120169')[0][0]

    # Temporary Reference values
    t_ref = d['t0']
    m_ref = d['mag']
    x_ref = d['x0']
    y_ref = d['y0']
    xe_ref = d['x0e']
    ye_ref = d['y0e']    
    
    # Calculate some quanitites we use for selecting reference stars.
    r_ref = np.hypot(x_ref - x_ref[tdx], y_ref - y_ref[tdx])

    for ee in range(N_epochs):
        # Pull out the X, Y positions (and errors) for the two
        # starlists we are going to align.
        x_epo = d['x_{0:d}'.format(ee)]
        y_epo = d['y_{0:d}'.format(ee)]
        t_epo = d['t_{0:d}'.format(ee)]
        xe_epo = d['xe_{0:d}'.format(ee)]
        ye_epo = d['ye_{0:d}'.format(ee)]

        # Shift the reference position by the polyfit for each star.
        dt = t_epo - t_ref
        if poly_deg >= 0:
            x_ref_ee = x_ref
            y_ref_ee = y_ref
            xe_ref_ee = x_ref
            ye_ref_ee = y_ref
            
        if poly_deg >= 1:
            x_ref_ee += d['vx'] * dt
            y_ref_ee += d['vy'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['vxe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['vye'] * dt)

        if poly_deg >= 2:
            x_ref_ee += d['ax'] * dt
            y_ref_ee += d['ay'] * dt
            xe_ref_ee = np.hypot(xe_ref_ee, d['axe'] * dt)
            ye_ref_ee = np.hypot(ye_ref_ee, d['aye'] * dt)
            
        # Figure out the set of stars detected in both.
        idx = np.where((t_ref != 0) & (t_epo != 0) & (xe_ref != 0) & (xe_epo != 0))[0]

        # Find those in both AND reference stars. This is [idx][rdx]
        rdx = np.where((r_ref[idx] < ref_radius_lim) & (m_ref[idx] < ref_mag_lim))[0]
        
        # Average the positional errors together to get one weight per star.
        xye_ref = (xe_ref_ee + ye_ref_ee) / 2.0
        xye_epo = (xe_epo + ye_epo) / 2.0
        xye_wgt = (xye_ref**2 + xye_epo**2)**0.5
        
        # Calculate transform based on the matched stars    
        trans_tmp = transforms.PolyTransform(x_epo[idx][rdx], y_epo[idx][rdx], x_ref_ee[idx][rdx], y_ref_ee[idx][rdx],
                                                 weights=xye_wgt[idx][rdx], order=2)
        trans.append(trans_tmp)

        # Apply thte transformation to the stars positions and errors:
        xt_epo = np.zeros(len(d), dtype=float)
        yt_epo = np.zeros(len(d), dtype=float)
        xet_epo = np.zeros(len(d), dtype=float)
        yet_epo = np.zeros(len(d), dtype=float)
        
        xt_epo[idx], xet_epo[idx], yt_epo[idx], yet_epo[idx] = trans_tmp.evaluate_errors(x_epo[idx], xe_epo[idx],
                                                                                         y_epo[idx], ye_epo[idx],
                                                                                         nsim=100)
        d['xt_{0:d}'.format(ee)] = xt_epo
        d['yt_{0:d}'.format(ee)] = yt_epo
        d['xet_{0:d}'.format(ee)] = xet_epo
        d['yet_{0:d}'.format(ee)] = yet_epo

        # Record which stars we used in the transform.
        used_tmp = np.zeros(len(d), dtype=bool)
        used_tmp[idx[rdx]] = True

        used.append(used_tmp)

        if True:
            plot_quiver_residuals(xt_epo, yt_epo, x_ref_ee, y_ref_ee, idx, rdx, 'Epoch: ' + str(ee))

    used = np.array(used)
    
    return trans, used

def calc_polyfit_all_stars(d, poly_deg, init_fig_idx=0):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])
    
    # Setup some variables to save the results
    t0_all = []
    px_all = []
    py_all = []
    pxe_all = []
    pye_all = []
    chi2x_all = []
    chi2y_all = []
    dof_all = []

    # Get the time array, which is the same for all stars.
    # Also, sort the time indices.
    t = np.array([d['t_{0:d}'.format(ee)][0] for ee in range(N_epochs)])
    tdx = t.argsort()
    t_sorted = t[tdx]
    
    # Run polyfit on each star.
    for ss in range(N_stars):
        # Get the x, y, xe, ye, and t arrays for this star.
        xt = np.array([d['xt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yt = np.array([d['yt_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        xet = np.array([d['xet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        yet = np.array([d['yet_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])
        t_tmp = np.array([d['t_{0:d}'.format(ee)][ss] for ee in range(N_epochs)])

        # Sort these arrays.
        xt_sorted = xt[tdx]
        yt_sorted = yt[tdx]
        xet_sorted = xet[tdx]
        yet_sorted = yet[tdx]
        t_tmp_sorted = t_tmp[tdx]

        # Get only the detected epochs.
        edx = np.where(t_tmp_sorted != 0)[0]

        # Calculate the weighted t0 (using the transformed errors).
        weight_for_t0 = 1.0 / np.hypot(xet_sorted, yet_sorted)
        t0 = np.average(t_sorted[edx], weights=weight_for_t0[edx])

        # for ee in edx:
        #     print('{0:8.3f}  {1:10.5f}  {2:10.5f}  {3:8.5f}  {4:8.5f}'.format(t[ee], xt[ee], yt[ee], xet[ee], yet[ee]))
        # pdb.set_trace()

        # Run polyfit
        dt = t_sorted - t0
        px, covx = np.polyfit(dt[edx], xt_sorted[edx], poly_deg, w=1./xet_sorted[edx], cov=True)
        py, covy = np.polyfit(dt[edx], yt_sorted[edx], poly_deg, w=1./yet_sorted[edx], cov=True)

        pxe = np.sqrt(np.diag(covx))
        pye = np.sqrt(np.diag(covy))


        x_mod = np.polyval(px, dt[edx])
        y_mod = np.polyval(py, dt[edx])
        chi2x = np.sum( ((x_mod - xt_sorted[edx]) / xet_sorted[edx])**2 )
        chi2y = np.sum( ((y_mod - yt_sorted[edx]) / yet_sorted[edx])**2 )
        dof = len(edx) - (poly_deg + 1)

        # Save results:
        t0_all.append(t0)
        px_all.append(px)
        py_all.append(py)
        pxe_all.append(pxe)
        pye_all.append(pye)
        chi2x_all.append(chi2x)
        chi2y_all.append(chi2y)
        dof_all.append(dof)

        if d[ss]['name_0'] in ['OB120169', 'OB120169_L']:
            gs = GridSpec(3, 2) # 3 rows, 1 column
            fig = plt.figure(ss + 1 + init_fig_idx, figsize=(12, 8))
            a0 = fig.add_subplot(gs[0:2, 0])
            a1 = fig.add_subplot(gs[2, 0])
            a2 = fig.add_subplot(gs[0:2, 1])
            a3 = fig.add_subplot(gs[2, 1])
            
            a0.errorbar(t_sorted[edx], xt_sorted[edx], yerr=xet_sorted[edx], fmt='ro')
            a0.plot(t_sorted[edx], x_mod, 'k-')
            a0.set_title(d[ss]['name_0'] + ' X')
            a1.errorbar(t_sorted[edx], xt_sorted[edx] - x_mod, yerr=xet_sorted[edx], fmt='ro')
            a1.axhline(0, linestyle='--')
            a1.set_xlabel('Time (yrs)')
            a2.errorbar(t_sorted[edx], yt_sorted[edx], yerr=yet_sorted[edx], fmt='ro')
            a2.plot(t_sorted[edx], y_mod, 'k-')
            a2.set_title(d[ss]['name_0'] + ' Y')
            a3.errorbar(t_sorted[edx], yt_sorted[edx] - y_mod, yerr=yet_sorted[edx], fmt='ro')
            a3.axhline(0, linestyle='--')
            a3.set_xlabel('Time (yrs)')

            

    t0_all = np.array(t0_all)
    px_all = np.array(px_all)
    py_all = np.array(py_all)
    pxe_all = np.array(pxe_all)
    pye_all = np.array(pye_all)
    chi2x_all = np.array(chi2x_all)
    chi2y_all = np.array(chi2y_all)
    dof_all = np.array(dof_all)
        
    # Done with all the stars... recast as numpy arrays and save to output table.
    d['t0'] = t0_all
    d['chi2x'] = chi2x_all
    d['chi2y'] = chi2y_all
    d['dof'] = dof_all
    if poly_deg >= 0:
        d['x0'] = px_all[:, -1]
        d['y0'] = py_all[:, -1]
        d['x0e'] = pxe_all[:, -1]
        d['y0e'] = pye_all[:, -1]
        
    if poly_deg >= 1:
        d['vx'] = px_all[:, -2]
        d['vy'] = py_all[:, -2]
        d['vxe'] = pxe_all[:, -2]
        d['vye'] = pye_all[:, -2]

    if poly_deg >= 2:
        d['ax'] = px_all[:, -3]
        d['ay'] = py_all[:, -3]
        d['axe'] = pxe_all[:, -3]
        d['aye'] = pye_all[:, -3]

        
    return

def calc_mag_avg_all_stars(d):
    # Determine how many stars there are. 
    N_stars = len(d)

    # Determine how many epochs there are.
    N_epochs = len([n for n, c in enumerate(d.colnames) if c.startswith('name')])

    # 2D mag array
    mag_all = np.zeros((N_epochs, N_stars), dtype=float)

    for ee in range(N_epochs):
        mag_all[ee, :] = d['m_{0:d}'.format(ee)]

    mag_all_masked = np.ma.masked_equal(mag_all, 0)
    flux_all_masked = 10**(-mag_all_masked / 2.5)

    flux_avg = flux_all_masked.mean(axis=0)
    mag_avg = -2.5 * np.log10(flux_avg)

    d['mag'] = mag_avg

    return



def initial_align(table1, table2, briteN=100,
                      transformModel=transforms.PolyTransform, order=1, req_match=5):
    """
    Calculates an initial (unweighted) transformation from table1 starlist into
    table2 starlist (i.e., table2 is the reference starlist). Matching is done using
    a blind triangle-matching algorithm of the brightest briteN stars in both starlists.
    Transformation is done using the transformModel in the input parameter.

    Starlists must be astropy tables with standard column headers. All
    positions must be at the same epoch, and +x must be in the same
    direction.

    Standard column headers:
    name: name
    x: x position
    y: y position
    xe: error in x position
    ye: error in y position
    
    vx: proper motion in x direction
    vy proper motion in y direction
    vxe: error in x proper motion
    vye: error in y proper motion

    m: magnitude
    me: magnitude error
    
    t0: linear motion time zero point

    use: specify use in transformation
    

    Parameters:
    ----------
    -table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0.

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -briteN: int
        The number of brightest stars used to match two starlists.

    -transformModel:  transformation model object (class)
        The transformation model class that will be instantiated to find the
        best-fit transformation parameters between matched table1 and table2.
        eg: transforms.four_paramNW, transforms.PolyTransform

    -order: int
         Order of the transformation. Not relevant for 4 parameter or spline fit

    -req_match: int
         Number of required matches of the input catalog to the total reference

    Output:
    ------
    Transformation object
    
    """
    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']

    # Run the blind triangle-matching algorithm to find the matches between the starlists
    print(( 'Attempting match with {0} and {1} stars from starlist1 and starlist2'.format(len(table1), len(table2))))
    print( 'Begin initial match')

    N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(x1, y1, m1, x2, y2, m2, briteN)
    #assert len(x1m) > req_match, 'Failed to find at least '+str(req_match)+' (only ' + str(len(x1m)) + ') matches, giving up'
    print(( '{0} stars matched between starlist1 and starlist2'.format(N)))

    # Calculate transformation based on matches
    t = transformModel.derive_transform(x1m, y1m ,x2m, y2m, order=order, weights=None)

    print( 'End initial match \n')
    return t



def transform_and_match(table1, table2, transform, dr_tol=1.0, dm_tol=None, verbose=True):
    """
    apply transformation to starlist1 and
    match stars to given radius and magnitude tolerance.

    Starlists must be astropy tables with standard columns names as specified
    in initial_align.

    Parameters:
   -----------
    -table1: astropy.table
        contains name,m,x,y,xe,ye,vx,vy,vxe,vye,t0.

    -table2: astropy.table
        contains name,m,x,y,xe,ye.
        this is the reference template

    -dr_tol: float (default=1.0)
        The search radius for the matching algorithm, in the same units as the
        starlist file positions.

    -transform: transformation object
    
    -verbose: bool, optional
        Prints on screen information on the matching


    Output:
    -------
    -idx1: indicies of matched stars from table1
    -idx2: indicies of matched stars from tabel2
    """

    # Extract necessary information from tables (x, y, m)
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']


    # Transform x, y coordinates from starlist 1 into starlist 2
    x1t, y1t = transform.evaluate(x1, y1)

    # Match starlist 1 and 2
    idx1, idx2, dr, dm = match.match(x1t, y1t, m1, x2, y2, m2, dr_tol, dm_tol, verbose=verbose)

    if verbose:
        print(( '{0} of {1} stars matched'.format(len(idx1), len(x1t))))

    return idx1, idx2


def find_transform(table1, table1_trans, table2, transModel=transforms.PolyTransform, order=1,
                weights=None, verbose=True):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1: astropy table
        Table which we have calculated the transformation for, trimmed to only
        stars which match with table2. Original coords, not transformed into
        reference frame.

    table1_trans: astropy table
        Table which we calculated the transformation fo, trimmed to only
        stars which match with table2. Contains transformed coords. Only
        used when calculating weights.

    table2: astropy table
        Table with the reference starlist. Trimmed to only stars which
        match table1.

    trans: transformation object
        Transformation used to transform table1 coords in transform_and_match
        in order to do the star matching.

    transModel: transformation class (default: transform.four_paramNW)
        Desired transform to apply to matched stars, e.g. four_paramNW or PolyTransform.
        If PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default=None)
        if weights=='both', we use both position error in transformed starlist and
        reference starlist as uncertanty. And weights is the reciprocal of this uncertanty.
        if weights=='starlist', we only use postion error in transformed starlist.
        if weights=='reference', we only use position error in reference starlist.
        if weights==None, we don't use weights.
    
    verbose: bool (default=True)
        Prints on screen information on the matching

    Output:
    ------
    -transformation object
    -number of stars used in transform
    """
    # First, check that desired transform is supported
    if ( (transModel != transforms.four_paramNW) &
         (transModel != transforms.PolyTransform) &
         (transModel != transforms.Shift) &
         (transModel != transforms.LegTransform) ):
        print(( '{0} not supported yet!'.format(transModel)))
        return
    
    # Extract *untransformed* coordinates from starlist 1
    # and the matching coordinates from starlist 2
    x1 = table1['x']
    y1 = table1['y']
    m1 = table1['m']
    x2 = table2['x']
    y2 = table2['y']
    m2 = table2['m']

    # calculate weights from *transformed* coords. This is where we use the
    # transformation object
    if (table1_trans != None) and ('xe' in table1_trans.colnames):
        x1e = table1_trans['xe']
        y1e = table1_trans['ye']

    if 'xe' in table2.colnames:
        x2e = table2['xe']
        y2e = table2['ye']

    # Calculate weights as to user specification
    if weights=='both':
        weight = 1/np.sqrt( x1e**2 + y1e**2 + x2e**2 + y2e**2)
    elif weights=='starlist':
        weight = 1/np.sqrt( x1e**2 + y1e**2 )
    elif weights=='reference':
        weight = 1/np.sqrt( x2e**2 +  y2e**2)
    else:
        weight = None

    # Calculate transform based on the matched stars
    t = transModel.derive_transform(x1, y1, x2, y2, order, m=m1, mref=m2, weights=weight)

    N_trans = len(x1)
    if verbose:
        print(( '{0} stars used in transform\n'.format(N_trans)))

    # Ret3urn transformation object and number of stars used in transform
    return t, N_trans


def find_transform_new(table1_mat, table2_mat,
                       transModel=transforms.four_paramNW, order=1,
                       weights=None, transInit=None, verbose=True):
    """
    Given a matched starlist, derive a new transform. This transformation is
    calculated for starlist 1 into starlist 2

    Parameters:
    -----------
    table1_mat: astropy table
        Table with matched stars from starlist 1, with original positions
        (not transformed into starlist 2 frame)

    table2_mat: astropy table
        Table with matched stars from starlist 2, in starlist 2 frame.

    transModel: transformation class (default: transform.four_paramNW)
        Specify desired transform, e.g. four_paramNW or PolyTransform. If
        PolyTransform is selected, order defines the order of polynomial used

    order: int (default=1)
        Order of polynomial to use in the transformation. Only active if
        PolyTransform is selected

    weights: string (default=None)
        if weights=='both', we use position error  in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty.
        if weights==None, we don't use weights.

    transInit: Transform Object (default=None)
        if weights = 'both' or 'starlist' then the positions in table 1 are first transformed
        using the transInit object. This is necessary if the plate scales are very different
        between the table 1 and the reference list.
    
    verbose: bool (default=True)
        Prints on screen information on the matching

    Output:
    ------
    -transformation object
    -number of stars used in transform
    """
    # First, check that desired transform is supported
    if ( (transModel != transforms.four_paramNW) & (transModel != transforms.PolyTransform) ):
        print(( '{0} not supported yet!'.format(transModel)))
        return
    
    # Extract *untransformed* coordinates from starlist 1
    # and the matching coordinates from starlist 2
    x1 = table1_mat['x']
    y1 = table1_mat['y']
    x2 = table2_mat['x']
    y2 = table2_mat['y']

    # Get the uncertainties (if needed) and calculate the weights.
    if weights != None:
        x1e = table1_mat['xe']
        y1e = table1_mat['ye']
        x2e = table2_mat['xe']
        y2e = table2_mat['ye']

        if transInit != None:
            table1T_mat = table1_mat.copy()
            table1T_mat = transform_by_object(table1T_mat, transInit)

            x1e = table1T_mag['xe']
            y1e = table1T_mag['ye']

        # Calculate weights as to user specification
        if weights == 'both':
            weight = 1.0 / ( x1e**2 + x2e**2 + y1e**2 + y2e**2 )
        elif weights == 'starlist':
            weight = 1.0 / ( x1e**2 + y1e**2 )
        elif weights == 'reference':
            weight = 1.0 / ( x2e**2 +  y2e**2 )
        else:
            weight = None

    # Calculate transform based on the matched stars
    t = transModel(x1, y1, x2, y2, order=order, weights=weight)

    N_trans = len(x1)
    if verbose:
        print(( '{0} stars used in transform\n'.format(N_trans)))

    # Return transformation object and number of stars used in transform
    return t, N_trans


def write_transform(transform, starlist, reference, N_trans, deltaMag=0, restrict=False, weights=None,
                    outFile='outTrans.txt'):
    """
    Given a transformation object, write out the coefficients in a java align
    readable format. Outfile name is specified by user.

    Coefficients are output in file in the following way:
    x' = a0 + a1*x + a2*y + a3*x**2. + a4*x*y  + a5*y**2. + ...
    y' = b0 + b1*x + b2*y + b3*x**2. + b4*x*y  + b5*y**2. + ...

    Parameters:
    ----------
    transform: transformation object
        Transformation object we want to feed into java align

    starlist: string
        File name of starlist; this is the starlist the transformation should
        be applied to. For output purposes only

    reference: string
        File name of reference; this is what the starlist is transformed to.
        For output purposes only

    N_trans: int
        Number of stars used in the transformation

    deltaMag: float (default = 0)
        Average magnitude difference between reference and starlist
        (reference - starlist)

    restrict: boolean (default=False)
        Set to True if transformation restricted to stars with use > 2. Purely
        for output purposes

    weights: string (default=None)
        if weights=='both', we use both position error and velocity error in transformed
        starlist and reference starlist as uncertanties. And weights is the reciprocal
            of this uncertanty.
        if weights=='starlist', we only use postion error and velocity error in transformed
        starlist as uncertainty.
        if weights=='reference', we only use position error in reference starlist as uncertainty
        if weights==None, we don't use weights.

    outFile: string (default: 'outTrans.txt')
        Name of output text file
        
    Output:
    ------
    txt file with the file name outFile
    """
    # Extract info about transformation
    trans_name = transform.__class__.__name__
    trans_order = transform.order
    
    # Extract X, Y coefficients from transform
    if trans_name == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
    elif trans_name == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
    else:
        print(( '{0} not yet supported!'.format(transType)))
        return
        
    # Write output
    _out = open(outFile, 'w')
    
    # Write the header. DO NOT CHANGE, HARDCODED IN JAVA ALIGN
    _out.write('## Date: {0}\n'.format(datetime.date.today()) )
    _out.write('## File: {0}, Reference: {1}\n'.format(starlist, reference) )
    _out.write('## Directory: {0}\n'.format(os.getcwd()) )
    _out.write('## Transform Class: {0}\n'.format(transform.__class__.__name__))
    _out.write('## Order: {0}\n'.format(transform.order))
    _out.write('## Restrict: {0}\n'.format(restrict))
    _out.write('## Weights: {0}\n'.format(weights))
    _out.write('## N_coeff: {0}\n'.format(len(Xcoeff)))
    _out.write('## N_trans: {0}\n'.format(N_trans))
    _out.write('## Delta Mag: {0}\n'.format(deltaMag))
    _out.write('{0:16s} {1:16s}\n'.format('# Xcoeff', 'Ycoeff'))
    
    # Write the coefficients such that the orders are together as defined in
    # documentation. This is a pain because PolyTransform output is weird.
    # (see astropy Polynomial2D documentation)
    if (trans_name == 'four_paramNW'):
        for i in range(len(Xcoeff)):
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )
    elif (trans_name == 'PolyTransform'):
        # CODE TO GET INDICIES
        N = trans_order - 1
        idx_list = list()
        
        # when trans_order=1, N=0
        idx_list.append(0)
        idx_list.append(1)
        idx_list.append(N+2)
        
        if trans_order >= 2:
            for k in range(2, N+2):
                idx_list.append(k)
                for j in range(1, k):
                    i = k-j
                    idx_list.append(int(2*N +2 +j + (2*N+2-i)*(i-1)/2.))
                idx_list.append(N+1+k)

        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[0], Ycoeff[0]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[1], Ycoeff[1]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[3], Ycoeff[3]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[2], Ycoeff[2]) )
        #_out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[5], Ycoeff[5]) )
        #_out.write('{0:16.6e}  {1:16.6e}'.format(Xcoeff[4], Ycoeff[4]) )

        for i in idx_list:
            _out.write('{0:16.6e}  {1:16.6e}\n'.format(Xcoeff[i], Ycoeff[i]) )


    _out.close()
    
    return


def transform_from_file(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers

    transFile: ascii file
        File with the transformation coefficients. Assumed to be output of
        write_transform, with coefficients specified as code documents

    Output:
    ------
    Copy of starlist astropy table with transformed coordinates.
    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)
    
    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = False
    keys = list(starlist.keys())
    if 'vx' in keys:
        vel = True
    
    # Extract needed information from starlist
    x_orig = starlist['x']
    y_orig = starlist['y']
    xe_orig = starlist['xe']
    ye_orig = starlist['ye']

    if vel:
        x0_orig = starlist['x0']
        y0_orig = starlist['y0']
        x0e_orig = starlist['x0e']
        y0e_orig = starlist['y0e']
        
        vx_orig = starlist['vx']
        vy_orig = starlist['vy']
        vxe_orig = starlist['vxe']
        vye_orig = starlist['vye']
    
    # Read transFile
    trans = Table.read(transFile, format='ascii.commented_header', header_start=-1)
    Xcoeff = trans['Xcoeff']
    Ycoeff = trans['Ycoeff']

    #-----------------------------------------------#
    # General equation for applying the transform
    #-----------------------------------------------#
    #"""
    # First determine the order based on the number of terms
    # Comes from Nterms = (N+1)*(N+2) / 2.
    order = (np.sqrt(1 + 8*len(Xcoeff)) - 3) / 2.

    if order%1 != 0:
        print( 'Incorrect number of coefficients for polynomial')
        print( 'Stopping')
        return
    order = int(order)

    # Position transformation
    x_new, y_new = transform_pos_from_file(Xcoeff, Ycoeff, order, x_orig,
                                           y_orig)
    
    if vel:
        x0_new, y0_new = transform_pos_from_file(Xcoeff, Ycoeff, order, x0_orig,
                                             y0_orig)

    # Position error transformation
    xe_new, ye_new = transform_poserr_from_file(Xcoeff, Ycoeff, order, xe_orig,
                                                ye_orig, x_orig, y_orig)

    if vel:
        x0e_new, y0e_new = transform_poserr_from_file(Xcoeff, Ycoeff, order, x0e_orig,
                                                y0e_orig, x0_orig, y0_orig)

    if vel:
        # Velocity transformation
        vx_new, vy_new = transform_vel_from_file(Xcoeff, Ycoeff, order, vx_orig,
                                                 vy_orig, x_orig, y_orig)
            
        # Velocity error transformation
        vxe_new, vye_new = transform_velerr_from_file(Xcoeff, Ycoeff, order,
                                                      vxe_orig, vye_orig,
                                                      vx_orig, vy_orig,
                                                      xe_orig, ye_orig,
                                                      x_orig, y_orig)

    #----------------------------------------#
    # Hard coded example: old but functional
    #----------------------------------------#
    """
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    if len(Xcoeff) == 3:
        x_new = Xcoeff[0] + Xcoeff[1] * x_orig + Xcoeff[2] * y_orig
        y_new = Ycoeff[0] + Ycoeff[1] * x_orig + Ycoeff[2] * y_orig
        xe_new = np.sqrt( (Xcoeff[1] * xe_orig)**2 + (Xcoeff[2] * ye_orig)**2 )
        ye_new = np.sqrt( (Ycoeff[1] * xe_orig)**2 + (Ycoeff[2] * ye_orig)**2 )

        if vel:
            vx_new = Xcoeff[1] * vx_orig + Xcoeff[2] * vy_orig
            vy_new = Ycoeff[1] * vx_orig + Ycoeff[2] * vy_orig
            vxe_new = np.sqrt( (Xcoeff[1] * vxe_orig)**2 + (Xcoeff[2] * vye_orig)**2 )
            vye_new = np.sqrt( (Ycoeff[1] * vxe_orig)**2 + (Ycoeff[2] * vye_orig)**2 )

    elif len(Xcoeff) == 6:
        x_new = Xcoeff[0] + Xcoeff[1]*x_orig + Xcoeff[3]*x_orig**2 + Xcoeff[2]*y_orig + \
                Xcoeff[5]*y_orig**2. + Xcoeff[4]*x_orig*y_orig
          
        y_new = Ycoeff[0] + Ycoeff[1]*x_orig + Ycoeff[3]*x_orig**2 + Ycoeff[2]*y_orig + \
                Ycoeff[5]*y_orig**2. + Ycoeff[4]*x_orig*y_orig
          
        xe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * ye_orig**2 )
          
        ye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * xe_orig**2 + \
                          (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * ye_orig**2 )

        if vel:
            vx_new = Xcoeff[1]*vx_orig + 2*Xcoeff[3]*x_orig*vx_orig + Xcoeff[2]*vy_orig + \
                    2.*Xcoeff[5]*y_orig*vy_orig + Xcoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vy_new = Ycoeff[1]*vx_orig + 2*Ycoeff[3]*x_orig*vx_orig + Ycoeff[2]*vy_orig + \
                    2.*Ycoeff[5]*y_orig*vy_orig + Ycoeff[4]*(x_orig*vy_orig + vx_orig*y_orig)
          
            vxe_new = np.sqrt( (Xcoeff[1] + 2*Xcoeff[3]*x_orig + Xcoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Xcoeff[2] + 2*Xcoeff[5]*y_orig + Xcoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Xcoeff[3]*vx_orig + Xcoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Xcoeff[5]*vy_orig + Xcoeff[4]*vx_orig)**2 * ye_orig**2 )
                               
            vye_new = np.sqrt( (Ycoeff[1] + 2*Ycoeff[3]*x_orig + Ycoeff[4]*y_orig)**2 * vxe_orig**2 + \
                               (Ycoeff[2] + 2*Ycoeff[5]*y_orig + Ycoeff[4]*x_orig)**2 * vye_orig**2 + \
                               (2*Ycoeff[3]*vx_orig + Ycoeff[4]*vy_orig)**2 * xe_orig**2 + \
                               (2*Ycoeff[5]*vy_orig + Ycoeff[4]*vx_orig)**2 * ye_orig**2 )
    """
    #Update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0e'] = x0e_new
        starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new

    return starlist_f


def transform_from_object(starlist, transform):
    """
    Apply transformation to starlist. Returns astropy table with
    transformed positions/position errors, velocities and velocity errors
    if they are present in starlits
    
    Parameters:
    ----------
    starlist: astropy table
         Starlist we want to apply the transformation too. Must already
         have standard column headers
         x0, y0, x0e, y0e, vx, vy, vxe, vye, x, y, xe, ye

    transform: transformation object

    Output:
    ------
    Copy of starlist astropy table with transformed x0, y0, x0e, y0e,
    vx, vy, vxe, vye, x, y, xe, ye

    """
    # Make a copy of starlist. This is what we will eventually modify with
    # the transformed coordinates
    starlist_f = copy.deepcopy(starlist)
    keys = list(starlist.keys())

    # Check to see if velocities are present in starlist. If so, we will
    # need to transform these as well as positions
    vel = 'vx' in keys
    err = 'xe' in keys
    
    # Extract needed information from starlist
    x = starlist_f['x']
    y = starlist_f['y']

    if err:
        xe = starlist_f['xe']
        ye = starlist_f['ye']

    if vel:
        x0 = starlist_f['x0']
        y0 = starlist_f['y0']
        x0e = starlist_f['x0e']
        y0e = starlist_f['y0e']
        vx = starlist_f['vx']
        vy = starlist_f['vy']
        vxe = starlist_f['vxe']
        vye = starlist_f['vye']
    
    # calculate the transformed position and velocity
    
    # (x_new, y_new, xe_new, ye_new) in (x,y)
    x_new, y_new, xe_new, ye_new = position_transform_from_object(x, y, xe, ye, transform)

    
    if vel:
        # (x0_new,  y0_new, x0e_new, y0e_new) in (x0, y0, x0e, y0e)
        x0_new, y0_new, x0e_new, y0e_new = position_transform_from_object(x0, y0, x0e, y0e, transform)
        # (vx_new, vy_new, vxe_new, vye_new) in (x0, y0, x0e, y0e, vx, vy, vxe, vye)
        vx_new, vy_new, vxe_new, vye_new = velocity_transform_from_object(x0, y0, x0e, y0e, vx, vy, vxe, vye, transform)

    # update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0e'] = x0e_new
        starlist_f['y0e'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vxe'] = vxe_new
        starlist_f['vye'] = vye_new
        
    return starlist_f





def position_transform_from_object(x, y, xe, ye, transform):
    """
    given the orginal position and position error, calculate the transformed
    position and position error based on transformation object from
    astropy.modeling.models.polynomial2D.
    Input:
        - x, y: original position
        - xe, ye: original position error
        - transform: transformation object from astropy.modeling.models.polynomial2D
        
    Outpus:
        - x_new, y_new: transformed position
        - xe_new, ye_new: transformed position error
    """

    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
        order = 1
    elif (transform.__class__.__name__ == 'PolyTransform'):
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        order = transform.order
    else:
        txt = 'Transform not yet supported by position_transform_from_object'
        raise StandardError(txt)
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    N = order - 1

    x_new = 0
    for i in range(0, N+2):
        x_new += Xcoeff[i] * (x**i)
    for j in range(1, N+2):
        x_new += Xcoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            x_new += Xcoeff[sub] * (x**i) * (y**j)

    y_new = 0
    for i in range(0, N+2):
        y_new += Ycoeff[i] * (x**i)
    for j in range(1, N+2):
        y_new += Ycoeff[N+1+j] * (y**j)
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            y_new += Ycoeff[sub] * (x**i) * (y**j)

            
    """
    THIS IS WRONG BELOW!

    Currently doing:
    ((A + B + C) * xe)**2

    Should be doing:
    ((A**2 + B**2 + C**2) * xe**2)
    """
            
    # xe_new & ye_new in (x,y,xe,ye)
    xe_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Xcoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp1 += i * Xcoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Xcoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp2 += j * Xcoeff[sub] * (x**i) * (y**(j-1))
    xe_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)

    ye_new = 0
    temp1 = 0
    temp2 = 0
    for i in range(1, N+2):
        temp1 += i * Ycoeff[i] * (x**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp1 += i * Ycoeff[sub] * (x**(i-1)) * (y**j)
    for j in range(1, N+2):
        temp2 += j * Ycoeff[N+1+j] * (y**(j-1))
    for i in range(1, N+1):
        for j in range(N+2-i):
            sub = int(2*N + 2 + j + (2*N+2-i) * (i-1)/2.)
            temp2 += j * Ycoeff[sub] * (x**i) * (y**(j-1))
    ye_new = np.sqrt((temp1*xe)**2 + (temp2*ye)**2)

    return x_new, y_new, xe_new, ye_new


def velocity_transform_from_object(x0, y0, x0e, y0e, vx, vy, vxe, vye, transform):
    """
    given the orginal position & position error & velocity & veolicty error,
    calculat the transformed velocity and velocity error based on transformation
    from astropy.modling.models.polynomial2D.
    Input:
        - x0, y0, x0e, y0e: original position and position error
        - vx, vy, vxe, vye: original velocity and velocity error
        - transform: transformation object from astropy.modeling.models.polynomial2D
        
    Outpus:
        - vx_new, vy_new, vxe_new, vye_new: transformed velocity and velocity error
    """
 
    # Read transformation: Extract X, Y coefficients from transform
    if transform.__class__.__name__ == 'four_paramNW':
        Xcoeff = transform.px
        Ycoeff = transform.py
        order = 1
    elif transform.__class__.__name__ == 'PolyTransform':
        Xcoeff = transform.px.parameters
        Ycoeff = transform.py.parameters
        order = transform.order
    else:
        txt = 'Transform not yet supported by velocity_transform_from_object'
        raise StandardError(txt)
        
    # How the transformation is applied depends on the type of transform.
    # This can be determined by the length of Xcoeff, Ycoeff
    N = order - 1

    # vx_new & vy_new
    vx_new = 0
    for i in range(1, N+2):
        vx_new += i * Xcoeff[i] * (x0**(i-1)) * vx
    for j in range(1, N+2):
        vx_new += j * Xcoeff[N+1+j] * (y0**(j-1)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            vx_new += i * Xcoeff[int(sub)] * (x0**(i-1)) * (y0**j) * vx
            vx_new += j * Xcoeff[int(sub)] * (x0**i) * (y0**(j-1)) * vy

    vy_new = 0
    for i in range(1, N+2):
        vy_new += i * Ycoeff[i] * (x0**(i-1)) * vx
    for j in range(1, N+2):
        vy_new += j * Ycoeff[N+1+j] * (y0**(j-1)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            vy_new += i * Ycoeff[int(sub)] * (x0**(i-1)) * (y0**j) * vx
            vy_new += j * Ycoeff[int(sub)] * (x0**i) * (y0**(j-1)) * vy

    # vxe_new & vye_new
    vxe_new = 0
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i in range(2, N+2):
        temp1 += i * (i-1) * Xcoeff[i] * (x0**(i-2)) * vx
    for i in range(2, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * (i-1) * Xcoeff[int(sub)] * (x0**(i-2)) * (y0**j) * vx
    for i in range(1,N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * j * Xcoeff[int(sub)] * (x0**(i-1)) * (y0**(j-1)) * vy

    for j in range(2, N+2):
        temp2 += j * (j-1) * Xcoeff[N+1+j] * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += i * j * Xcoeff[int(sub)] * (x0**(i-1)) * (y0**(j-1)) * vx
    for i in range(1, N+1):
        for j in range(2, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * (j-1) * Xcoeff[int(sub)] * (x0**i) * (y0**(j-2)) * vy

    for i in range(1, N+2):
        temp3 += i * Xcoeff[i] * (x0**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp3 += i * Xcoeff[int(sub)] * (x0**(i-1)) * (y0**j) 

    for j in range(1, N+2):
        temp4 += j * Xcoeff[N+1+j] * (y0**(j-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp4 += j * Xcoeff[int(sub)] * (x0**i) * (y0**(j-1))

    vxe_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)


    vye_new = 0
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i in range(2, N+2):
        temp1 += i * (i-1) * Ycoeff[i] * (x0**(i-2)) * vx
    for i in range(2, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += i * (i-1) * Ycoeff[int(sub)] * (x0**(i-2)) * (y0**j) * vx
    for i in range(1,N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp1 += j * i * Ycoeff[int(sub)] * (x0**(i-1)) * (y0**(j-1)) * vy

    for j in range(2, N+2):
        temp2 += j * (j-1) * Ycoeff[N+1+j] * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(2, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += j * (j-1) * Ycoeff[int(sub)] * (x0**i) * (y0**(j-2)) * vy
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp2 += i * j * Ycoeff[int(sub)] * (x0**(i-1)) * (y0**(j-1)) * vx

    for i in range(1, N+2):
        temp3 += i * Ycoeff[i] * (x0**(i-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp3 += i * Ycoeff[int(sub)] * (x0**(i-1)) * (y0**j) 

    for j in range(1, N+2):
        temp4 += j * Ycoeff[N+1+j] * (y0**(j-1))
    for i in range(1, N+1):
        for j in range(1, N+2-i):
            sub = 2*N + 2 + j + (2*N+2-i) * (i-1)/2.
            temp4 += j * Ycoeff[int(sub)] * (x0**i) * (y0**(j-1))

    vye_new = np.sqrt((temp1*x0e)**2 + (temp2*y0e)**2 + (temp3*vxe)**2 + (temp4*vye)**2)

    return vx_new, vy_new, vxe_new, vye_new


def transform_pos_from_file(Xcoeff, Ycoeff, order, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed positions. This is generalized to
    work with any order polynomial transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS

    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    x_new: array
       Transformed X positions

    y_new: array
        Transformed Y positions

    """
    idx = 0 # coeff index
    x_new = 0.0
    y_new = 0.0
    for i in range(order+1):
        for j in range(i+1):
            x_new += Xcoeff[idx] * x_orig**(i-j) * y_orig**j
            y_new += Ycoeff[idx] * x_orig**(i-j) * y_orig**j

            idx += 1

    return x_new, y_new

def transform_poserr_from_file(Xcoeff, Ycoeff, order, xe_orig, ye_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed position errors. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS

    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    xe_orig: array
        Array with the original X position errs

    ye_orig: array
        Array with the original Y position errs
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    xe_new: array
       Transformed X position errs

    ye_new: array
        Transformed Y position errs
    """
    idx = 0 # coeff index
    xe_new_tmp1 = 0.0
    ye_new_tmp1 = 0.0
    xe_new_tmp2 = 0.0
    ye_new_tmp2 = 0.0
    
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            xe_new_tmp1 += Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j
            ye_new_tmp1 += Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j

            idx += 1
            
    # Second loop: dy'/dy
    idx = 0 # coeff index
    for i in range(order+1):
        for j in range(i+1):
            xe_new_tmp2 += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)
            ye_new_tmp2 += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)

            idx += 1
    # Take square root for xe/ye_new
    xe_new = np.sqrt((xe_new_tmp1 * xe_orig)**2 + (xe_new_tmp2 * ye_orig)**2)
    ye_new = np.sqrt((ye_new_tmp1 * ye_orig)**2 + (ye_new_tmp2 * ye_orig)**2)

    return xe_new, ye_new

def transform_vel_from_file(Xcoeff, Ycoeff, order, vx_orig, vy_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed proper motions. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    vx_orig: array
        Array with the original X proper motions

    vy_orig: array
        Array with the original Y proper motions
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    vx_new: array
       Transformed X proper motions

    vy_new: array
        Transformed Y proper motions
    """
    idx = 0 # coeff index
    vx_new = 0.0
    vy_new = 0.0
    # First loop: dx'/dx
    for i in range(order+1):
        for j in range(i+1):
            vx_new += Xcoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig
            vy_new += Ycoeff[idx] * (i - j) * x_orig**(i-j-1) * y_orig**j * vx_orig
            
            idx += 1
    # Second loop: dy'/dy
    idx = 0 # coeff index
    for i in range(order+1):
        for j in range(i+1):
            vx_new += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * vy_orig
            vy_new += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1) * vy_orig

            idx += 1

    return vx_new, vy_new

def transform_velerr_from_file(Xcoeff, Ycoeff, order, vxe_orig, vye_orig, vx_orig,
                                vy_orig, xe_orig, ye_orig, x_orig, y_orig):
    """
    Given the read-in coefficients from transform_from_file, apply the
    transformation to the observed proper motion errors. This is generalized to
    work with any order transform.

    WARNING: THIS CODE WILL NOT WORK FOR LEGENDRE POLYNOMIAL
    TRANSFORMS
    
    Parameters:
    ----------
    Xcoeff: Array
        Array with the coefficients of the X pos transformation

    Ycoeff: Array
        Array with the coefficients of the Y pos transformation

    order: int
        Order of transformation

    vxe_orig: array
        Array with the original X proper motion errs

    vye_orig: array
        Array with the original Y proper motion errs
        
    vx_orig: array
        Array with the original X proper motions

    vy_orig: array
        Array with the original Y proper motions

    xe_orig: array
        Array with the original X position errs

    ye_orig: array
        Array with the original Y position errs
        
    x_orig: array
        Array with the original X positions

    y_orig: array
        Array with the original Y positions

    Output:
    ------
    vxe_new: array
       Transformed X proper motion errs

    vye_new: array
        Transformed Y proper motion errs
    """
    idx = 0
    vxe_new_tmp1 = 0.0
    vye_new_tmp1 = 0.0
    vxe_new_tmp2 = 0.0
    vye_new_tmp2 = 0.0
    vxe_new_tmp3 = 0.0
    vye_new_tmp3 = 0.0
    vxe_new_tmp4 = 0.0
    vye_new_tmp4 = 0.0

    
    # First loop: dvx' / dx
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp1 += Xcoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig
            vxe_new_tmp1 += Xcoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig
            vye_new_tmp1 += Ycoeff[idx] * (i-j) * (i-j-1) * x_orig**(i-j-2) * y_orig**j * vx_orig
            vye_new_tmp1 += Ycoeff[idx] * (j) * (i-j) * x_orig**(i-j-1) * y_orig**(j-1) * vy_orig

            idx += 1

    # Second loop: dvx' / dy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp2 += Xcoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig
            vxe_new_tmp2 += Xcoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig
            vye_new_tmp2 += Ycoeff[idx] * (i-j) * (j) * x_orig**(i-j-1) * y_orig**(j-1) * vx_orig
            vye_new_tmp2 += Ycoeff[idx] * (j) * (j-1) * x_orig**(i-j-1) * y_orig**(j-2) * vy_orig

            idx += 1

    # Third loop: dvx' / dvx
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp3 += Xcoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j
            vye_new_tmp3 += Ycoeff[idx] * (i-j) * x_orig**(i-j-1) * y_orig**j

            idx += 1

    # Fourth loop: dvx' / dvy
    idx = 0
    for i in range(order+1):
        for j in range(i+1):
            vxe_new_tmp4 += Xcoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)
            vye_new_tmp4 += Ycoeff[idx] * (j) * x_orig**(i-j) * y_orig**(j-1)

            idx += 1

    vxe_new = np.sqrt((vxe_new_tmp1 * xe_orig)**2 + (vxe_new_tmp2 * ye_orig)**2 + \
                      (vxe_new_tmp3 * vxe_orig)**2 + (vxe_new_tmp4 * vye_orig)**2)
    vye_new = np.sqrt((vye_new_tmp1 * xe_orig)**2 + (vye_new_tmp2 * ye_orig)**2 + \
                      (vye_new_tmp3 * vxe_orig)**2 + (vye_new_tmp4 * vye_orig)**2)

    return vxe_new, vye_new


def check_iter_tolerances(iters, dr_tol, dm_tol, outlier_tol):
    # iteration tolerances must match the number of iterations requested.
    assert iters == len(dr_tol)
    assert iters == len(dm_tol)
    assert iters == len(outlier_tol)
    return

def check_trans_input(list_of_starlists, trans_input, mag_trans):
    # Check trans_input
    # If we are transforming magnitudes and their are input transformations,
    # then they need to have a mag_offset on them.
    if trans_input != None:
        assert len(trans_input) == len(list_of_starlists)

        if mag_trans: 
            for ii in range(len(trans_input)):
                if trans_input[ii] != None:
                    try:
                        trans_input[ii].mag_offset
                    except NameError:
                        print('Missing trans.mag_offset on trans_input[{0:d}].'.format(ii))
                        print('Setting mag_offset = 0 and dm_tol[0] = 100 and hoping for the best!!')
                        trans_input[ii].mag_offset = 0.0
                
    return

def trans_initial_guess(ref_list, star_list, trans_args, mode='miracle',
                        ignore_contains='star', verbose=True, n_req_match=2):
    """
    Take two starlists and perform an initial matching and transformation.

    This function will grow with time to handle difference types of initial
    guess transformations (triangle matching, match by name, etc.). For now it
    is just blind triangle matching on the brightest 50 stars. 
    """
    warnings.filterwarnings('ignore', category=AstropyUserWarning)
    
    if mode == 'name':
        # First trim the two lists down to only those that don't contain
        # the "ignore_contains" string.
        idx_r = np.flatnonzero(np.char.find(ref_list['name'], ignore_contains) == -1)
        idx_s = np.flatnonzero(np.char.find(star_list['name'], ignore_contains) == -1)

        # Match the star names
        name_matches, ndx_r, ndx_s = np.intersect1d(ref_list['name'][idx_r],
                                                    star_list['name'][idx_s],
                                                    assume_unique=True,
                                                    return_indices=True)
        
        x1m = star_list['x'][idx_s][ndx_s]
        y1m = star_list['y'][idx_s][ndx_s]
        m1m = star_list['m'][idx_s][ndx_s]
        x2m = ref_list['x0'][idx_r][ndx_r]
        y2m = ref_list['y0'][idx_r][ndx_r]
        m2m = ref_list['m0'][idx_r][ndx_r]
        N = len(x1m)

        # # If there are velocities in the reference list, use them.
        # # We assume velocities are in the same units as the positions.
        # # We also assume that all stars in the star_list have a fixed t.
        # x2m, y2m = get_pos_at_time(star_list['t'][0], ref_list[idx_r])
        
    else:
        # Default is miracle match.
        briteN = np.min([50, len(star_list), len(ref_list)])

        # If there are velocities in the reference list, use them.
        # We assume velocities are in the same units as the positions.
        xref, yref = get_pos_at_time(star_list['t'][0], ref_list)
        mref = ref_list['m0']
            

        N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(star_list['x'],
                                                                     star_list['y'],
                                                                     star_list['m'],
                                                                     xref,
                                                                     yref,
                                                                     mref,
                                                                     briteN)

        
    err_msg = 'Failed to find more than '+str(n_req_match)
    err_msg += ' (only ' + str(len(x1m)) + ') matches, giving up.'
    assert len(x1m) > n_req_match, err_msg
    if verbose:
        print('initial_guess: {0:d} stars matched between starlist and reference list'.format(N))

    # Calculate position transformation based on matches
    if ('order' in trans_args) and (trans_args['order'] == 0):
        order = 0
    else:
        order = 1
    trans = transforms.PolyTransform.derive_transform(x1m, y1m ,x2m, y2m, order=order, weights=None)

    # Calculate flux transformation based on matches. Should be applied as
    #     m' = m + mag_offset
    # where m is the original magnitude and m' is in the reference frame mag system.
    trans.mag_offset = np.mean(m2m - m1m)

    if verbose:
        print('initial_guess: ', trans.px.parameters, trans.py.parameters, trans.mag_offset)

    warnings.filterwarnings('default', category=AstropyUserWarning)
        
    return trans

def trans_initial_guess_new(ref_list, star_list, trans_args, mode='miracle',
                            ignore_contains='star', verbose=True, n_req_match=2):
    """
    Take two starlists and perform an initial matching and transformation.

    This function will grow with time to handle difference types of initial
    guess transformations (triangle matching, match by name, etc.). For now it
    is just blind triangle matching on the brightest 50 stars. 
    """
    warnings.filterwarnings('ignore', category=AstropyUserWarning)
    
    if mode == 'name':
        # First trim the two lists down to only those that don't contain
        # the "ignore_contains" string.
        idx_r = np.flatnonzero(np.char.find(ref_list['name'], ignore_contains) == -1)
        idx_s = np.flatnonzero(np.char.find(star_list['name'], ignore_contains) == -1)
        
        # Match the star names
        name_matches, ndx_r, ndx_s = np.intersect1d(ref_list['name'][idx_r],
                                                    star_list['name'][idx_s],
                                                    assume_unique=False,
                                                    return_indices=True)
        
        x1m = star_list['x'][idx_s][ndx_s]
        y1m = star_list['y'][idx_s][ndx_s]
        m1m = star_list['m'][idx_s][ndx_s]
        x2m = ref_list['x'][idx_r][ndx_r]
        y2m = ref_list['y'][idx_r][ndx_r]
        m2m = ref_list['m'][idx_r][ndx_r]
        N = len(x1m)

    else:
        # Default is miracle match.
        briteN = np.min([50, len(star_list), len(ref_list)])

        # If there are velocities in the reference list, use them.
        # We assume velocities are in the same units as the positions.
        N, x1m, y1m, m1m, x2m, y2m, m2m = match.miracle_match_briteN(star_list['x'],
                                                                     star_list['y'],
                                                                     star_list['m'],
                                                                     ref_list['x'],
                                                                     ref_list['y'],
                                                                     ref_list['m'],
                                                                     briteN)


        
    err_msg = 'Failed to find more than '+str(n_req_match)
    err_msg += ' (only ' + str(len(x1m)) + ') matches, giving up.'
    assert len(x1m) >= n_req_match, err_msg
    if verbose:
        print('initial_guess: {0:d} stars matched between starlist and reference list'.format(N))

    # Calculate position transformation based on matches
    if ('order' in trans_args) and (trans_args['order'] == 0):
        order = 0
    else:
        order = 1

    trans = transforms.PolyTransform.derive_transform(x1m, y1m, x2m, y2m, order=order, weights=None)

    # Calculate flux transformation based on matches. Should be applied as
    #     m' = m + mag_offset
    # where m is the original magnitude and m' is in the reference frame mag system.
    trans.mag_offset = np.mean(m2m - m1m)

    if verbose:
        x1m_t, y1m_t = trans.evaluate(x1m, y1m)
        m1m_t = trans.evaluate_mag(m1m)
        
        hdr = '{0:>7s} {1:>7s} {2:>7s}   {3:>7s} {4:>7s} {5:>7s}   {6:>7s} {7:>7s} {8:>7s}'
        print(hdr.format('lis_x_t', 'ref_x', 'dx', 'lis_y_t', 'ref_y', 'dy', 'lis_m_t', 'ref_m', 'dm'))
        for ii in range(len(x1m)):
            fmt = '{0:7.3f} {1:7.3f} {2:7.3f}   {3:7.3f} {4:7.3f} {5:7.3f}   {6:7.2f} {7:7.2f} {8:7.2f}'
            print(fmt.format(x1m_t[ii], x2m[ii], x1m_t[ii] - x2m[ii],
                             y1m_t[ii], y2m[ii], y1m_t[ii] - y2m[ii],
                             m1m_t[ii], m2m[ii], m1m_t[ii] - m2m[ii]))

        print('initial_guess: ', trans.px.parameters, trans.py.parameters, trans.mag_offset)
        
        
    warnings.filterwarnings('default', category=AstropyUserWarning)   
    return trans


def update_old_and_new_names(ref_table, list_index, idx_ref_new):
    # Make new ref_list names for the new stars.
    new_names = []
    new_name_len_max = 0

    for ss in idx_ref_new:
        new_name = '{0:3d}_{1:s}'.format(list_index, ref_table['name_in_list'][ss, list_index])
        new_names.append(new_name)
        new_name_len_max = max(new_name_len_max, len(new_name))

    old_names = ref_table['name']
    old_name_len = [len(old_name) for old_name in old_names]
    old_name_len_max = np.max(old_name_len)

    if new_name_len_max > old_name_len_max:
        all_names = old_names.astype('U{0:d}'.format(new_name_len_max))
    else:
        all_names = old_names
    
    all_names[idx_ref_new] = new_names
    
    return all_names

def copy_and_rename_for_ref(star_list):
    """
    Make a deep copy of the starlist and rename the columns to include
    "0". This only applies to x, y, m and xe, ye, me (if they exist) 
    columns.

    Input
    ----------
    star_list : StarList
        The starlist to copy.
    """
    old_cols = ['x', 'y', 'm']
    new_cols = ['x0', 'y0', 'm0']

    if 'xe' in star_list.colnames:
        old_cols += ['xe']
        new_cols += ['x0e']
    if 'ye' in star_list.colnames:
        old_cols += ['ye']
        new_cols += ['y0e']
    if 'me' in star_list.colnames:
        old_cols += ['me']
        new_cols += ['m0e']
    if 'w' in star_list.colnames:
        old_cols += ['w']
        new_cols += ['w']
        
    ref_list = copy.deepcopy(star_list)

    for ii in range(len(old_cols)):
        ref_list.rename_column(old_cols[ii], new_cols[ii])

    return ref_list

def outlier_rejection_indices(star_list, ref_list, outlier_tol, verbose=True):
    """
    Determine the outliers based on the residual positions between two different
    starlists and some threshold (in sigma). Return the indices of the stars 
    to keep (that shouldn't be rejected as outliers). 

    Note that we assume that the star_list and ref_list are already transformed and
    matched. 

    Parameters
    ----------
    star_list : StarList
        starlist with 'x', 'y'

    ref_list : StarList
        starlist with 'x0', 'y0'

    outlier_tol : float
        Number of sigma inside which we keep stars and outside of which we 
        reject stars as outliers. 

    Optional Parameters
    --------------------
    verbose : boolean

    Returns
    ----------
    keepers : nd.array
        The indicies of the stars to keep. 
    """
    # Optionally propogate the reference positions forward in time.
    xref, yref = get_pos_in_time(star_list['t'][0], ref_list)
    
    # Residuals
    x_resid_on_old_trans = star_list['x'] - xref
    y_resid_on_old_trans = star_list['y'] - yref
    resid_on_old_trans = np.hypot(x_resid_on_old_trans, y_resid_on_old_trans)

    threshold = outlier_tol * resid_on_old_trans.std()
    keepers = np.where(resid_on_old_trans < threshold)[0]

    if verbose:
        msg = '  Outlier Rejection: Keeping {0:d} of {1:d}'
        print(msg.format(len(keepers), len(resid_on_old_trans)))
        
    return keepers

def setup_trans_info(trans_input, trans_args, N_lists, iters):
    """ Setup transformation info into a usable format.

    trans_input : list or None
    trans_args : dict or None
    N_lists : int
    iters : int
    """
    trans_list = [None for ii in range(N_lists)]
    if trans_input != None:
        trans_list = [trans_input[ii] for ii in range(N_lists)]

    # Keep a list of trans_args, one for each starlist. If only
    # a single is passed in, replicate for all star lists, all loop iterations.
    if type(trans_args) == dict:
        tmp = trans_args
        trans_args = [tmp for ii in range(iters)]
        
    return trans_list, trans_args

def apply_mag_lim(star_list, mag_lim):
    """ Apply a magnitude limit to the list. If no magnitude limit is 
    specified, then return a copy of the list.  This works on a 
    reference list (with 'm0') or a star_list ('m') with 'm0' taking
    priority.

    mag_lim : 2 element array
        Contains the minimum and maximum magnitude cut to apply. If none,
        no magnitude cut is applied.

    """
    star_list_T = copy.deepcopy(star_list)

    if (mag_lim is not None):
        # Support 'm0' (primary) or 'm' column name.
        if 'm0' in star_list_T.colnames:
            mcol = 'm0'
        else:
            mcol = 'm'

        conditions = {}
    
        cond_key = '{0:s}_min'.format(mcol)
        conditions[cond_key] = mag_lim[0]

        cond_key = '{0:s}_max'.format(mcol)
        conditions[cond_key] = mag_lim[1]

        star_list_T.restrict_by_value(**conditions)

    return star_list_T
    
    
def get_weighting_scheme(weights, ref_list, star_list):
    if 'xe' in ref_list.colnames:
        var_xref = ref_list['xe']**2
        var_yref = ref_list['ye']**2
    else:
        var_xref = 0.0
        var_yref = 0.0
        
    if 'xe' in star_list.colnames:
        var_xlis = star_list['xe']**2
        var_ylis = star_list['ye']**2
    else:
        var_xlis = 0.0
        var_ylis = 0.0

    if weights != None:
        if weights == 'both,var':
            weight = 1.0 / (var_xref + var_xlis + var_yref + var_ylis)
        if weights == 'both,std':
            weight = 1.0 / np.sqrt(var_xref + var_xlis + var_yref + var_ylis)
        if weights == 'ref,var':
            weight = 1.0 / (var_xref + var_yref)
        if weights == 'ref,std':
            weight = 1.0 / np.sqrt(var_xref + var_yref)
        if weights == 'list,var':
            weight = 1.0 / (var_xlis, var_ylis)
        if weights == 'list,std':
            weight = 1.0 / np.sqrt(var_xlis, var_ylis)
    else:
        weight = None

    # One last check to make sure we had weights at all.
    # Technically, this is mis-use; but lets handle it anyhow.
    if ('xe' not in ref_list.colnames) and ('ye' not in star_list.colnames):
        weight = None

    return weight

def get_pos_at_time(t, starlist):
    """
    Take a starlist, check to see if it has velocity columns.
    If it does, then propogate the positions forward in time 
    to the desired epoch. If no velocities exist, then just 
    use ['x0', 'y0'] or ['x', 'y']

    Inputs
    ----------
    t_array : float
        The time to propogate to. Usually in decimal years; 
        but it should be in the same units
        as the 't0' column in starlist.
    """
    if ('vx' in starlist.colnames) and ('vy' in starlist.colnames):
        dt = t - starlist['t0']
        x = starlist['x0'] + (starlist['vx'] * dt)
        y = starlist['y0'] + (starlist['vy'] * dt)
    else:
        if ('x0' in starlist.colnames) and ('y0' in starlist.colnames):
            x = starlist['x0']
            y = starlist['y0']
        else:
            x = starlist['x']
            y = starlist['y']
        
    return (x, y)

