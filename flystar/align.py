import numpy as np
from flystar import match
from flystar import transforms
from flystar import plots
from flystar.starlists import StarList
from flystar.startables import StarTable
from flystar import motion_model
from astropy.table import Table, Column, vstack
import datetime
import copy
import os
import pdb
import time
import warnings
from astropy.utils.exceptions import AstropyUserWarning

class MosaicSelfRef(object):
    def __init__(self, list_of_starlists, ref_index=0, iters=2,
                 dr_tol=[1, 1], dm_tol=[2, 1],
                 outlier_tol=[None, None],
                 trans_args=[{'order': 2}, {'order': 2}],
                 init_order=1,
                 mag_trans=True, mag_lim=None, weights=None,
                 trans_input=None, trans_class=transforms.PolyTransform,
                 # TODO: consider deleting use_vel fully, for now I'm putting in
                 # a fallback so people can still use old code for now
                 use_vel=None, default_motion_model='Fixed',
                 calc_trans_inverse=False,
                 init_guess_mode='miracle', iter_callback=None,
                 verbose=True):
        """
        Make a mosaic object by passing in a list of starlists and then running fit(). 

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

        outlier_tol : list or array
            The outlier tolerance (in units of sigma) for rejecting outlier stars. 
            This is a list of tol values, one for each iteration of matching/transformation.

        mag_trans : boolean
            If true, this will also calculate and (temporarily) apply a zeropoint offset to 
            magnitudes in each list to bring them into a common magnitude system. This is 
            essential for matching (with finite dm_tol) starlists of different filters or 
            starlists that are not photometrically calibrated. Note that the final_table columns 
            of 'm', 'm0', and 'm0_err' will contain the transformed magnitudes while the
            final_table column 'm_orig' will contain the original un-transformed magnitudes. 
            If mag_trans = False, then no such zeropoint offset it applied at any point. 

        mag_lim : array
            If different from None, it indicates the minimum and maximum magnitude
            on the catalogs for finding the transformations. Note, if you want specify the mag_lim
            separately for each list and each iteration, you need to pass in a 2D array that
            has shape (N_lists, 2).

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

        # TODO: update when decided
        use_vel : boolean
            If velocities are present in the reference list and use_vel == True, then during
            each iteration of the alignment, the reference list will be propogated in time
            using the velocity information. So all transformations will be derived w.r.t. 
            the propogated positions. See also update_vel.

       calc_trans_inverse: boolean
            If true, then calculate the inverse transformation (from reference to starlist)
            in addition to the normal transformation (from starlist to reference). The inverse
            calculation is calculated by switching the order to the positions in match_and_transform.
            The inverse transformations are saved in self.trans_list_inverse.

            self.trans_list_inverse doesn't exist if calc_trans_inverse == False

        init_guess_mode : string
            If no initial transformations are passed in via the trans_input keyword, then we have
            to make the initial transformation and matching blindly. We can do this in a couple of 
            different ways. Options are 'miracle' or 'name' (see trans_initial_guess() for more details).

        iter_callback : None or function
            A function to call (that accepts a StarTable object and an iteration number)
            at the end of every iteration. This can be used for plotting or printing state. 

        verbose : int (0 to 9, inclusive)
            Controls the verbosity of print statements. (0 least, 9 most verbose).
            For backwards compatibility, 0 = False, 9 = True.
            (Note: technically right now no checks on whether the number is an integer or not...)

        Example
        ----------
        msc = align.MosaicToRef(list_of_starlists, iters=1,
                                dr_tol=[0.1], dm_tol=[5],
                                outlier_tol=[None], mag_lim=[13, 21],
                                trans_class=transforms.PolyTransform,
                                trans_args=[{'order': 1}],
                                weights='both,std',
                                init_guess_mode='miracle', verbose=False)
        msc.fit()

        # Access a list of all the transformation parameters:
        trans_list = msc.trans_list 

        # Access the fully-combined reference table.
        stars_table = msc.ref_table

        # Plot the magnitude of the first star vs. time:
        # Overplot the mean magnitude. 
        plt.plot(stars_table['t'][0, :], stars_table['m'][0, :], 'k.')
        plt.axhline(stars_table['m0'][0])    

        # Plot the X position of the first star vs. time:
        # Overplot the best-fit proper motion.
        times = stars_table['t'][0, :]
        plt.errorbar(times, stars_table['x'][0, :], yerr=stars_table['xe'][0, :])
        plt.axhline(stars_table['x0'][0] + stars_table['vx'][0]*(times - stars_table['t0'][0]))   

        """

        self.star_lists = list_of_starlists
        self.ref_index = ref_index
        self.iters = iters
        self.dr_tol = dr_tol
        self.dm_tol = dm_tol
        self.outlier_tol = outlier_tol
        self.trans_args = trans_args
        self.init_order = init_order
        self.mag_trans = mag_trans
        self.mag_lim = mag_lim
        self.weights = weights
        self.trans_input = trans_input
        self.trans_class = trans_class
        self.calc_trans_inverse = calc_trans_inverse
        # TODO: consider whether we want this fallback
        if use_vel is None:
            self.default_motion_model = default_motion_model
        else:
            if use_vel:
                self.default_motion_model = 'Linear'
            else:
                self.default_motion_model = 'Fixed'
        self.init_guess_mode = init_guess_mode
        self.iter_callback = iter_callback
        self.verbose = verbose

        # For backwards compatibility.
        if self.verbose is True:
            self.verbose = 9
        if self.verbose is False:
            self.verbose = 0
            
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

        if self.mag_lim is None:
            self.mag_lim = np.repeat([[None, None]], len(self.star_lists), axis=0)
        elif (len(self.mag_lim) == 2):
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
        vx  (only if use_motion=True)
        vy  (only if use_motion=True)
        vxe (only if use_motion=True)
        vye (only if use_motion=True)

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

            if self.verbose > 0:
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

            if self.iter_callback != None:
                self.iter_callback(self.ref_table, nn)
            

        ##########
        #
        # Re-do all matching given final transformations.
        #        No trimming this time.
        #        First rest the reference table 2D values. 
        ##########
        self.reset_ref_values(exclude=['used_in_trans'])

        if self.verbose > 0:
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
        if self.verbose > 0:
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
            if self.verbose > 0:
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
            #       star_list_orig_trim is actually trimmed but not yet transformed.
            #       star_list_T is trimmed and transformed
            self.apply_mag_lim_via_use_in_trans(ref_list, ref_mag_lim)
            star_list_orig_trim = apply_mag_lim(star_list, self.mag_lim[ii])  # trimmed, untransformed copy
            star_list_T = apply_mag_lim(star_list, self.mag_lim[ii])  # trimmed, will be transformed copy

            ### Initial match and transform: 1st order (if we haven't already).
            if trans is None:
                # Only use "use_in_trans" reference stars, even for initial guessing.
                keepers = np.where(ref_list['use_in_trans'] == True)[0]

                trans = trans_initial_guess(ref_list[keepers], star_list_orig_trim, self.trans_args[0],
                                            mode=self.init_guess_mode,
                                            order=self.init_order,
                                            verbose=self.verbose,
                                            mag_trans=self.mag_trans)

            if self.mag_trans:
                star_list_T.transform_xym(trans) # trimmed, transformed
            else:
                star_list_T.transform_xy(trans) 

            # Match stars between the transformed, trimmed lists.
            idx1, idx2, dr, dm = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                             ref_list['x'], ref_list['y'], ref_list['m'],
                                             dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)
            if self.verbose > 1:
                print( '  Match 1: Found ', len(idx1), ' matches out of ', len(star_list_T),
                       '. If match count is low, check dr_tol, dm_tol.' )

            # Outlier rejection
            if outlier_tol != None:
                keepers =  self.outlier_rejection_indices(star_list_T[idx1], ref_list[idx2],
                                                          outlier_tol)
                if self.verbose > 1:
                    print( '  Rejected ', len(idx1) - len(keepers), ' outliers.' )
                    
                idx1 = idx1[keepers]
                idx2 = idx2[keepers]

            # Only use stars specified by "use_in_trans" column.
            if 'use_in_trans' in ref_list.colnames:
                keepers = np.where(ref_list[idx2]['use_in_trans'] == True)[0]
                
                if self.verbose > 1:
                    print( '  Rejected ', len(idx1) - len(keepers), ' with use_in_trans=False.' )
                    
                idx1 = idx1[keepers]
                idx2 = idx2[keepers]

            # Determine weights in the fit.
            weight = self.get_weights_for_lists(ref_list[idx2], star_list_T[idx1])

            # Derive the best-fit transformation parameters. 
            if self.verbose > 1:
                print( '  Using ', len(idx1), ' stars in transformation.' )
            trans = self.trans_class.derive_transform(star_list_orig_trim['x'][idx1], star_list_orig_trim['y'][idx1], 
                                                      ref_list['x'][idx2], ref_list['y'][idx2],
                                                      **trans_args,
                                                      m=star_list_orig_trim['m'][idx1], mref=ref_list['m'][idx2],
                                                      weights=weight, mag_trans=self.mag_trans)

            # Save the final transformation.
            self.trans_list[ii] = trans

            # If desired, calculate and save the inverse transformation
            # NOTE: We will not recalculate weights here
            if self.calc_trans_inverse:
                if self.verbose > 1:
                    print('Doing inverse')
                trans_inv = self.trans_class.derive_transform(ref_list['x'][idx2], ref_list['y'][idx2],
                                                              star_list_orig_trim['x'][idx1], star_list_orig_trim['y'][idx1],
                                                              trans_args['order'], m=ref_list['m'][idx2],
                                                              mref=star_list_orig_trim['m'][idx1], weights=weight,
                                                              mag_trans=self.mag_trans)
                self.trans_list_inverse[ii] = trans_inv

            # Apply the XY transformation to a new copy of the starlist and
            # do one final match between the two (now transformed) lists.
            star_list_T = copy.deepcopy(star_list)
            if self.mag_trans:
                star_list_T.transform_xym(self.trans_list[ii])
            else:
                star_list_T.transform_xy(self.trans_list[ii])

            if self.verbose > 7:
                hdr = '{nr:20s} {n:20s} {xl:9s} {xr:9s} {yl:9s} {yr:9s} {ml:6s} {mr:6s} '
                hdr += '{dx:7s} {dy:7s} {dm:6s} {xo:9s} {yo:9s} {mo:6s}'
                print(hdr.format(nr='name_ref', n='name_lis',
                                     xl='x_lis_T', xr='x_ref',
                                     yl='y_lis_T', yr='y_ref',
                                     ml='m_lis_T', mr='m_ref',
                                     dx='dx_mpix', dy='dy_mpix', dm='dm',
                                     xo='x_orig', yo='y_orig', mo='m_orig'))
                
                fmt = '{nr:20s} {n:20s} {xl:9.5f} {xr:9.5f} {yl:9.5f} {yr:9.5f} {ml:6.2f} {mr:6.2f} '
                fmt += '{dx:7.2f} {dy:7.2f} {dm:6.2f} {xo:9.5f} {yo:9.5f} {mo:6.2f}'
                for foo in range(len(idx1)):
                    star_s = star_list_orig_trim[idx1[foo]]
                    star_r = ref_list[idx2[foo]]
                    star_t = star_list_T[idx1[foo]]
                    print(fmt.format(nr=star_r['name'], n=star_s['name'], xl=star_t['x'], xr=star_r['x'],
                                     yl=star_t['y'], yr=star_r['y'],
                                     ml=star_t['m'], mr=star_r['m'],
                                     dx=(star_t['x'] - star_r['x']) * 1e3, 
                                     dy=(star_t['y'] - star_r['y']) * 1e3,
                                     dm=(star_t['m'] - star_r['m']),
                                     xo=star_s['x'], yo=star_s['y'], mo=star_s['m']))
                    
            idx_lis, idx_ref, dr, dm = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                                   ref_list['x'], ref_list['y'], ref_list['m'],
                                                   dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)
            
            if self.verbose > 1:
                print( '  Match 2: After trans, found ', len(idx_lis), ' matches out of ', len(star_list_T),
                       '. If match count is low, check dr_tol, dm_tol.' )

            ## Make plot, if desired
            plots.trans_positions(ref_list, ref_list[idx_ref], star_list_T, star_list_T[idx_lis],
                                  fileName='{0}'.format(star_list_T['t'][0]))

            ### Update the observed (but transformed) values in the reference table.
            self.update_ref_table_from_list(star_list, star_list_T, ii, idx_ref, idx_lis, idx2)

            ### Update the "average" values to be used as the reference frame for the next list.
            if self.update_ref_orig != 'periter':
                self.update_ref_table_aggregates()

            # Print out some metrics
            if self.verbose > 0:
                msg1 = '    {0:2s} (mean and std) for {1:10s}: {2:8.5f} +/- {3:8.5f}'
                print('  Residuals: ')
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
                print('    Used {0:d} trans ref stars.'.format(len(used)))
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

        # Add inverse trans list, if desired
        if self.calc_trans_inverse:
            trans_list_inverse = [None for ii in range(N_lists)]
            self.trans_list_inverse = trans_list_inverse

        return

    def setup_ref_table_from_starlist(self, star_list):
        """ 
        Start with the reference list.... this will change and grow
        over time, so make a copy that we will keep updating.
        The reference table will contain one column for every named
        array in the original reference star list.
        """
        col_arrays = {}
        motion_model_col_names = motion_model.get_all_motion_model_param_names(with_errors=True) + ['m0','m0_err','use_in_trans']
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

        # Make sure ref_table has the necessary x0, y0, m0 and associated
        # error columns. If they don't exist, then add them as a copy of
        # the original x,y,m etc columns. 
        new_cols_arr = ['x0', 'y0', 'm0']
        orig_cols_arr = ['x', 'y', 'm']
        ref_cols = ref_table.keys()
        for ii in range(len(new_cols_arr)):
            if not new_cols_arr[ii] in ref_cols:
                # Some munging to convert data shape from (N,1) to (N,),
                # since these are all 1D cols
                vals = np.transpose(np.array(ref_table[orig_cols_arr[ii]]))[0]

                # Now add to ref_table
                new_col = Column(vals, name=new_cols_arr[ii])
                ref_table.add_column(new_col)

        # Do the same thing for the x0e, y0e, m0e columns, but
        # ONLY IF THEY ALREADY EXIST IN REF_TABLE! Otherwise,
        # just fill these tables with zeros. We need something
        # in these columns in order for the error propagation to
        # work later on.
        new_err_cols = ['x0_err', 'y0_err', 'm0_err']
        orig_err_cols = ['xe', 'ye', 'me']
        for ii in range(len(new_err_cols)):
            # If the orig col name (e.g. xe) is in the ref_table, but the new col name
            # (e.g. x0e) doesn't exist, then add the x0e column as a duplicate of xe.
            if (orig_err_cols[ii] in ref_cols) & (not new_err_cols[ii] in ref_cols):
                # Some munging to convert data shape from (N,1) to (N,),
                # since these are all 1D cols
                vals = np.transpose(np.array(ref_table[orig_err_cols[ii]]))[0]

                # Now add to ref_table
                new_col = Column(vals, name=new_err_cols[ii])
                ref_table.add_column(new_col)
            elif (not orig_err_cols[ii] in ref_cols) & (not new_err_cols[ii] in ref_cols):
                # If neither the orig_err_col or new_err_col is in the ref_table, put in the
                # new_err_cols as an array of zeros
                vals = np.zeros(len(ref_table))
                new_col = Column(vals, name=new_err_cols[ii])
                ref_table.add_column(new_col)                

        # Final check: ref_table should now have x0, y0, m0, x0e, y0e, and m0e columns
        # This is necessary for later steps, even if the columns are just zeros.
        final_new_cols = np.concatenate((new_cols_arr, new_err_cols))
        for ii in final_new_cols:
            assert ii in ref_table.keys()
                
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
        self.ref_table, idx_lis_new, idx_ref_new = add_rows_for_new_stars(self.ref_table, star_list, idx_lis,
                                                        default_motion_model=self.default_motion_model)
        if len(idx_ref_new) > 0:
            if self.verbose > 0:
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
    
    def update_ref_table_aggregates(self, n_boot=0):
        """
        Average positions or fit velocities.
        Average magnitudes.
        Calculate bootstrap errors if desired.

        Update the use_in_trans values as needed.

        Updates aggregate columns in self.ref_table in place.
        """
        # Keep track of the original reference values.
        # In certain cases, we will NOT update these.
        if not self.update_ref_orig:
            ref_orig_idx = np.where(self.ref_table['ref_orig'] == True)[0]
            vals_orig = {}
            vals_orig['m0'] = self.ref_table['m0'][ref_orig_idx]
            vals_orig['m0_err'] = self.ref_table['m0_err'][ref_orig_idx]
            motion_model_class_names = [self.default_motion_model]
            if 'motion_model_used' in self.ref_table.keys():
                motion_model_class_names += self.ref_table['motion_model_used'][ref_orig_idx].tolist()
            motion_model_col_names = motion_model.get_list_motion_model_param_names(motion_model_class_names, with_errors=True)
            for mm in motion_model_col_names:
                if mm in self.ref_table.keys():
                    vals_orig[mm] = self.ref_table[mm][ref_orig_idx]
                
        #if self.use_motion:
            # Combine positions with a velocity fit.
        self.ref_table.fit_velocities(bootstrap=n_boot, verbose=self.verbose, default_motion_model=self.default_motion_model)

        # Combine (transformed) magnitudes
        # TODO: how does this work?
        if 'me' in self.ref_table.colnames:
            weights_col = None
        else:
            weights_col = 'me'
            
        self.ref_table.combine_lists('m', weights_col=weights_col, ismag=True)
        '''else:
            weighted_xy = ('xe' in self.ref_table.colnames) and ('ye' in self.ref_table.colnames)
            weighted_m = ('me' in self.ref_table.colnames)
    
            self.ref_table.combine_lists_xym(weighted_xy=weighted_xy, weighted_m=weighted_m)'''
        # Replace the originals if we are supposed to keep them fixed.
        if not self.update_ref_orig:
            for val in vals_orig.keys():
                self.ref_table[val][ref_orig_idx] = vals_orig[val]
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
                weight = 1.0 / (var_xlis + var_ylis)
            if self.weights == 'list,std':
                weight = 1.0 / np.sqrt(var_xlis, var_ylis)
        else:
            weight = None

        # One last check to make sure we had weights at all.
        # Technically, this is mis-use; but lets handle it anyhow.
        if ('xe' not in ref_list.colnames) and ('ye' not in star_list.colnames):
            weight = None

        if weight is not None:
            bad = np.where(np.isfinite(weight) == False)[0]
            if len(bad) == len(weight):
                # Catch the case where we had no positional errors at all...
                # The fit should be unweighted.
                weight = None
            else:
                # Fix bad weights:
                weight[bad] = 0.0

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
            if self.mag_trans:
                star_list_T.transform_xym(self.trans_list[ii])
            else:
                star_list_T.transform_xy(self.trans_list[ii])
            
            xref, yref = get_pos_at_time(star_list_T['t'][0], self.ref_table) #, use_motion=self.use_motion)  # optional velocity propogation.
            mref = self.ref_table['m0']

            idx_lis, idx_ref, dr, dm = match.match(star_list_T['x'], star_list_T['y'], star_list_T['m'],
                                                   xref, yref, mref,
                                                   dr_tol=dr_tol, dm_tol=dm_tol, verbose=self.verbose)
            if self.verbose > 0:
                fmt = 'Matched {0:5d} out of {1:5d} stars in list {2:2d} [dr = {3:7.4f} +/- {4:6.4f}, dm = {5:5.2f} +/- {6:4.2f}'
                print(fmt.format(len(idx_lis), len(star_list_T), ii, dr.mean(), dr.std(), dm.mean(), dm.std()))

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

        if ('motion_model_used' in self.ref_table.colnames):
            x,y,xe,ye = self.ref_table.get_star_positions_at_time(epoch)
        else:
            # No velocities... just used average positions.
            x = self.ref_table['x0']
            y = self.ref_table['y0']
            
            if 'x0_err' in self.ref_table.colnames:
                xe = self.ref_table['x0_err']
                ye = self.ref_table['y0_err']
            else:
                xe = None
                ye = None

        m = self.ref_table['m0']
        
        if 'm0_err' in self.ref_table.colnames:
            me = self.ref_table['m0_err']
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
    
    def calc_bootstrap_errors(self, n_boot=100, boot_epochs_min=-1, calc_vel_in_bootstrap=True):
        """
        Function to calculate bootstrap errors for the transformations as well
        as the proper motions. For each iteration, this will:

        1) Draw full-size bootstrap w/replacement sample from reference stars in 
        ref_table and re-calculate the transformations for each epoch
        2) Apply transformation to all stars in each epoch
        If calc_vel_in_bootstraps:
            3) For each star, draw full-size boostrap sample w/replacement from epochs
            4) Calculate proper motion for each star using resampled epochs
    
        The saved outputs will be: x_trans, y_trans, m_trans (transformed postions/mags),
        as well as the proper motion fit parameters.

        Final calculated errors:
        std(x_trans) ---> x-direction transformation error (and likewise for y_trans, m_trans)
        std(x0) --> x0e (and same with all proper motion fit parameters)

        Parameters:
        ----------
        mosaic_object: MosaicToRef object
            MosaicToRef object after the complete match_and_transform process

       n_boot: int, must be greater than 0
            Number of bootstrap iterations when calculating transformations and the proper motion. 
            PM bootstrap is only done for final proper motion
            calculation (e.g., not for each iteration of the starlist for matching)

        boot_epochs_min: int or -1
            In order to be included in bootstrap analysis, non-reference stars must be detected in 
            at least boot_epochs_min epochs. If boot_epochs_min = -1, then all stars will 
            be included in the analysis, regardless of the number of epochs detected.
            For stars that fail boot_epochs_min criteria, np.nan is used

        calc_vel_in_bootstrap: boolean
           If true, do bootstrap sample w/ replacement over the epochs and calculate 
           stellar proper motions, as well as the bootstrap over reference stars
           to calculate positional alignment errors. If false, only 
           calculate position alignment errors.

        Output:
        ------
        Seven new columns will be added to self.ref_table:
        'xe_boot', 2D column: bootstrap x pos uncertainties due to transformation for each epoch
        'ye_boot', 2D column: bootstrap y pos uncertainties due to transformation for each epoch
        'me_boot', 2D column: bootstrap mag uncertainties due to transformation for each epoch
        
        If calc_vel_in_bootstrap:
        'x0_err_boot', 1D column: bootstrap uncertainties in x0 for PM fit
        'y0_err_boot', 1D column: bootstrap uncertainties in y0 for PM fit
        'vx_err_boot', 1D column: bootstrap uncertainties in vx for PM fit
        'vy_err_boot', 1D column: bootstrap uncertainties in vy for PM fit

        For stars that fail boot_epochs_min criteria, np.nan is used
        """
        # First, assert than n_boot > 0
        assert n_boot > 0

        ref_table = copy.deepcopy(self.ref_table)
        n_epochs = len(ref_table['x'][0])
        t_arr = get_all_epochs(ref_table)
        #t_arr = ref_table['t'][np.where(ref_table['n_detect'] == np.max(ref_table['n_detect']))[0][0]]
        t0_arr = ref_table['t0']

        # Identify reference stars. If desired, trim ref_table to only stars to only
        # reference stars and those that pass boot_epochs_min criteria
        if boot_epochs_min > 0:
            idx_good = np.where( (ref_table['n_detect'] >= boot_epochs_min) | (ref_table['use_in_trans']) )
            ref_table = ref_table[idx_good]
            t0_arr = t0_arr[idx_good]
        else:
            idx_good = np.arange(0, len(ref_table), 1)
        idx_ref = np.where(ref_table['use_in_trans'] == True)

        # Initialize output arrays
        x_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999
        y_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999
        m_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999
        xe_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999
        ye_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999
        me_trans_arr = np.ones((len(ref_table['x']), n_boot, n_epochs)) * -999

        # Set up motion model parameters
        motion_model_list = ['Fixed', self.default_motion_model]
        if 'motion_model_used' in ref_table.keys():
            motion_model_list += ref_table['motion_model_used'].tolist()
        elif 'motion_model_assigned' in ref_table.keys():
            motion_model_list += ref_table['motion_model_assigned'].tolist()
        motion_col_list = motion_model.get_list_motion_model_param_names(np.unique(motion_model_list).tolist(), with_errors=False, with_fixed=False)
        if calc_vel_in_bootstrap:
            motion_data = {}
            for col in motion_col_list:
                motion_data[col] = np.ones((len(ref_table['x']), n_boot)) * -999

        ### IF MEMORY PROBLEMS HERE:
        ### DEFINE MEAN, STD VARIABLES AND BUILD THEM RATHER THAN SAVING FULL ARRAY
        ### DECREASE PRECISION ON ARRAYS (32 bit instead of 64: dtype=np.float32)
        ### AT SOME POINT, NEED TO CONVERT BACK (LOOK UP HOW TO DO THIS CAREFULLY)
        t1 = time.time()
        for ii in range(n_boot):
            # Recalculate transformations using bootstrap sample of
            # reference stars. Use a loop for each epoch here, so we
            # can handle case where different reference stars are used
            # in different epochs
            for jj in range(n_epochs):
                # Extract bootstrap sample of matched reference stars
                good = np.where(~np.isnan(ref_table['x_orig'][idx_ref][:,jj]))
                samp_idx = np.random.choice(good[0], len(good[0]), replace=True)
                
                # Get reference star positions in particular epoch from ref_list.
                t_epoch = t_arr[jj]
                ref_orig = self.get_ref_list_from_table(t_epoch)

                # Get idx of reference stars in bootstrap sample in the ref_orig.
                # Then, use these to build reference starlist for the alignment
                idx_tmp = []
                for ff in range(len(samp_idx)):
                    name_tmp = ref_table['name'][samp_idx[ff]]
                    foo = np.where(ref_orig['name'] == name_tmp)[0][0]
                    idx_tmp.append(foo)

                ref_boot = StarList(name=ref_orig['name'][idx_tmp],
                                       x=ref_orig['x'][idx_tmp],
                                       y=ref_orig['y'][idx_tmp],
                                       m=ref_orig['m'][idx_tmp],
                                       xe=ref_orig['xe'][idx_tmp],
                                       ye=ref_orig['ye'][idx_tmp],
                                       me=ref_orig['me'][idx_tmp])

                # Now build star list with original positions of the reference stars
                # in the bootstrap sample
                starlist_boot = StarList(name=ref_table['name'][idx_ref][samp_idx],
                                         x=ref_table['x_orig'][:,jj][idx_ref][samp_idx],
                                         y=ref_table['y_orig'][:,jj][idx_ref][samp_idx],
                                         m=ref_table['m_orig'][:,jj][idx_ref][samp_idx],
                                         xe=ref_table['xe_orig'][:,jj][idx_ref][samp_idx],
                                         ye=ref_table['ye_orig'][:,jj][idx_ref][samp_idx],
                                         me=ref_table['me_orig'][:,jj][idx_ref][samp_idx])
            
                # Calculate weights based on weights keyword. If weights desired, will need to
                # make starlist objects for this
                if self.weights != None:
                    # In order for weights calculation to work, we need to apply a transformation
                    # to the star_list_T so it is in the same units as ref_boot. So, we'll apply
                    # the final transformation for the epoch to get close enough for the
                    # purposes of the bootstrap calculation
                    starlist_boot_T = copy.deepcopy(starlist_boot)
                    if self.mag_trans:
                        starlist_boot_T.transform_xym(self.trans_list[jj])
                    else:
                        starlist_boot_T.transform_xy(self.trans_list[jj])
                        
                    weight = self.get_weights_for_lists(ref_boot, starlist_boot_T)
                else:
                    weight = None
            
                # Recalculate transformation
                trans = self.trans_class.derive_transform(starlist_boot['x'], starlist_boot['y'],
                                                                   ref_boot['x'], ref_boot['y'],
                                                                   self.trans_args[0]['order'],
                                                                   m=starlist_boot['m'], mref=ref_boot['m'],
                                                                   weights=weight, mag_trans=self.mag_trans)

                # Apply transformation to *all* orig positions in this epoch. Need to make a new
                # FLYSTAR starlist object with the original positions for this. We don't
                # use the original starlist itself (i.e. self.star_lists[jj])
                # because we want to preserve the matched order of ref_table
                starlist = StarList(name=ref_table['name'], x=ref_table['x_orig'][:,jj],
                                               y=ref_table['y_orig'][:,jj],
                                               m=ref_table['m_orig'][:,jj],
                                               xe=ref_table['xe_orig'][:,jj],
                                               ye=ref_table['ye_orig'][:,jj],
                                               me=ref_table['me_orig'][:,jj])
                starlist_T = copy.deepcopy(starlist)
                if self.mag_trans:
                    starlist_T.transform_xym(trans)
                else:
                    starlist_T.transform_xy(trans)
                    
                # Add output to pos arrays
                x_trans_arr[:,ii,jj] = starlist_T['x']
                y_trans_arr[:,ii,jj] = starlist_T['y']
                m_trans_arr[:,ii,jj] = starlist_T['m']
                xe_trans_arr[:,ii,jj] = starlist_T['xe']
                ye_trans_arr[:,ii,jj] = starlist_T['ye']
                me_trans_arr[:,ii,jj] = starlist_T['me']
 
            t2 = time.time()
            #print('=================================================')
            #print('Time to do {0} epochs: {1}s'.format(n_epochs,  t2-t1))
            #print('=================================================')
            
            # Finally, calculate proper motions for this bootstrap iteration
            # for each star, if desired. Draw a full-sample bootstrap over the epochs
            # for each star, and then run it through the startable fit_velocities machinery
            if calc_vel_in_bootstrap:
                boot_idx = np.random.choice(np.arange(0, n_epochs, 1), size=n_epochs)
                t_boot = t_arr[boot_idx]
            
                star_table = StarTable(name=ref_table['name'],
                                        x=x_trans_arr[:,ii,boot_idx],
                                        y=y_trans_arr[:,ii,boot_idx],
                                        m=m_trans_arr[:,ii,boot_idx],
                                        xe=xe_trans_arr[:,ii,boot_idx],
                                        ye=ye_trans_arr[:,ii,boot_idx],
                                        me=me_trans_arr[:,ii,boot_idx],
                                        t=np.tile(t_boot, (len(ref_table),1)))

                # Now, do proper motion calculation, making sure to fix t0 to the
                # orig value (so we can get a reasonable error on x0, y0)
                star_table.fit_velocities(fixed_t0=t0_arr, default_motion_model=self.default_motion_model)

                # Save proper motion fit results to output arrays
                for col in motion_col_list:
                    motion_data[col][:,ii] = star_table[col]

                # Quick check to make sure bootstrap calc was valid: output t0 should be
                # same as input t0_arr, since we used fixed_t0 option
                assert np.sum(abs(star_table['t0'] - t0_arr) == 0)

                #t3 = time.time()
                #print('=================================================')
                #print('Time to calc proper motions: {0}s'.format(t3-t2))
                #print('=================================================')

        # Calculate the bootstrap error values.
        x_err_b = np.std(x_trans_arr, ddof=1, axis=1)
        y_err_b = np.std(y_trans_arr, ddof=1, axis=1)
        m_err_b = np.std(m_trans_arr, ddof=1, axis=1)

        motion_data_err = {}
        if calc_vel_in_bootstrap:
            for col in motion_col_list:
                motion_data_err[col] = np.nanstd(motion_data[col], ddof=1,axis=1)
        else:
            for col in motion_col_list:
                motion_data_err[col] = np.nan

        # Add summary statistics to *original* ref_table, i.e. ref_table
        # hanging off of mosaic object.
        col_heads_2D = ['xe_boot', 'ye_boot', 'me_boot']
        data_dict = {'xe_boot': x_err_b, 'ye_boot': y_err_b, 'me_boot': m_err_b}
        for col in motion_col_list:
            data_dict[col+'_err_boot'] = motion_data_err[col]
            
        for ff in col_heads_2D:
            col = Column(np.ones((len(self.ref_table), n_epochs)), name=ff)
            col.fill(np.nan)
            
            col[idx_good] = data_dict[ff]
            self.ref_table.add_column(col)

        # Now handle the velocities, if they were calculated
        if calc_vel_in_bootstrap:
            col_heads_1D = [col+'_err_boot' for col in motion_col_list]
            
            for ff in col_heads_1D:
                col = Column(np.ones(len(self.ref_table)), name=ff)
                col.fill(np.nan)
                
                col[idx_good] = data_dict[ff]
                self.ref_table.add_column(col)

        print('===============================')
        print('Done with bootstrap')
        print('===============================')
        
        return
    

class MosaicToRef(MosaicSelfRef):
    def __init__(self, ref_list, list_of_starlists, iters=2,
                 dr_tol=[1, 1], dm_tol=[2, 1],
                 outlier_tol=[None, None],
                 trans_args=[{'order': 2}, {'order': 2}],
                 init_order=1,
                 mag_trans=True, mag_lim=None, ref_mag_lim=None,
                 weights=None,
                 trans_input=None,
                 trans_class=transforms.PolyTransform,
                 calc_trans_inverse=False,
                 use_ref_new=False,
                 use_vel=None, default_motion_model='Fixed',
                 update_ref_orig=False,
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

        outlier_tol : list or array
            The outlier tolerance (in units of sigma) for rejecting outlier stars. 
            This is a list of tol values, one for each iteration of matching/transformation.

        mag_trans : boolean
            If true, this will also calculate and (temporarily) apply a zeropoint offset to 
            magnitudes in each list to bring them into a common magnitude system. This is 
            essential for matching (with finite dm_tol) starlists of different filters or 
            starlists that are not photometrically calibrated. Note that the final_table columns 
            of 'm', 'm0', and 'm0_err' will contain the transformed magnitudes while the
            final_table column 'm_orig' will contain the original un-transformed magnitudes. 
            If mag_trans = False, then no such zeropoint offset it applied at any point. 

        mag_lim : array
            If different from None, it indicates the minimum and maximum magnitude
            on the catalogs for finding the transformations. Note, if you want specify the mag_lim
            separately for each list and each iteration, you need to pass in a 2D array that
            has shape (N_lists, 2).

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

        init_order: int
            Polynomial transformation order to use for initial guess transformation.
            Order=1 should be used in most cases, but sometimes higher order is needed

        calc_trans_inverse: boolean
            If true, then calculate the inverse transformation (from reference to starlist)
            in addition to the normal transformation (from starlist to reference). The inverse
            calculation is calculated by switching the order to the positions in match_and_transform.
            The inverse transformations are saved in self.trans_list_inverse.

            self.trans_list_inverse doesn't exist if calc_trans_inverse == False

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

        use_motion : boolean
            If velocities are present in the reference list and use_motion == True, then during
            each iteration of the alignment, the reference list will be propogated in time
            using the velocity information. So all transformations will be derived w.r.t. 
            the propogated positions. See also update_vel.

        init_guess_mode : string
            If no initial transformations are passed in via the trans_input keyword, then we have
            to make the initial transformation and matching blindly. We can do this in a couple of 
            different ways. Options are 'miracle' or 'name' (see trans_initial_guess() for more details).

        iter_callback : None or function
            A function to call (that accepts a StarTable object and an iteration number)
            at the end of every iteration. This can be used for plotting or printing state. 

        Example
        ----------
        msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=1,
                                dr_tol=[0.1], dm_tol=[5],
                                outlier_tol=[None], mag_lim=[13, 21],
                                trans_class=transforms.PolyTransform,
                                trans_args=[{'order': 1}],
                                use_motion=True,
                                use_ref_new=False,
                                update_ref_orig=False,
                                mag_trans=False,
                                weights='both,std',
                                init_guess_mode='miracle', verbose=False)
        msc.fit()

        # Access a list of all the transformation parameters:
        trans_list = msc.trans_list 

        # Access the fully-combined reference table.
        stars_table = msc.ref_table

        # Plot the magnitude of the first star vs. time:
        # Overplot the mean magnitude. 
        plt.plot(stars_table['t'][0, :], stars_table['m'][0, :], 'k.')
        plt.axhline(stars_table['m0'][0])    

        # Plot the X position of the first star vs. time:
        # Overplot the best-fit proper motion.
        times = stars_table['t'][0, :]
        plt.errorbar(times, stars_table['x'][0, :], yerr=stars_table['xe'][0, :])
        plt.axhline(stars_table['x0'][0] + stars_table['vx'][0]*(times - stars_table['t0'][0]))   
        """
        super().__init__(list_of_starlists, ref_index=-1, iters=iters,
                         dr_tol=dr_tol, dm_tol=dm_tol,
                         outlier_tol=outlier_tol, trans_args=trans_args,
                         init_order=init_order,
                         mag_trans=mag_trans, mag_lim=mag_lim, weights=weights,
                         trans_input=trans_input, trans_class=trans_class,
                         calc_trans_inverse=calc_trans_inverse, use_vel=use_vel,
                         default_motion_model = default_motion_model,
                         init_guess_mode=init_guess_mode,
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
        if ('xe' not in self.ref_list.colnames) and ('x0_err' in self.ref_list.colnames):
            self.ref_list['xe'] = self.ref_list['x0_err']
            self.ref_list['ye'] = self.ref_list['y0_err']
        if ('m' not in self.ref_list.colnames) and ('m0' in self.ref_list.colnames):
            self.ref_list['m'] = self.ref_list['m0']
        if ('me' not in self.ref_list.colnames) and ('m0_err' in self.ref_list.colnames):
            self.ref_list['me'] = self.ref_list['m0_err']
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
        vx  (only if use_motion=True)
        vy  (only if use_motion=True)
        vxe (only if use_motion=True)
        vye (only if use_motion=True)

        """
        # Create a log file of the parameters used in the fit.
        with open('MosaicToRef_input_params.log', 'w',) as _log:
            logger(_log, 'Parameters used for fit: ', self.verbose)
            logger(_log, '------------------------- ', self.verbose)
            logger(_log, '  dr_tol = ' + str(self.dr_tol), self.verbose)
            logger(_log, '  dm_tol = ' + str(self.dm_tol), self.verbose)
            logger(_log, '  outlier_tol = ' + str(self.outlier_tol), self.verbose)
            logger(_log, '  trans_args = ' + str(self.trans_args), self.verbose)
            logger(_log, '  mag_trans = ' + str(self.mag_trans), self.verbose)
            logger(_log, '  mag_lim = ' + str(self.mag_lim), self.verbose)
            logger(_log, '  ref_mag_lim = ' + str(self.ref_mag_lim), self.verbose)
            logger(_log, '  weights = ' + str(self.weights), self.verbose)
            logger(_log, '  trans_input = ' + str(self.trans_input), self.verbose)
            logger(_log, '  trans_class = ' + str(self.trans_class), self.verbose)
            logger(_log, '  calc_trans_inverse = ' + str(self.calc_trans_inverse), self.verbose)
            logger(_log, '  use_ref_new = ' + str(self.use_ref_new), self.verbose)
            #logger(_log, '  use_vel = ' + str(self.use_vel), self.verbose)
            logger(_log, '  default_motion_model = ' + str(self.default_motion_model), self.verbose)
            logger(_log, '  update_ref_orig = ' + str(self.update_ref_orig), self.verbose)
            logger(_log, '  init_guess_mode = ' + str(self.init_guess_mode), self.verbose)
            logger(_log, '  iter_callback = ' + str(self.iter_callback), self.verbose)
            logger(_log, '-------------------------\n', self.verbose)


        ##########
        # Setup a reference table to store data. It will contain:
        #    x0, y0, m0 -- the running average of positions: 1D
        #    x, y, m, (opt. errors) -- the transformed positions for the lists: 2D
        #    x_orig, y_orig, m_orig, (opt. errors) -- the transformed errors for the lists: 2D
        #    w, w_orig (optiona) -- the input and output weights of stars in transform: 2D
        ##########
        self.ref_table = self.setup_ref_table_from_starlist(self.ref_list)
        
        # copy over motion model parameters if they exist in the reference list
        if 'motion_model_assigned' in self.ref_list.colnames:
            self.ref_table['motion_model_assigned'] = self.ref_list['motion_model_assigned']
        if 'motion_model' in self.ref_list.colnames:
            self.ref_table['motion_model_assigned'] = self.ref_list['motion_model']
        for param in motion_model.get_all_motion_model_param_names(with_fixed=True, with_errors=True):
            if param in self.ref_list.colnames:
                self.ref_table[param] = self.ref_list[param]

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

            if self.verbose > 0:
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
            if self.verbose > 0:
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

        if self.verbose > 0:
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
        if self.verbose > 0:
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

def get_all_epochs(t):
    """
    Helper function to get times of all epochs from a ref table.
    This is required because our previous approach 
    of simply taking the time array of the star with the most detections 
    fails for mosaicked catalogs, because it is then possible that 
    no star is detected in all fields.
    """
    nepochs = len(t['t'][0])

    # Loop through each time entry and get year
    # from a non-masked source
    all_epochs = []
    for ii in range(nepochs):
        tcol = t['t'][:,ii]

        good = np.where(np.isfinite(tcol))[0][0]

        all_epochs.append(t['t'][good,ii])

    all_epochs = np.array(all_epochs)
    return all_epochs
    

def setup_ref_table_from_starlist(star_list):
    """ 
    Start with the reference list.... this will change and grow
    over time, so make a copy that we will keep updating.
    The reference table will contain one columne for every named
    array in the original reference star list.
    """
    col_arrays = {}
    motion_model_col_names = motion_model.get_all_motion_model_param_names(with_errors=True)
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

    # Make sure ref_table has the necessary x0, y0, m0 and associated
    # error columns. If they don't exist, then add them as a copy of
    # the original x,y,m etc columns. 
    new_cols_arr = ['x0', 'x0_err', 'y0', 'y0_err', 'm0', 'm0_err']
    orig_cols_arr = ['x', 'xe', 'y', 'ye', 'm', 'me']
    assert len(new_cols_arr) == len(orig_cols_arr)
    ref_cols = ref_table.keys()

    for ii in range(len(new_cols_arr)):
        if not new_cols_arr[ii] in ref_cols:
            # Some munging to convert data shape from (N,1) to (N,),
            # since these are all 1D cols
            vals = np.transpose(np.array(ref_table[orig_cols_arr[ii]]))[0]

            # Now add to ref_table
            new_col = Column(vals, name=new_cols_arr[ii])
            ref_table.add_column(new_col)
    
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
        The indices into the star_list or star_lsit_T where values are copied from.
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

def add_rows_for_new_stars(ref_table, star_list, idx_lis, default_motion_model='Fixed'):
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
            elif col_name=='motion_model_input':
                new_col_empty = default_motion_model
            elif col_name=='motion_model_used':
                new_col_empty = 'None'
            else:
                new_col_empty = np.nan
            
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

"""
Functions specific to OB120169 moved to align_old_functions,py
"""

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
    #assert len(x1m) >= req_match, 'Failed to find at least '+str(req_match)+' (only ' + str(len(x1m)) + ') matches, giving up'
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


# Transform_from_file original version moved to align_old_functions.py
# This version makes the transFile an object and uses transform_from_object
def transform_from_file(starlist, transFile):
    """
    Apply transformation from transFile to starlist. Returns astropy table with
    added columns with the transformed coordinates. NOTE: Transforms
    positions/position errors, plus velocities and velocity errors if they
    are present in starlist.

    WARNING: THIS CODE WORKS FOR POLYTRANSFORM
    
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
    # Make transform object
    trans_table = Table.read(transFile, format='ascii.commented_header', header_start=-1)
    Xcoeff = trans_table['Xcoeff']
    Ycoeff = trans_table['Ycoeff']
    # First determine the order based on the number of terms
    # Comes from Nterms = (N+1)*(N+2) / 2.
    order = (np.sqrt(1 + 8*len(Xcoeff)) - 3) / 2.
    if order%1 != 0:
        print( 'Incorrect number of coefficients for polynomial')
        print( 'Stopping')
        return
    order = int(order)
    # Do transform
    transform = transforms.PolyTransform(order, Xcoeff, Ycoeff)
    return transform_from_object(starlist, transform)
    
    

def transform_from_object(starlist, transform):
    """
    Apply transformation to starlist. Returns astropy table with
    transformed positions/position errors, velocities and velocity errors
    if they are present in starlits. If a more complex motion_model is
    implemented, the motion parameters are set to nan, as we need the full time
    series to refit.
    
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

    # Check to see if velocities or motion_model are present in starlist.
    vel = ('vx' in keys)and ~("motion_model_assigned" in keys)
    mot = ("motion_model_assigned" in keys)
    # If the only motion models used are Fixed and Linear, we can still transform velocities.
    if mot:
        motion_models_unique = list(np.unique(starlist_f['motion_model_assigned']))
        if 'Linear' in motion_models_unique:
            motion_models_unique.remove('Linear')
        if 'Fixed' in motion_models_unique:
            motion_models_unique.remove('Fixed')
        if len(motion_models_unique)==0:
            vel=True
            mot=False
    
    # Prior code before motion_model implementation
    # Can still be used as shortcut for Linear+Fixed motion_model only
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
        x0e = starlist_f['x0_err']
        y0e = starlist_f['y0_err']
        vx = starlist_f['vx']
        vy = starlist_f['vy']
        vxe = starlist_f['vx_err']
        vye = starlist_f['vy_err']
        
    # calculate the transformed position and velocity
    x_new, y_new, xe_new, ye_new = position_transform_from_object(x, y, xe, ye, transform)

    if vel:
        x0_new, y0_new, x0e_new, y0e_new = position_transform_from_object(x0, y0, x0e, y0e, transform)
        vx_new, vy_new, vxe_new, vye_new = velocity_transform_from_object(x0, y0, x0e, y0e, vx, vy, vxe, vye, transform)

    # update transformed coords to copy of astropy table
    starlist_f['x'] = x_new
    starlist_f['y'] = y_new
    starlist_f['xe'] = xe_new
    starlist_f['ye'] = ye_new
    
    if vel:
        starlist_f['x0'] = x0_new
        starlist_f['y0'] = y0_new
        starlist_f['x0_err'] = x0e_new
        starlist_f['y0_err'] = y0e_new
        starlist_f['vx'] = vx_new
        starlist_f['vy'] = vy_new
        starlist_f['vx_err'] = vxe_new
        starlist_f['vy_err'] = vye_new
            
    # For more complicated motion_models,
    #  we can't easily transform them, set the values to nans and refit later.
    if mot:
        motion_model_params = motion_model.get_all_motion_model_param_names()
        for param in motion_model_params:
            if param in keys:
                starlist_f[param] = np.nan
        
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
    THIS IS WRONG BELOW! - NOTE: I don't think this is wrong any more

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
                        ignore_contains='star', verbose=True, n_req_match=3,
                            mag_trans=True, order=1):
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
        x2m = ref_list['x'][idx_r][ndx_r]
        y2m = ref_list['y'][idx_r][ndx_r]
        m2m = ref_list['m'][idx_r][ndx_r]
        N = len(x1m)

    else:
        # Default is miracle match.
        briteN = min(50, len(star_list))

        # If there are velocities in the reference list, use them.
        # We assume velocities are in the same units as the positions.
        xref, yref = get_pos_at_time(star_list['t'][0], ref_list)
        if 'm' in ref_list.colnames:
            mref = ref_list['m']
        else:
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
    assert len(x1m) >= n_req_match, err_msg
    if verbose > 1:
        print('initial_guess: {0:d} stars matched between starlist and reference list'.format(N))

    # Calculate position transformation based on matches
    if ('order' in trans_args) and (trans_args['order'] == 0):
        order = 0
    else:
        order = order

    trans = transforms.PolyTransform.derive_transform(x1m, y1m ,x2m, y2m, order=order, weights=None)

    # Calculate flux transformation based on matches. If desired, should be applied as
    #     m' = m + mag_offset
    # where m is the original magnitude and m' is in the reference frame mag system.
    if mag_trans:
        trans.mag_offset = np.mean(m2m - m1m)
    else:
        trans.mag_offset = 0
        
    if verbose > 1:
        print('init guess: ', trans.px.parameters, trans.py.parameters)

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
        new_cols += ['x0_err']
    if 'ye' in star_list.colnames:
        old_cols += ['ye']
        new_cols += ['y0_err']
    if 'me' in star_list.colnames:
        old_cols += ['me']
        new_cols += ['m0_err']
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

    if verbose > 0:
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

# TODO: I think this is a startable, not a starlist
def get_pos_at_time(t, starlist):
    """
    Take a starlist, check to see if it has motion/velocity columns.
    If it does, then propogate the positions forward in time 
    to the desired epoch. If no motion/velocities exist, then just
    use ['x0', 'y0'] or ['x', 'y']

    Inputs
    ----------
    t_array : float
        The time to propogate to. Usually in decimal years; 
        but it should be in the same units
        as the 't0' column in starlist.
    """
    # Check for motion model
    if 'motion_model_used' in starlist.colnames:
        x,y,xe,ye = starlist.get_star_positions_at_time(t)
    # If no motion model, check for velocities
    elif ('vx' in starlist.colnames) and ('vy' in starlist.colnames) and ('x0' in starlist.colnames) and ('y0' in starlist.colnames):
        x = starlist['x0'] + np.nan_to_num(starlist['vx'])*(t-starlist['t0'])
        y = starlist['y0'] + np.nan_to_num(starlist['vy'])*(t-starlist['t0'])
    # If no velocities, try fitted positon
    elif ('x0' in starlist.colnames) and ('y0' in starlist.colnames):
        x = starlist['x0']
        y = starlist['y0']
    # Otherwise, use measured position
    else:
        x = starlist['x']
        y = starlist['y']
        
    return (x, y)

def logger(logfile, message, verbose = 9):
    if verbose > 4:
        print(message)
    logfile.write(message + '\n')
    return
