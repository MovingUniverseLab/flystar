import pickle
import flystar
import matplotlib.pyplot as plt
from flystar import align, transforms, motion_model
from flystar.plots import plot_stars

branch = 'mm_rework'  # 'mm_compare' or 'mm_rework'

test_data_path = f'{flystar.__path__[0]}/tests/test_data'

with open(f'{test_data_path}/my_gaia.pkl', 'rb') as f:
    my_gaia = pickle.load(f)
with open(f'{test_data_path}/list_of_starlists.pkl', 'rb') as f:
    list_of_starlists = pickle.load(f)
ra_deg, dec_deg = 18.0, -30.0
my_gaia.remove_column('motion_model_used')
if branch == 'mm_compare':
    msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=3,
                        dr_tol=[0.2, 0.1, 0.08], dm_tol=[5,5,5],
                        outlier_tol=[None, None, 3], mag_lim=[6, 20],
                        trans_class=transforms.PolyTransform,
                        trans_args=[{'order': 1}, {'order': 1}, {'order': 1}], 
                        motion_models=['Fixed'],
                        fixed_params_dict = {'ra':ra_deg, 'dec':dec_deg, 'pa':0.0, 'obsLocation':'earth'},
                        use_ref_new=True,
                        update_ref_orig=False, 
                        mag_trans=True,
                        trans_weights='both,std',
                        init_guess_mode='name', verbose=3)
elif branch == 'mm_rework':
    my_gaia['motion_model_input'] = ['Fixed'] * len(my_gaia)
    msc = align.MosaicToRef(my_gaia, list_of_starlists, iters=3,
                        dr_tol=[0.2, 0.1, 0.08], dm_tol=[5,5,5],
                        outlier_tol=[None, None, 3], mag_lim=[6, 20],
                        trans_class=transforms.PolyTransform,
                        trans_args=[{'order': 1}, {'order': 1}, {'order': 1}], 
                        default_motion_model='Fixed',
                        # motion_model_dict = {'Parallax': motion_model.Parallax(RA=ra_deg, Dec=dec_deg, PA=0.0, obsLocation='earth')},
                        use_ref_new=True,
                        update_ref_orig=False, 
                        mag_trans=True,
                        trans_weights='both,std',
                        init_guess_mode='name', verbose=3)

msc.fit()

with open(f'{test_data_path}/ref_table_old.pkl', 'wb') as f:
    pickle.dump(msc.ref_table, f)

for i in range(msc.ref_table['x'].shape[1]):
    plt.scatter(msc.ref_table['x'][:, i], msc.ref_table['y'][:, i])
plt.show()
plot_stars(msc.ref_table, msc.ref_table['name'][:3])