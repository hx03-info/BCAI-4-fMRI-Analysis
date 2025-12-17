# This python file is for
# Searchlight MVPA code for Recall Prediction

import numpy as np
import nibabel as nib
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import balanced_accuracy_score

BEST_C = 1

# STEP 1 READ-IN DATA and neighbors ==========================
# data
final_recall = np.load('data/free_recall_mat_filtered.npy')         # [n_subs, n_pics]
final_data = np.load('data/full_encoding_data_flat_filtered.npy')   # [n_subs, n_pics, n_voxels]
n_subs, n_pics, n_voxels = final_data.shape
print("z-scoring...")
X_norm = np.zeros_like(final_data)
# normalize
for s in range(n_subs):
    sub_dat = final_data[s, :, :]
    # mean and std across pic
    mean_vec = np.mean(sub_dat, axis=0)
    std_vec = np.std(sub_dat, axis=0)
    std_vec[std_vec == 0] = 1.0
    X_norm[s, :, :] = (sub_dat - mean_vec) / std_vec
# get searchlight index
mask_img = nib.load("atlas/mask_all_valid_voxels.nii.gz")
mask_data = mask_img.get_fdata()
mask_bool = mask_data.astype(bool) 
mask_coords = np.where(mask_data != 0)
mask_coords = np.vstack(mask_coords).T # (n_voxels, 3) 3 for x,y,z
# use KNN to find neighbors
radius_vox = 3.1 # in case of float data
nn = NearestNeighbors(radius=radius_vox)
nn.fit(mask_coords)
# radius_neighbors return each voxel's neighbors' row index in mask_coords
# aligh to beta[mask_bool, :, :]
neighbor_indices = nn.radius_neighbors(mask_coords, return_distance=False)
print(f"searchlight neighbors get: {len(neighbor_indices)}")


# STEP 2 PREPARE FOR SEARCHLIGHT ==========================
X_flat = X_norm.reshape(n_subs * n_pics, n_voxels)  # [n_subs*n_pics, n_voxels]
y_flat = final_recall.reshape(n_subs * n_pics)
groups = np.repeat(np.arange(n_subs), n_pics)

# STEP 3a Searchlight functions and sample split ==========================
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
def run_final_stats_kernel(voxel_idx, neighbor_idxs, X_full, y, groups, c_param):
    """
    return beta_acc list
    """
    X_sphere = X_full[:, neighbor_idxs]
    
    n_unique_subs = len(np.unique(groups))
    sub_scores = np.zeros(n_unique_subs)
    if np.std(X_sphere) == 0:
        return voxel_idx, np.full(n_unique_subs, 0.5)
    
    # 5-fold
    for train_idx, test_idx in gkf.split(X_sphere, y, groups):
        
        X_train, y_train = X_sphere[train_idx], y[train_idx]
        clf = LogisticRegression(
            C=c_param, 
            penalty='l2', 
            solver='liblinear', 
            class_weight='balanced',
            max_iter=100
        )
        clf.fit(X_train, y_train)
        
        # get current test fold's subject IDs
        test_group_ids = groups[test_idx]
        current_fold_subs = np.unique(test_group_ids)
        
        for sub_id in current_fold_subs:
            # get subid mask from previous group mask
            sub_mask = (groups == sub_id)
            
            X_test_sub = X_sphere[sub_mask,:]
            y_test_sub = y[sub_mask]
            
            # calc accuracy
            try:
                y_pred = clf.predict(X_test_sub)
                score = balanced_accuracy_score(y_test_sub, y_pred)
            except:
                score = 0.5
            
            # put into sub_scores
            sub_scores[sub_id] = score
            
    return voxel_idx, sub_scores

# STEP 3b Searchlight parallel run ==========================
results = Parallel(n_jobs=16, verbose=5)(
    delayed(run_final_stats_kernel)(
        i, idxs, X_flat, y_flat, groups, BEST_C
    ) 
    for i, idxs in enumerate(neighbor_indices)
)


# STEP 4 t-test on balanced accuracy ==========================
mean_acc_data = np.zeros(n_voxels)
t_val_data = np.zeros(n_voxels)
p_val_data = np.ones(n_voxels)

for v_idx, scores in results:
    scores = np.array(scores)
    
    # 1. mean bAcc
    mean_acc_data[v_idx] = np.mean(scores)
    
    # 2. one-sided accuracy compared to chance (0.5)
    if np.std(scores) == 0:
        t_stat = 0
        p_val = 1.0
    else:
        t_stat, p_val = ttest_1samp(scores, popmean=0.5, alternative='greater')
        
    # 3. handle nan
    if np.isnan(t_stat):
        t_stat = 0
        p_val = 1.0
        
    t_val_data[v_idx] = t_stat
    p_val_data[v_idx] = p_val

# STEP 5 save to nifti ==========================
def save_nii(data, name, mask_bool=mask_bool):
    vol = np.full(mask_img.shape, np.nan, dtype=np.float32)
    vol[mask_bool] = data
    img = nib.Nifti1Image(vol, mask_img.affine)
    nib.save(img, name)
    print(f"Saved: {name}")
save_nii(mean_acc_data, f'Final_Mean_BalAcc_C{BEST_C}.nii.gz')
save_nii(t_val_data,    f'Final_T_Map_C{BEST_C}.nii.gz')
save_nii(p_val_data,    f'Final_P_Map_C{BEST_C}.nii.gz')