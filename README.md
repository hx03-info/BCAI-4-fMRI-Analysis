# Integrating Multi-Scale MVPA to Identify Neural Representations Predictive of Future Memory Recall

This repository hosts a reproducible computational pipeline designed to analyze fMRI data and identify neural correlates of successful free recall. It integrates multi-scale multivariate pattern analysis (MVPA) methods—including ROI-based and searchlight approaches—to process raw fMRI data, test regional and whole-brain neural patterns, and quantify their ability to predict subsequent memory performance.


## Repository Structure & Workflow
The pipeline is organized into sequential, numbered scripts/notebooks (0–5) to ensure clear data flow and reproducibility. Below is a detailed breakdown of each core component:

![Multi-ROI Logistic Regression Workflow](https://github.com/hx03-info/BCAI-4-fMRI-Analysis/blob/main/results/Workflow%20of%20Multi-ROI%20Logestic%20Regression.jpg?raw=true)
## Key Workflow Steps
1. Data Extraction
  - Extract voxel activation patterns from target ROIs in fMRI data
  - Filter trials with excessive missing values
2. Feature Compression
  - Apply MLP + Sigmoid to reduce ROI feature dimensionality
  - Normalize compressed features (z-score)

**`0_datapreprocess.ipynb`**
Handles raw fMRI data preprocessing, including missing value handling and z-score normalization. It also filters trials (e.g., removing stimuli with low recall rates) to reduce unwanted variance and improve the reliability of downstream analyses.


**`1_Batch_ROI_Calculation.ipynb`**
Performs single-region-of-interest (ROI) analyses using L2-regularized logistic regression. This script tests whether individual anatomical ROIs—such as visual cortex, hippocampus, and prefrontal regions—can independently predict recall success, validating a priori hypotheses about memory-related brain areas.


**`2_Mutilple-ROI_Logistic_Regression.ipynb`**
Builds a multi-region predictive model by integrating top-performing ROIs. It uses MLP feature compression to reduce dimensionality before applying logistic regression, with a total loss function incorporating dual regularization. This script also quantifies the relative contribution of each ROI to the overall recall prediction.


**`3_searchlight_recall_stat.py & 3_voxel-based_5d.ipynb`**
Conducts whole-brain searchlight MVPA with a 3-voxel radius sphere. The scripts process 5D fMRI data (subjects × trials × voxels) for voxel-level pattern analysis, using five-fold group cross-validation to map clusters that predict recall success.


**`4_accordance_schlight.ipynb`**
Measures pattern fidelity by calculating the Pearson correlation between individual neural patterns and group-level templates. It compares fidelity scores between recalled (R) and non-recalled (F) trials to link neural representation stability to memory success.


**`5_get_searchlight-level_feature.ipynb`**
Extracts and summarizes key features from searchlight results, including cluster peaks, prediction accuracy metrics, and voxel-wise significance values. These outputs are formatted for downstream visualization and statistical reporting in manuscripts.


## Key Methods
This pipeline combines two complementary MVPA frameworks to balance hypothesis-driven and data-driven discovery:
- ROI-based MVPA: Tests pre-specified anatomical regions (derived from the Harvard–Oxford atlas) to validate hypotheses about domain-specific memory processes, such as visual feature encoding (visual cortex) and episodic binding (hippocampus).
- 
- Searchlight MVPA: Enables unbiased whole-brain mapping by sliding a small spherical kernel across the entire brain. This approach identifies unexpected voxel clusters—such as attention-related networks—that predict recall, avoiding pre-selection bias.‘
  
Both frameworks use L2-regularized logistic regression (optimized for high-dimensional fMRI data) and five-fold group cross-validation to ensure generalizability to new subjects.


## Expected Outputs
Running the pipeline in sequential order (0 → 5) will generate the following core outputs:
- ROI-level prediction accuracy scores for individual brain regions and the integrated multi-region model.
- Whole-brain statistical maps (searchlight results) highlighting voxel clusters that significantly predict recall success.
- Pattern fidelity scores comparing recalled vs. non-recalled trials, linking neural stability to memory performance.
- Summary figures and tables (e.g., ROI contribution weights, cluster peak coordinates) formatted for academic manuscript submission.
