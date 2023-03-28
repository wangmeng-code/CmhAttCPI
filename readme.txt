cross_validation:

./
1.preprocessing_data.py
2.train.py

unseen setting validation:
./unseen setting/
1.file_preparation.py
2.clustering.py
3.split_dataset_according_clustering.py
4.preprocessing_data.py (scenario = 'unseen setting', cluster_cross_validation_setting and cluster_threshold should be set a fixed value from ['compound clustering', 'protein clustering'] and [0.3,0.4,0.5], respectively)
5.train.py(scenario = 'unseen setting', cluster_cross_validation_setting and cluster_threshold should be set a fixed value from ['compound clustering', 'protein clustering'] and [0.3,0.4,0.5], respectively)


## A bidirectional Interpretable compound-protein interaction prediction framework based on cross attention

## Scripts
# Performance evaluation on BindingDB, DrugBank and GPCR datasets
1. './preprocessing_data.py': converting SMILES to molecular graph and constructing word dictionary for protein sequence 
2. './train.py': training model

# Generalization evaluation in the unseen setting for cluster-cross validation
1. './unseen setting/file_preparation.py': preparating files for clustering
2. './unseen setting/clustering.py': clustering for compounds and proteins
3. './unseen setting/split_dataset_according_clustering.py': spliting dataset based on clustering
4. './preprocessing_data.py': scenario should be 'unseen setting', cluster_cross_validation_setting and cluster_threshold should be set a fixed value from ['compound clustering', 'protein clustering'] and [0.3,0.4,0.5], respectively
5. './train.py': scenario should be 'unseen setting', cluster_cross_validation_setting and cluster_threshold should be set a fixed value from ['compound clustering', 'protein clustering'] and [0.3,0.4,0.5], respectively
