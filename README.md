### A bidirectional Interpretable compound-protein interaction prediction framework based on cross attention

### Scripts
## 1. Performance evaluation on BindingDB, DrugBank, GPCR and Davis datasets
-step1-  './preprocessing_data.py': converting SMILES to molecular graph and constructing word dictionary for protein sequence (scenario should be 'cross_validation')
run: python preprocessing_data.py

-step2-  './train.py': training model (scenario should be 'cross_validation')
run: python train.py

# 1.1 Training model on user-provided datasets
-step1- Users need to disposal data format (user-dataset.txt) as follows:

+----------------+--------------+-----+
|SMILES          |Sequence      |Label|
+----------------+--------------+-----+

-step2- Splitting dataset. run:python kfold_split_data.py
-step3-  run:python preprocessing_data.py ('DATASET' should be set to 'user-dataset', 'scenario' should be set to 'cross_validation')
-step4-  run:python train.py

# 1.2 Applying pre-trained model on user-provided datasets
-step1-  Users need to disposal data format as exhibted above 1.1 step1 and save the data in fold './Predicting_user-dataset/prediction_data.txt'
-step2-  run:python preprocessing_user_test_data.py
-step3-  run:python predict.py 


## 2. Generalization evaluation in the unseen setting for cluster-cross validation
-step1-  './unseen_setting/file_preparation.py': preparating files for clustering. 
run: python file_preparation.py

-step2-  './unseen_setting/clustering.py': clustering for compounds and proteins. 
run: python clustering.py

-step3-  './unseen_setting/split_dataset_according_clustering.py': spliting dataset into train/val/test set based on clustering. 
run:python split_dataset_according_clustering.py

-step4-  './preprocessing_data.py': 'scenario' should be 'unseen_setting', 'cluster_cross_validation_setting' and 'cluster_threshold' should be set a fixed value from ['compound_clustering_unseen', 'protein_cluster_unseen', 'both_unseen'] and [0.3,0.4,0.5], respectively
run: preprocessing_data.py

-step5-  './train.py': 'scenario' should be 'unseen_setting', 'setting' and 'cluster_threshold' should be set a fixed value from ['compound_clustering_unseen', 'protein_cluster_unseen', 'both_unseen'] and [0.3,0.4,0.5], respectively
run:train.py

# 2.1 Training model on user-provided datasets
-step1- Users need to disposal data format as exhibted above 1.1(1) step1.

Splitting data into train/val/test set based on clustering:
-step2- run:file_preparation.py ('DATASET' set to 'user-dataset')
-step3- run:clustering.py 
-step4- run:split_dataset_according_clustering.py 

Traing model:
-step5-  After splitting data based on cluster then preprocessing data. run:python preprocessing_data.py ('DATASET' set to 'user-dataset', 'scenario' to 'unseen_setting','cluster_cross_validation_setting' and 'cluster_threshold' should be set a fixed value from ['compound_clustering_unseen', 'protein_cluster_unseen', 'both_unseen'] and [0.3,0.4,0.5], respectively)
-step6-  Training model. run:python train.py 

# 2.2 Applying pre-trained model on user-provided datasets
-step1-  Users need to disposal data format as exhibted above 1.1 step1 and save the data in fold './Predicting_user-dataset/prediction_data.txt'
-step2-  run:python preprocessing_user_test_data.py
-step3-  run:python predict.py 

### 2.Data
## 5 fold data of BindingDB, DrugBank, GPCR and Davis
./cross_validation/BindingDB/5fold_data/*.txt

./cross_validation/DrugBank/5fold_data/*.txt

./cross_validation/GPCR/5fold_data/*.txt

./cross_validation/Davis/5fold_data/*.txt

The original datset BindingDB.txt, DrugBank.txt, GPCR.txt and Davis.txt are provided at https://drive.google.com/drive/folders/1lfVZNHlpgdlBhozK_oeS1upU8NvpBOx6?usp=sharing
