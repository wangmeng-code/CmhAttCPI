### A bidirectional Interpretable compound-protein interaction prediction framework based on cross attention

### Scripts
## 1. Training model
# (1) 5-fold cross validation
# Usage
We provided two major scripts for training model:

-- ./preprocessing_data.py Converting SMILESs to molecular graphs and constructing word dictionary for protein sequences.

-- ./train.py Training model.

Train model with the following command:
```
bash Run_Train.sh
```
The parameter 'scenario' in Run_Train.sh should be set to 'cross_validation'.

# (2) cluster-cross validation
# Usage
-- ./file_preparation.py Preparating files for clustering. 

-- ./clustering.py Clustering for compounds and proteins. 

-- ./split_dataset_according_clustering.py Spliting dataset into train/val/test set based on clustering. 

-- ./preprocessing_data.py Converting SMILESs to molecular graphs and constructing word dictionary for protein sequences.

-- ./train.py Training model.

Train model with the following command:
```
bash Run_Train.sh
```
The parameter 'scenario' in Run_Train.sh should be set to 'unknown_setting'.


## 2. Applying pre-trained model on test data
# Usage
We provided two major scripts for predicting test data:

-- ./preprocessing_test_data.py Converting SMILESs of the test data to molecular graphs and constructing word dictionary for protein sequences of the test data.

-- ./predict.py Make predictions on the test data.

Predict on the test data with the following command:
```
bash Run_Predicting.sh
```


### Data
## Four CPI datasets: BindingDB, DrugBank, GPCR and Davis
The datsets BindingDB.txt, DrugBank.txt, GPCR.txt and Davis.txt used in this study are provided at https://drive.google.com/drive/folders/1lfVZNHlpgdlBhozK_oeS1upU8NvpBOx6?usp=sharing
