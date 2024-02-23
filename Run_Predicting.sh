#!/bin/bash

# path of test data (user-provided datasets)
testdata_path='./Predicting_user-dataset/prediction_data.txt'

# paths of the preprocessed training data used for pretraining the model
atom_dict_path='./Davis/preprocessing/0fold_atom_dict'
bond_dict_path='./Davis/preprocessing/0fold_bond_dict'
word_dict_path='./Davis/preprocessing/0fold_word_dict'

# path of saving preprocessed test data 
save_preprocessed_data_path='./Predicting_user-dataset/preprocessing/'

# path of pre-trained model
model_path='./Davis/results/0fold_model.model'

# path of saving predicted results
save_results_path='./Predicting_user-dataset/results/'

TRAIN_BATCH_SIZE=128
cuda_name="cuda:1"
num_features_xd=82
num_head1=1
num_head2=4
embed_dim=128
dropout=0.2
hidden_size=64
kernel1=11
kernel2=15
kernel3=21
dropout_cnn=0
cnn_hid_dim=64
num_layers=2

python preprocessing_test_data.py $testdata_path $atom_dict_path $bond_dict_path $word_dict_path $save_preprocessed_data_path &&
python predict.py $cuda_name $save_preprocessed_data_path $model_path $save_results_path $TRAIN_BATCH_SIZE $num_features_xd $num_head1 $num_head2 $embed_dim $dropout $hidden_size $kernel1 $kernel2 $kernel3 $dropout_cnn $cnn_hid_dim $num_layers
