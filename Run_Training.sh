#!/bin/bash
scenario='cross_validation'  # ['unknown_setting', 'cross_validation']

TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=128
TEST_BATCH_SIZE=128

LR=0.0001
LOG_INTERVAL=20
NUM_EPOCHS=50
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


if [ "$scenario" == "cross_validation" ]; then
    kfold_data_path='./Davis/5fold_data/'
    save_preprocessed_data_path='./Davis/preprocessing/'
    save_results_path='./Davis/results/'

    python preprocessing_data.py $scenario $kfold_data_path $save_preprocessed_data_path &&
    python train.py $scenario $cuda_name $save_preprocessed_data_path $save_results_path $TRAIN_BATCH_SIZE $VAL_BATCH_SIZE $TEST_BATCH_SIZE $LR $LOG_INTERVAL $NUM_EPOCHS $num_features_xd $num_head1 $num_head2 $embed_dim $dropout $hidden_size $kernel1 $kernel2 $kernel3 $dropout_cnn $cnn_hid_dim $num_layers


elif [ "$scenario" == "unknown_setting" ]; then
    setting='compound_clustering_unknown' # ['compound_clustering_unknown', 'protein_clustering_unknown', 'both_unknown']
    cluster_threshold=0.5
    
    # dataset path
    dataset_path='./BindingDB/BindingDB.txt'

    # save path of prepared files for clustering
    save_prepared_files_path='./BindingDB/files_preparation/'
    
    # save path of compound and protein clusters
    save_cluster_path='./BindingDB/'

    # split 5fold data based on clusters
    save_kfold_data_path='./BindingDB/5fold_data/'  

    # save path of preprocessed data
    save_preprocessed_data_path='./BindingDB/preprocessing/'

    # save path of training model results
    save_results_path='./BindingDB/results/'
    
    python file_preparation.py $dataset_path $save_prepared_files_path &&
    python clustering.py $save_prepared_files_path $save_cluster_path &&
    python split_dataset_according_clusting.py $save_prepared_files_path $save_cluster_path $save_kfold_data_path $setting $cluster_threshold &&
    python preprocessing_data.py $scenario $save_kfold_data_path $save_preprocessed_data_path $cluster_threshold &&
    python train.py $scenario $setting $cluster_threshold $cuda_name $save_preprocessed_data_path $save_results_path $TRAIN_BATCH_SIZE $VAL_BATCH_SIZE $TEST_BATCH_SIZE $LR $LOG_INTERVAL $NUM_EPOCHS $num_features_xd $num_head1 $num_head2 $embed_dim $dropout $hidden_size $kernel1 $kernel2 $kernel3 $dropout_cnn $cnn_hid_dim $num_layers
fi
