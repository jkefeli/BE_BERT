#Example shell script for multi-class diagnosis classifier, sectioned reports 

CUDA_VISIBLE_DEVICES=0 nohup python3 train_gi_multi.py gi_bigbird_2048 0 bigbird 2 2048 lower_lr f2_beta 30 gi_multiclass_tvt_data/ > gi_bigbird_2048_bs2_rs0_30e_lowerLR_f2beta.txt  2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python3 train_gi_multi.py gi_bigbird_2048 300 bigbird 2 2048 lower_lr f2_beta 30 gi_multiclass_tvt_data/ > gi_bigbird_2048_bs2_rs300_30e_lowerLR_f2beta.txt  2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python3 train_gi_multi.py gi_bigbird_2048 1000 bigbird 2 2048 lower_lr f2_beta 30 gi_multiclass_tvt_data/ > gi_bigbird_2048_bs2_rs1000_30e_lowerLR_f2beta.txt  2>&1 &
