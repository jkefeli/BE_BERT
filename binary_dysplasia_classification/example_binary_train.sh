#Example shell script for running train_BE_binary.py with relevant input parameters 

CUDA_VISIBLE_DEVICES=0 nohup python3 train_BE_binary.py gi_bigbird_2048_binary_sectionedreports 0 bigbird 2 2048 lower_lr f2_beta 30  gi_binary_tvt_data/ \
 > gi_bigbird_2048_bs2_rs0_30e_lowerLR_f2beta_binary_eval_tvt_sectionedreports.txt  2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python3 train_BE_binary.py gi_bigbird_2048_binary_sectionedreports 300 bigbird 2 2048 lower_lr f2_beta 30 gi_binary_tvt_data/ \
 > gi_bigbird_2048_bs2_rs300_30e_lowerLR_f2beta_binary_eval_tvt_sectionedreports.txt  2>&1 &&
CUDA_VISIBLE_DEVICES=0 nohup python3 train_BE_binary.py gi_bigbird_2048_binary_sectionedreports 1000 bigbird 2 2048 lower_lr f2_beta 30 gi_binary_tvt_data/ \
 > gi_bigbird_2048_bs2_rs1000_30e_lowerLR_f2beta_binary_eval_tvt_sectionedreports.txt  2>&1 &
