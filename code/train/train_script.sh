#!/bin/bash
PROJECT_DIR=/home/sungj4/projects/digital-pathology
cd $PROJECT_DIR/code/train
# nohup bash $PROJECT_DIR/code/train/train_script.sh > $PROJECT_DIR/logs/log_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# tail -f $PROJECT_DIR/logs/log_<timestamp>.log # monitor
# watch -n 1 nvidia-smi
# pgrep -fl train_script.sh #(to grep)
# pkill -f train_script.sh #(to kill)
# ps aux | grep python

# model + loss > find best combination > Dataset 
# model_name (9) = 'ACMIL', 'CLAM_SB', 'CLAM_MB', 'TransMIL', 'DSMIL', 'MeanMIL', 'MaxMIL', 'ABMIL', 'GABMIL'
# loss (4) = 'coxph', 'rank', 'MSE', 'SurvPLE'
# ExternalDatasets = CHOL COAD LUAD

# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses rank,SurvPLE --num_loss 1 --repeat 0
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --repeat 2
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --learning_rate 1e-4 --repeat 3 # with updated continuous c-index
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --learning_rate 2e-4 --repeat 3 
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --learning_rate 3e-4 --repeat 3 
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --learning_rate 4e-4 --repeat 3 
# python survival.py --models CLAM_SB,CLAM_MB,TransMIL,DSMIL,MeanMIL,MaxMIL,ABMIL,GABMIL,ACMIL --losses coxph,MSE,rank,SurvPLE --num_loss 1 --learning_rate 5e-4 --repeat 3 
# python survival.py --models ABMIL --losses rank --learning_rate 1e-6,2e-6,5e-6,1e-5,2e-5,5e-5 --repeat 4 # removed lower batch_size with rank. maybe it is working ok now. 
# python survival.py --models ABMIL --losses rank --learning_rate 6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,5e-4  --repeat 5 # cv with repeat 5,6,7,8,9
# python survival.py --models ABMIL --losses rank --learning_rate 6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,5e-4 --repeat 6 # cv with repeat 5,6,7,8,9
# python survival.py --models ABMIL --losses rank --learning_rate 6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,5e-4 --repeat 7 # cv with repeat 5,6,7,8,9
# python survival.py --models ABMIL --losses rank --learning_rate 6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,5e-4 --repeat 8 # cv with repeat 5,6,7,8,9
# python survival.py --models ABMIL --losses rank --learning_rate 6e-6,8e-6,1e-5,2e-5,4e-5,6e-5,8e-5,1e-4,2e-4,3e-4,4e-4,5e-4 --repeat 9 # cv with repeat 5,6,7,8,9
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,1e-4,1.2e-4 --repeat 10 # with external data
# batch_size test
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 11 # batch_size = random.randint(8, 32)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 12 # batch_size = random.randint(12, 48)
# # external data, batch_size (8, 32)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4,1.1e-4 --repeat 13 # with external data
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4,1.1e-4 --repeat 14 --ExternalDatasets CHOL # with external data
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4,1.1e-4 --repeat 15 --ExternalDatasets COAD # with external data
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4,1.1e-4 --repeat 16 --ExternalDatasets LUAD # with external data

# # external data, batch_size (16, 64), use initial test_cindex_min to check intial condition
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 21 # without external data (PDAC 202)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 22 --ExternalDatasets CHOL # CHOL (39)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 23 --ExternalDatasets COAD # COAD (438)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,9e-5,1e-4 --repeat 24 --ExternalDatasets LUAD # LUAD (521)

# external data, More datapoints. add max 160 cells or entire if less, batch_size (16, 64)
# python survival.py --models ABMIL --losses rank --learning_rate 8.2e-5,8.4e-5,8.6e-5,8.8e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5 --repeat 21 # without external data (PDAC 202)
# python survival.py --models ABMIL --losses rank --learning_rate 8.2e-5,8.4e-5,8.6e-5,8.8e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5 --repeat 22 --ExternalDatasets CHOL # CHOL (39)
# python survival.py --models ABMIL --losses rank --learning_rate 8.2e-5,8.4e-5,8.6e-5,8.8e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5 --repeat 23 --ExternalDatasets COAD # COAD (438)
# python survival.py --models ABMIL --losses rank --learning_rate 8.2e-5,8.4e-5,8.6e-5,8.8e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5 --repeat 24 --ExternalDatasets LUAD # LUAD (521)

# python survival.py --models ABMIL --losses rank --learning_rate 8.9e-5,9.1e-5 --repeat 21 # without external data (PDAC 202)
# python survival.py --models ABMIL --losses rank --learning_rate 8.6e-5,9.2e-5 --repeat 22 --ExternalDatasets CHOL # CHOL (39)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,8.2e-5,8.4e-5,8.6e-5,8.8e-5,9e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5,1e-4 --repeat 25 --ExternalDatasets LIHC # LIHC (378)
# python survival.py --models ABMIL --losses rank --learning_rate 8e-5,8.2e-5,8.4e-5,8.6e-5,8.8e-5,9e-5,9.2e-5,9.4e-5,9.6e-5,9.8e-5,1e-4 --repeat 26 --ExternalDatasets ESCA # ESCA (158)
# python survival.py --models ABMIL --losses rank --learning_rate 9.8e-5,1e-4 --repeat 26 --ExternalDatasets ESCA # ESCA (158)
# python survival.py --models ABMIL --losses rank --learning_rate 8.1e-5,8.3e-5,8.5e-5,8.7e-5,8.9e-5,9.1e-5,9.3e-5,9.5e-5,9.7e-5,9.9e-5 --repeat 22 --ExternalDatasets CHOL # CHOL (39)


# sudo shutdown -h now # auto shutdown vm