#!/bin/bash
PROJECT_DIR=/home/sungj4/projects/digital-pathology
cd $PROJECT_DIR/code/train
# nohup bash $PROJECT_DIR/code/train/train_script2.sh > $PROJECT_DIR/logs/log_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# tail -f $PROJECT_DIR/logs/log_<timestamp>.log # monitor
# watch -n 1 nvidia-smi
# pgrep -fl train_script.sh #(to grep)
# pkill -f train_script.sh #(to kill)
# ps aux | grep python

PID=32555
while ps -p $PID > /dev/null; do
    echo "Job $PID is still running. Waiting..."
    sleep 1800  # Check every 30 minute.
done

echo "Job $PID finished. Starting second job..."
python survival.py --models ABMIL --losses rank --learning_rate 2e-4,3e-4,4e-4,5e-4 --repeat 5 # cv with repeat 5,6,7,8,9
python survival.py --models ABMIL --losses rank --learning_rate 2e-4,3e-4,4e-4,5e-4 --repeat 6 # cv with repeat 5,6,7,8,9
python survival.py --models ABMIL --losses rank --learning_rate 2e-4,3e-4,4e-4,5e-4 --repeat 7 # cv with repeat 5,6,7,8,9
python survival.py --models ABMIL --losses rank --learning_rate 2e-4,3e-4,4e-4,5e-4 --repeat 8 # cv with repeat 5,6,7,8,9
python survival.py --models ABMIL --losses rank --learning_rate 2e-4,3e-4,4e-4,5e-4 --repeat 9 # cv with repeat 5,6,7,8,9
