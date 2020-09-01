# nohup sh train.sh &
python3 train.py --snapshot ../../../vinai/khanhpd4/PFLD-pytorch/checkpoint/snapshot 
        --log_file ../../../vinai/khanhpd4/PFLD-pytorch/checkpoint/train.logs 
        --tensorboard ../../../vinai/khanhpd4/PFLD-pytorch/checkpoint/tensorboard