
# nohup sh train.sh &
snapshot="../../../vinai/khanhpd4/PFLD-pytorch-v2/checkpoint/snapshot"
log_file="../../../vinai/khanhpd4/PFLD-pytorch-v2/checkpoint/train.logs"
tensorboard="../../../vinai/khanhpd4/PFLD-pytorch-v2/checkpoint/tensorboard"
model_path="./checkpoints_landmark/mobilenetv3/snapshot/checkpoint_epoch_301.pth.tar"
logs="log.txt"

CUDA_VISIBLE_DEVICES=0\
nohup python3 -u train_v3.py --snapshot {snapshot} 
                             --log_file {log_file}
                             --tensorboard {tensorboard}
                             --model_path {model_path}
                             --train_batchsize 256
                             > ${logs} 2>&1 &
tail -f ${logs}