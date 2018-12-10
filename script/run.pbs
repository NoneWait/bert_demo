#!/bin/bash
#PBS -N delaiQ
#PBS -q gpu
#PBS -l nodes=03
#PBS -j oe

cd /home/lxw/delaiQ/bert/

# source activate tensorflowgpu
source activate tf_1.9
# ldconfig -p|grep libcuda
# cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lxw/cuda-8.0/lib64/
# export CUDA_HOME=/home/lxw/cuda-8.0
# export PATH=$PATH:/home/lxw/cuda-8.0/bin

ldconfig -p|grep libcuda
nvidia-smi
export PYTHONPATH=${PYTHONPATH}:`pwd`
# export CUDA_VISIBLE_DEVICES=0
home=/home/lxw/delaiQ
bert_home=/home/lxw/delaiQ/data/bert/chinese_L-12_H-768_A-12
python demo/run_my.py --vocab_file=$bert_home/vocab.txt \
    --bert_config_file=$bert_home/bert_config.json \
    --init_checkpoint=$bert_home/bert_model.ckpt \
    --do_train=True \
    --train_file=$home/data/aic_datasets/train_data_p.json \
    --do_predict=True \
    --predict_file=$home/data/aic_datasets/dev_data_p.json \
    --train_batch_size=12 \
    --learning_rate=3e-5 \
    -num_train_epochs=10.0 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=$home/bert/tmp/aic_base





