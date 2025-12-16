#!/bin/sh
#------ qsub option --------#
#PBS -q lg
#PBS -l select=2:ngpus=4
#PBS -l walltime=8:00:00
#PBS -W group_list=c30636
#PBS -j oe


cd /work/c30636/hit
conda activate hde

torchrun --nnodes=2 --nproc_per_node=4 ./main.py --config ./configs/dit-xl_IN1K.yaml