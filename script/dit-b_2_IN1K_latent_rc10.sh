#!/bin/sh
#------ qsub option --------#
#PBS -q sg
#PBS -l select=2:ngpus=8
#PBS -l walltime=30:00
#PBS -W group_list=group1
#PBS -j oe

cd ${PBS_O_WORKDIR}

torchrun --nnodes=2 --nproc_per_node=4 ./main.py --config ./configs/dit-b_IN1K.yaml