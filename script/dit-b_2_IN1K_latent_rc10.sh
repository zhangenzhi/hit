#!/bin/sh
#------ qsub option --------#
#PBS -q lg
#PBS -l select=2:ngpus=4
#PBS -l walltime=8:00:00
#PBS -W group_list=c30636
#PBS -j oe

# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/c30746/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/c30746/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/c30746/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<

cd /work/c30636/hit
conda init
conda activate hde

torchrun --nnodes=2 --nproc_per_node=4 ./main.py --config ./configs/dit-xl_IN1K.yaml