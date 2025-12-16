#!/bin/sh
#PBS -q lg
#PBS -l select=2:ngpus=4:mpiprocs=4
#PBS -l walltime=8:00:00
#PBS -W group_list=c30636
#PBS -j oe

# 2. 新增: 加载必要的系统模块以使用 mpirun 
module load gcc ompi

# 3. 新增: 获取主节点 IP 地址 (用于多节点通信) [cite: 1841, 1876]
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500  # 指定一个空闲端口
echo "Master Node: $MASTER_ADDR"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/c30746/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/c30746/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/c30746/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd /work/c30636/hit
conda activate hde

# 4. 修改: 使用 mpirun 启动 torchrun
# -np 2: 总共启动 2 个 torchrun 进程 (因为有 2 个节点)
# --map-by ppr:1:node: 每个节点启动 1 个 torchrun 进程
# torchrun 参数配置为使用 c10d 后端并指向 MASTER_ADDR
mpirun -np 2 --map-by ppr:1:node \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./main.py --config ./configs/dit-xl_IN1K.yaml