#!/bin/sh
#------ qsub option --------#
#PBS -q lg
#PBS -l select=2:ngpus=4:mpiprocs=4
#PBS -l walltime=8:00:00
#PBS -W group_list=c30636
#PBS -j oe

# 1. 加载模块
module load gcc ompi

# 2. 获取主节点 IP
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

# 3. [关键配置] NCCL / 网络设置
# 强制 GLOO (握手) 和 NCCL (数据) 使用 InfiniBand 网络接口 ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_IFNAME=ib0
# 开启 NCCL INFO 日志，运行后查看输出日志中是否有 "NET/IB" 字样，确认是否使用了 InfiniBand
export NCCL_DEBUG=INFO
# 显式指定使用 H100 的 RDMA 网卡 (通常是 mlx5_x)
export NCCL_IB_HCA=mlx5

echo "Master Node: $MASTER_ADDR"

# >>> conda initialize >>>
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

# 4. [核心修改] 添加 --bind-to none
# 如果不加这一行，mpirun 可能会把 torchrun 限制在单个 CPU 核上，
# 导致该节点下的 4 个 GPU 进程抢占 CPU，速度极慢。
mpirun -np 2 \
    --map-by ppr:1:node \
    --bind-to none \
    -x MASTER_ADDR \
    -x MASTER_PORT \
    -x GLOO_SOCKET_IFNAME \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_HCA \
    -x NCCL_DEBUG \
    -x PATH \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./main.py --config ./configs/dit-xl_IN1K.yaml