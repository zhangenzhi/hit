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

# === 调试步骤：打印当前节点的网络接口信息 ===
# 如果再次报错，请把输出日志中 "ip addr" 的结果发给我
echo "=== Network Interfaces info ==="
ip addr | grep inet
echo "=============================="

# 3. [核心修正] 网络分离配置

# A. 握手与控制 (Socket)：不要用 ib0，改用以太网
# "^lo,docker0" 意思是：使用除了本地回环和docker以外的第一个可用网卡
export GLOO_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=^lo,docker0

# B. 数据传输 (RDMA)：强制使用 InfiniBand 高速通道
# H100 通常使用 mlx5 系列网卡
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=0    # 确保开启 IB
export NCCL_P2P_DISABLE=0   # 确保开启 P2P

# 开启 INFO 日志，验证是否成功检测到 IB (NET/IB)
export NCCL_DEBUG=INFO

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

# 4. 启动命令 (保持 --bind-to none 以避免 CPU 单核瓶颈)
mpirun -np 2 \
    --map-by ppr:1:node \
    --bind-to none \
    -x MASTER_ADDR \
    -x MASTER_PORT \
    -x GLOO_SOCKET_IFNAME \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_HCA \
    -x NCCL_IB_DISABLE \
    -x NCCL_P2P_DISABLE \
    -x NCCL_DEBUG \
    -x PATH \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./main.py --config ./configs/dit-xl_IN1K.yaml