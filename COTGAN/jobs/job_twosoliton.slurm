#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="cotgan_tune"
#SBATCH --output=test5.out
#SBATCH --mail-user=eivind.kohmann@gmail.com
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


cd ${SLURM_SUBMIT_DIR}/..

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt

python3 trainer.py --model cotgan --n_epochs 1200 --batch_size 72 --l_rate 0.0005 --l_rate_g 0.0005 --use_opt_scheduler False --scheduler_rule stepwise --use_bn False --reg_lam 0.01 --sinkhorn_eps 10.0 --sinkhorn_l 200 --dataset sinus --max_seq_len 100 --trainset_size 1536 --testset_size 1000 --gen_rnn_num_layers 1 --gen_rnn_hidden_dim 64 --dis_rnn_num_layers 1 --dis_rnn_hidden_dim 64 --hidden_dim 64 --num_hidden_layers 2 --Z_dim 50 --J_dim 32 --device cuda