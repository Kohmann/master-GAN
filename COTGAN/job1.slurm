#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="cotgan_test"
#SBATCH --output=test.out
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


cd ${SLURM_SUBMIT_DIR}/yolov7

module purge
module load Python/3.8.6-GCCcore-10.2.0

pip install -r requirements.txt

python3 trainer.py --n_epochs 20 --l_rate 0.001 --batch_size 32 --max_seq_len 25 --trainset_size 1536 --testset_size 768 --gen_rnn_num_layers 2 --gen_rnn_hidden_dim 64 --dis_rnn_num_layers 2 --hidden_dim 128 --num_hidden_layers 3 --Z_dim 10 --scaling_coef 1.0 --sinkhorn_eps 0.8 --sinkhorn_l 100 --reg_lam 0.01 --device cuda --seed 1