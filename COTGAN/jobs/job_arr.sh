#!/bin/bash
for i in job_test*.slurm; do
    echo "sbatch $i"
    sbatch $i
done