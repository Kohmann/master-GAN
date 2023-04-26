#!/bin/bash
for i in job[0-5].slurm; do
    echo "sbatch $i"
    sbatch $i
done