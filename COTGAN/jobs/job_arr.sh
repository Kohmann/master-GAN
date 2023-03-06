#!/bin/bash
for i in job[0-4].slurm; do
    echo "sbatch $i"
    sbatch $i
done