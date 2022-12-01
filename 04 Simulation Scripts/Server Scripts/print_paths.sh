#!/bin/bash
#
#
#SBATCH --account=nklab          # The account name for the job.
#SBATCH --job-name=print    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:05:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=2gb        # The memory the job will use per cpu core.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hs3110@columbia.edu
#SBATCH --output=slurm-print_%A.out


module load shared
df -hl
