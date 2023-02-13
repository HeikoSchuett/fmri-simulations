#!/bin/bash
#
#SBATCH --account=nklab          # The account name for the job.
#SBATCH --job-name=summary       # The job name.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=6gb        # The memory the job will use per cpu core.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hs3110@columbia.edu


module load matlab
module load anaconda/3-5.3.1
conda activate fmri-sim
export PATH="/moto/home/hs3110/.conda/envs/fmri-sim/bin:$PATH"

export SOURCE='/moto/nklab/projects/ds001246_r/'
export CODE='/moto/home/hs3110/fmri-simulations/06 Paper Plots'
export MATLAB_CODE='/moto/home/hs3110/fmri-simulations/matlab'

cd "$CODE"
python housekeeping.py
