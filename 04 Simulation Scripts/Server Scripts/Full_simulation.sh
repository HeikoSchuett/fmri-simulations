#!/bin/bash
#
#SBATCH --account=nklab          # The account name for the job.
#SBATCH --job-name=full_sim      # The job name.
#SBATCH -c 8                     # The number of cpu cores to use.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hs3110@columbia.edu


module load matlab
module load anaconda/3-5.3.1
conda activate fmri-sim
export PATH="/moto/home/hs3110/.conda/envs/fmri-sim/bin:$PATH"
mkdir /tmp/$SLURM_JOB_ID
mkdir /tmp/$SLURM_JOB_ID/data

cd /moto/home/hs3110/fmri-simulations/04\ Simulation\ Scripts/Server\ Scripts

matlab -nodisplay -nosplash - nodesktop -r "cd /tmp/$SLURM_JOB_ID; try, run ('/moto/home/hs3110/fmri-simulations/04 Simulation Scripts/Server Scripts/Noise_shuffling_moto.m'); catch me, fprintf('%s / %s\n',me.identifier,me.message), end, exit"
matlab -nodisplay -nosplash - nodesktop -r "cd /tmp/$SLURM_JOB_ID; try, run ('/moto/home/hs3110/fmri-simulations/04 Simulation Scripts/Server Scripts/GLM_on_sim_moto.m'); catch me, fprintf('%s / %s\n',me.identifier,me.message), end, exit"
python 01-Pool_simulation_results_moto.py
python 02-Create_sim_pyrsa_dataset_moto.py
python 03-Create_sim_RDMs_moto.py
python 04-Test_sim_fixed_inference_moto.py

rm -rf /tmp/$SLURM_JOB_ID
