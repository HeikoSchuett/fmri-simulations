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

conda activate fmri

export SOURCE='/Users/heiko/ds001246_r/'
export INTERMEDIATE='/Users/heiko/fmri-simulations/tmp'
export CODE='/Users/heiko/fmri-simulations/04 Simulation Scripts/Server Scripts'
export MATLAB_CODE='/Users/heiko/fmri-simulations/matlab'

cd "$CODE"
for i in {1..5}
do
  /Applications/MATLAB_R2022b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "cd '$CODE'; Noise_shuffling_moto($i); exit"
  /Applications/MATLAB_R2022b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "cd '$CODE'; GLM_on_sim_moto($i); exit"
  rm -rf $INTERMEDIATE/Noise_perm
  python 01-Pool_simulation_results_moto.py -s $i
  python 02-Create_sim_pyrsa_dataset_moto.py -s $i
  python 03-Create_sim_RDMs_moto.py -s $i
  rm -rf $INTERMEDIATE/Data_perm
done
python 04-Test_sim_fixed_inference_moto.py
rm -rf $INTERMEDIATE
