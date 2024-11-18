# #!/bin/bash

# # Global parameters
# DATASET="mri_non_iid"
# OPTIMIZER="qffedavg"
# LEARNING_RATE=0.001
# LEARNING_RATE_LAMBDA=0.01
# NUM_ROUNDS=20
# EVAL_EVERY=1
# CLIENTS_PER_ROUND=5
# BATCH_SIZE=64
# MODEL="cnn"
# SAMPLING=2
# NUM_EPOCHS=10
# DATA_PARTITION_SEED=1
# LOG_INTERVAL=10
# STATIC_STEP_SIZE=0
# TRACK_INDIVIDUAL_ACCURACY=0
# OUTPUT_DIR="./log_pp_prelim_results_non_iid"

# # Loop lists as variables for flexibility
# DP_EPSILON_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
# SETUPS=("dp" "dp_he") #("dp" "dp_he" "dp_smc")

# # Create the output directory if not already existing
# mkdir -p $OUTPUT_DIR

# # Loop through the values of q and dp_epsilon
# for dp_epsilon_value in "${DP_EPSILON_VALUES[@]}"; do
  
#   # Loop through different privacy setups: DP, DP+HE, DP+SMC
#   for setup in "${SETUPS[@]}"; do

#     # Set flags based on the privacy setup
#     if [ "$setup" == "dp" ]; then
#       DP_FLAG=True
#       HE_FLAG=False
#       SMC_FLAG=False
#     elif [ "$setup" == "dp_he" ]; then
#       DP_FLAG=True
#       HE_FLAG=True
#       SMC_FLAG=False
#     elif [ "$setup" == "dp_smc" ]; then
#       DP_FLAG=True
#       HE_FLAG=False
#       SMC_FLAG=True
#     fi

#     # Create a new bash script for each configuration
#     script_name="run_q${q_value}_dp${dp_epsilon_value}_${setup}.sh"
#     echo "#!/bin/bash" > $script_name
#     echo "python -u main.py \\
#       --dataset=${DATASET} \\
#       --optimizer=${OPTIMIZER} \\
#       --learning_rate=${LEARNING_RATE} \\
#       --learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\
#       --num_rounds=${NUM_ROUNDS} \\
#       --eval_every=${EVAL_EVERY} \\
#       --clients_per_round=${CLIENTS_PER_ROUND} \\
#       --batch_size=${BATCH_SIZE} \\
#       --q=0 \\
#       --model=${MODEL} \\
#       --sampling=${SAMPLING} \\
#       --num_epochs=${NUM_EPOCHS} \\
#       --data_partition_seed=${DATA_PARTITION_SEED} \\
#       --log_interval=${LOG_INTERVAL} \\
#       --static_step_size=${STATIC_STEP_SIZE} \\
#       --track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\
#       --output=\"${OUTPUT_DIR}/qffedavg_mri_non_iid_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_dp${dp_epsilon_value}_${setup}.log\" \\
#       --dp_flag=${DP_FLAG} \\
#       --dp_epsilon=${dp_epsilon_value} \\
#       --he_flag=${HE_FLAG} \\
#       --smc_flag=${SMC_FLAG}" >> $script_name

#     # Make the script executable
#     chmod +x $script_name
#     # Run the generated script
#     ./$script_name

#   done
# done

# echo "All configurations have been processed."

#!/bin/bash

#################### multi threading ###########################


#!/bin/bash

# Function to run experiments
# run_experiments () {
#   local DATASET=$1
#   local OUTPUT_DIR=$2
#   local MAX_JOBS=5  # Set maximum number of parallel jobs

#   # Create the output directory if not already existing
#   mkdir -p $OUTPUT_DIR

#   # Loop through the values of dp_epsilon
#   for dp_epsilon_value in "${DP_EPSILON_VALUES[@]}"; do
    
#     # Loop through different privacy setups: DP, DP+HE, DP+SMC
#     for setup in "${SETUPS[@]}"; do

#       # Set flags based on the privacy setup
#       local DP_FLAG="False"
#       local HE_FLAG="False"
#       local SMC_FLAG="False"

#       if [ "$setup" == "dp" ]; then
#         DP_FLAG="True"
#       elif [ "$setup" == "dp_he" ]; then
#         DP_FLAG="True"
#         HE_FLAG="True"
#       elif [ "$setup" == "dp_smc" ]; then
#         DP_FLAG="True"
#         SMC_FLAG="True"
#       fi

#       # Loop for multiple run numbers
#       for run_number in $(seq 1 5); do
#         # Create a new bash script for each configuration
#         local script_name="${OUTPUT_DIR}/run_dp${dp_epsilon_value}_${setup}_run${run_number}.sh"
#         local full_script_path="${PWD}/${script_name}"
        
#         echo "#!/bin/bash" > "$full_script_path"
#         echo "python -u main.py \\" >> "$full_script_path"
#         echo "--dataset=${DATASET} \\" >> "$full_script_path"
#         echo "--optimizer=${OPTIMIZER} \\" >> "$full_script_path"
#         echo "--learning_rate=${LEARNING_RATE} \\" >> "$full_script_path"
#         echo "--learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> "$full_script_path"
#         echo "--num_rounds=${NUM_ROUNDS} \\" >> "$full_script_path"
#         echo "--eval_every=${EVAL_EVERY} \\" >> "$full_script_path"
#         echo "--clients_per_round=${CLIENTS_PER_ROUND} \\" >> "$full_script_path"
#         echo "--batch_size=${BATCH_SIZE} \\" >> "$full_script_path"
#         echo "--q=0 \\" >> "$full_script_path"
#         echo "--model=${MODEL} \\" >> "$full_script_path"
#         echo "--sampling=${SAMPLING} \\" >> "$full_script_path"
#         echo "--num_epochs=${NUM_EPOCHS} \\" >> "$full_script_path"
#         echo "--data_partition_seed=${DATA_PARTITION_SEED} \\" >> "$full_script_path"
#         echo "--log_interval=${LOG_INTERVAL} \\" >> "$full_script_path"
#         echo "--static_step_size=${STATIC_STEP_SIZE} \\" >> "$full_script_path"
#         echo "--track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> "$full_script_path"
#         echo "--output=\"${OUTPUT_DIR}/${OPTIMIZER}_${DATASET}_q0_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_dp${dp_epsilon_value}_${setup}_run${run_number}.log\" \\" >> "$full_script_path"
#         echo "--dp_flag=${DP_FLAG} \\" >> "$full_script_path"
#         echo "--dp_epsilon=${dp_epsilon_value} \\" >> "$full_script_path"
#         echo "--he_flag=${HE_FLAG} \\" >> "$full_script_path"
#         echo "--smc_flag=${SMC_FLAG} \\" >> "$full_script_path"
#         echo "--run_number=${run_number}" >> "$full_script_path"

#         # Make the script executable
#         chmod +x "$full_script_path"

#         # Run the generated script in the background and redirect output
#         "$full_script_path" > "${OUTPUT_DIR}/run_${dp_epsilon_value}_${setup}_run${run_number}.log" 2>&1 &

#         # Control the number of parallel jobs (ensure only 4 run simultaneously)
#         while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
#           sleep 1  # Wait for a job to finish before starting another
#         done
#       done
#     done
#   done
# }

# # Global parameters
# OPTIMIZER="qffedavg"
# LEARNING_RATE=0.001
# LEARNING_RATE_LAMBDA=0.01
# NUM_ROUNDS=20
# EVAL_EVERY=1
# CLIENTS_PER_ROUND=5
# BATCH_SIZE=256
# MODEL="cnn"
# SAMPLING=2
# NUM_EPOCHS=10
# DATA_PARTITION_SEED=1
# LOG_INTERVAL=10
# STATIC_STEP_SIZE=0
# TRACK_INDIVIDUAL_ACCURACY=0

# # Epsilon values and setups
# DP_EPSILON_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
# SETUPS=("dp" "dp_he")  # Ensure to include all setups

# # Run experiments for non-IID data
# run_experiments "mri_non_iid" "./log_pp_prelim_results_non_iid"

# # Run experiments for IID data
# run_experiments "mri_iid" "./log_pp_prelim_results_iid"

# # Wait for all background jobs to finish
# wait
# echo "All configurations have been processed."


#!/bin/bash

# Function to run experiments
run_experiments () {
  local DATASET=$1
  local OUTPUT_DIR=$2
  local MAX_JOBS=5  # Set maximum number of parallel jobs

  # Create the output directory if not already existing
  mkdir -p $OUTPUT_DIR

  # Loop through the values of dp_epsilon
  for dp_epsilon_value in "${DP_EPSILON_VALUES[@]}"; do
    
    # Loop through different privacy setups: DP, DP+HE
    for setup in "${SETUPS[@]}"; do

      # Set flags based on the privacy setup
      local DP_FLAG="False"
      local HE_FLAG="False"
      local SMC_FLAG="False"

      if [ "$setup" == "dp" ]; then
        DP_FLAG="True"
      elif [ "$setup" == "dp_he" ]; then
        DP_FLAG="True"
        HE_FLAG="True"
      elif [ "$setup" == "dp_smc" ]; then
        DP_FLAG="True"
        SMC_FLAG="True"
      fi

      # Loop for multiple run numbers
      for run_number in $(seq 1 5); do
        local output_log="${OUTPUT_DIR}/${OPTIMIZER}_${DATASET}_q0_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_dp${dp_epsilon_value}_${setup}_run${run_number}.log"
        local output_file="${OUTPUT_DIR}/${OPTIMIZER}_${DATASET}_q0_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_dp${dp_epsilon_value}_${setup}_run${run_number}.log_final_euclidean_distances.csv"
        
        # Check if the output file already exists
        if [ -f "$output_file" ]; then
          echo "Skipping existing experiment: $output_log"
          continue
        fi

        # Create a new bash script for each configuration
        local script_name="${OUTPUT_DIR}/run_dp${dp_epsilon_value}_${setup}_run${run_number}.sh"
        local full_script_path="${PWD}/${script_name}"
        

        echo "Executing script: $full_script_path"

        echo "#!/bin/bash" > "$full_script_path"
        echo "python -u main.py \\" >> "$full_script_path"
        echo "--dataset=${DATASET} \\" >> "$full_script_path"
        echo "--optimizer=${OPTIMIZER} \\" >> "$full_script_path"
        echo "--learning_rate=${LEARNING_RATE} \\" >> "$full_script_path"
        echo "--learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> "$full_script_path"
        echo "--num_rounds=${NUM_ROUNDS} \\" >> "$full_script_path"
        echo "--eval_every=${EVAL_EVERY} \\" >> "$full_script_path"
        echo "--clients_per_round=${CLIENTS_PER_ROUND} \\" >> "$full_script_path"
        echo "--batch_size=${BATCH_SIZE} \\" >> "$full_script_path"
        echo "--q=0 \\" >> "$full_script_path"
        echo "--model=${MODEL} \\" >> "$full_script_path"
        echo "--sampling=${SAMPLING} \\" >> "$full_script_path"
        echo "--num_epochs=${NUM_EPOCHS} \\" >> "$full_script_path"
        echo "--data_partition_seed=${DATA_PARTITION_SEED} \\" >> "$full_script_path"
        echo "--log_interval=${LOG_INTERVAL} \\" >> "$full_script_path"
        echo "--static_step_size=${STATIC_STEP_SIZE} \\" >> "$full_script_path"
        echo "--track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> "$full_script_path"
        echo "--output=\"$output_log\" \\" >> "$full_script_path"
        echo "--dp_flag=${DP_FLAG} \\" >> "$full_script_path"
        echo "--dp_epsilon=${dp_epsilon_value} \\" >> "$full_script_path"
        echo "--he_flag=${HE_FLAG} \\" >> "$full_script_path"
        echo "--smc_flag=${SMC_FLAG} \\" >> "$full_script_path"
        echo "--run_number=${run_number}" >> "$full_script_path"

        # Make the script executable
        chmod +x "$full_script_path"

        # Run the generated script in the background and redirect output
        "$full_script_path" > "$output_log" 2>&1 &

        # Control the number of parallel jobs (ensure only 5 run simultaneously)
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
          sleep 1  # Wait for a job to finish before starting another
        done
      done
    done
  done
}

# Global parameters
OPTIMIZER="qffedavg"
LEARNING_RATE=0.001
LEARNING_RATE_LAMBDA=0.01
NUM_ROUNDS=20
EVAL_EVERY=1
CLIENTS_PER_ROUND=5
BATCH_SIZE=256
MODEL="cnn"
SAMPLING=2
NUM_EPOCHS=10
DATA_PARTITION_SEED=1
LOG_INTERVAL=10
STATIC_STEP_SIZE=0
TRACK_INDIVIDUAL_ACCURACY=0

# Epsilon values and setups
DP_EPSILON_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
SETUPS=("dp" "dp_he" "dp_he")  # Include all setups

# Run experiments for non-IID data
# run_experiments "mri_non_iid" "./log_pp_prelim_results_non_iid"

# # Run experiments for IID data
# run_experiments "mri_iid" "./log_pp_prelim_results_iid"

run_experiments "mnist" "log_pp_privfair_results_mnist"

# Wait for all background jobs to finish
wait
echo "All configurations have been processed."
