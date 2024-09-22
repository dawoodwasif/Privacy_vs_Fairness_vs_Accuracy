#!/bin/bash

# Global parameters
DATASET="mri_iid"
OPTIMIZER="qffedavg"
LEARNING_RATE=0.001
LEARNING_RATE_LAMBDA=0.01
NUM_ROUNDS=20
EVAL_EVERY=1
CLIENTS_PER_ROUND=5
BATCH_SIZE=64
MODEL="cnn"
SAMPLING=2
NUM_EPOCHS=10
DATA_PARTITION_SEED=1
LOG_INTERVAL=10
STATIC_STEP_SIZE=0
TRACK_INDIVIDUAL_ACCURACY=0
OUTPUT_DIR="./log_prelim_results"

# Loop lists as variables for flexibility
Q_VALUES=(0 1 5 10 15 20)
DP_EPSILON_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
SETUPS=("dp" "dp_he" "dp_smc")

# Create a directory for scripts if not already existing
SCRIPTS_DIR="./generated_scripts"
mkdir -p $SCRIPTS_DIR

# Loop through the values of q and dp_epsilon
for q_value in "${Q_VALUES[@]}"; do
  for dp_epsilon_value in "${DP_EPSILON_VALUES[@]}"; do
    
    # Loop through different privacy setups: DP, DP+HE, DP+SMC
    for setup in "${SETUPS[@]}"; do

      # Set flags based on the privacy setup
      if [ "$setup" == "dp" ]; then
        DP_FLAG=True
        HE_FLAG=False
        SMC_FLAG=False
      elif [ "$setup" == "dp_he" ]; then
        DP_FLAG=True
        HE_FLAG=True
        SMC_FLAG=False
      elif [ "$setup" == "dp_smc" ]; then
        DP_FLAG=True
        HE_FLAG=False
        SMC_FLAG=True
      fi

      # Create output filename based on all parameter values, including the flags
      OUTPUT_FILE="${OUTPUT_DIR}/qffedavg_mri_iid_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_dp${dp_epsilon_value}_${setup}.log"

      # Create a script file name
      script_name="$run_q${q_value}_dp${dp_epsilon_value}_${setup}.sh"

      # Generate the Python command into the script file
      echo "#!/bin/bash" > $script_name
      echo "python -u main.py \\" >> $script_name
      echo "  --dataset=${DATASET} \\" >> $script_name
      echo "  --optimizer=${OPTIMIZER} \\" >> $script_name
      echo "  --learning_rate=${LEARNING_RATE} \\" >> $script_name
      echo "  --learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> $script_name
      echo "  --num_rounds=${NUM_ROUNDS} \\" >> $script_name
      echo "  --eval_every=${EVAL_EVERY} \\" >> $script_name
      echo "  --clients_per_round=${CLIENTS_PER_ROUND} \\" >> $script_name
      echo "  --batch_size=${BATCH_SIZE} \\" >> $script_name
      echo "  --q=${q_value} \\" >> $script_name
      echo "  --model=${MODEL} \\" >> $script_name
      echo "  --sampling=${SAMPLING} \\" >> $script_name
      echo "  --num_epochs=${NUM_EPOCHS} \\" >> $script_name
      echo "  --data_partition_seed=${DATA_PARTITION_SEED} \\" >> $script_name
      echo "  --log_interval=${LOG_INTERVAL} \\" >> $script_name
      echo "  --static_step_size=${STATIC_STEP_SIZE} \\" >> $script_name
      echo "  --track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> $script_name
      echo "  --output=${OUTPUT_FILE} \\" >> $script_name
      echo "  --dp_flag=${DP_FLAG} \\" >> $script_name
      echo "  --dp_epsilon=${dp_epsilon_value} \\" >> $script_name
      echo "  --he_flag=${HE_FLAG} \\" >> $script_name
      echo "  --smc_flag=${SMC_FLAG}" >> $script_name

      # Make the script executable
      chmod +x $script_name

      # Run the generated script
      ./$script_name
      
    done
  done
done
