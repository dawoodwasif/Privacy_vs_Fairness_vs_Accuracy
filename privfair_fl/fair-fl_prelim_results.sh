# # # #!/bin/bash

# # # # Global parameters
# # # DATASET="mri_iid"
# # # OPTIMIZERS=("qffedavg" "qffedsgd" "maml") # list of optimizers
# # # LEARNING_RATE=0.001
# # # LEARNING_RATE_LAMBDA=0.01
# # # NUM_ROUNDS=20
# # # EVAL_EVERY=1
# # # CLIENTS_PER_ROUND=5
# # # BATCH_SIZE=256
# # # MODEL="cnn"
# # # SAMPLING=2
# # # NUM_EPOCHS=10
# # # DATA_PARTITION_SEED=1
# # # LOG_INTERVAL=10
# # # STATIC_STEP_SIZE=0
# # # TRACK_INDIVIDUAL_ACCURACY=0
# # # OUTPUT_DIR="./log_fair_prelim_results"

# # # # Loop lists as variables for flexibility
# # # Q_VALUES=(0 1 5 10 15 20) # fixed the spacing issue

# # # # Create the output directory if not already existing
# # # mkdir -p $OUTPUT_DIR

# # # # Loop through the values of q
# # # for q_value in "${Q_VALUES[@]}"; do

# # #     # Loop through different optimizers
# # #     for optimizer in "${OPTIMIZERS[@]}"; do

# # #         # Create a new bash script for each configuration
# # #         script_name="run_${optimizer}_q${q_value}.sh"
# # #         echo "#!/bin/bash" > $script_name
# # #         echo "python -u main.py \\
# # #             --dataset=${DATASET} \\
# # #             --optimizer=${optimizer} \\
# # #             --learning_rate=${LEARNING_RATE} \\
# # #             --learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\
# # #             --num_rounds=${NUM_ROUNDS} \\
# # #             --eval_every=${EVAL_EVERY} \\
# # #             --clients_per_round=${CLIENTS_PER_ROUND} \\
# # #             --batch_size=${BATCH_SIZE} \\
# # #             --q=${q_value} \\
# # #             --model=${MODEL} \\
# # #             --sampling=${SAMPLING} \\
# # #             --num_epochs=${NUM_EPOCHS} \\
# # #             --data_partition_seed=${DATA_PARTITION_SEED} \\
# # #             --log_interval=${LOG_INTERVAL} \\
# # #             --static_step_size=${STATIC_STEP_SIZE} \\
# # #             --track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\
# # #             --output=\"${OUTPUT_DIR}/${optimizer}_mri_iid_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}.log\"" >> $script_name

# # #         # Make the script executable
# # #         chmod +x $script_name
# # #         # Run the generated script
# # #         ./$script_name

# # #     done
# # # done

# # # echo "All configurations have been processed."



# # # MRI NON - IID 
# # #!/bin/bash

# # # Global parameters
# # DATASET="mri_non_iid"
# # OPTIMIZERS=("qffedsgd") # list of optimizers
# # LEARNING_RATE=0.001
# # LEARNING_RATE_LAMBDA=0.01
# # NUM_ROUNDS=10
# # EVAL_EVERY=1
# # CLIENTS_PER_ROUND=5
# # BATCH_SIZE=256
# # MODEL="cnn"
# # SAMPLING=2
# # NUM_EPOCHS=0
# # DATA_PARTITION_SEED=1
# # LOG_INTERVAL=10
# # STATIC_STEP_SIZE=0
# # TRACK_INDIVIDUAL_ACCURACY=0
# # OUTPUT_DIR="./log_fair_prelim_results_non_iid"

# # # Loop lists as variables for flexibility
# # Q_VALUES=(0 1 5 10 15 20) # fixed the spacing issue

# # # Create the output directory if not already existing
# # mkdir -p $OUTPUT_DIR

# # # Loop through the values of q
# # for q_value in "${Q_VALUES[@]}"; do

# #     # Loop through different optimizers
# #     for optimizer in "${OPTIMIZERS[@]}"; do

# #         # Create a new bash script for each configuration
# #         script_name="run_${optimizer}_q${q_value}.sh"
# #         echo "#!/bin/bash" > $script_name
# #         echo "python -u main.py \\
# #             --dataset=${DATASET} \\
# #             --optimizer=${optimizer} \\
# #             --learning_rate=${LEARNING_RATE} \\
# #             --learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\
# #             --num_rounds=${NUM_ROUNDS} \\
# #             --eval_every=${EVAL_EVERY} \\
# #             --clients_per_round=${CLIENTS_PER_ROUND} \\
# #             --batch_size=${BATCH_SIZE} \\
# #             --q=${q_value} \\
# #             --model=${MODEL} \\
# #             --sampling=${SAMPLING} \\
# #             --num_epochs=${NUM_EPOCHS} \\
# #             --data_partition_seed=${DATA_PARTITION_SEED} \\
# #             --log_interval=${LOG_INTERVAL} \\
# #             --static_step_size=${STATIC_STEP_SIZE} \\
# #             --track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\
# #             --output=\"${OUTPUT_DIR}/${optimizer}_mri_non_iid_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}.log\"" >> $script_name

# #         # Make the script executable
# #         chmod +x $script_name
# #         # Run the generated script
# #         ./$script_name

# #     done
# # done

# # echo "All configurations have been processed."

# # MRI NON - IID 
# #!/bin/bash

# # Global parameters
# DATASET="mri_non_iid"
# OPTIMIZERS=("qffedavg" "qfedsgd") # list of optimizers
# LEARNING_RATE=0.001
# LEARNING_RATE_LAMBDA=0.01
# NUM_ROUNDS=10
# EVAL_EVERY=1
# CLIENTS_PER_ROUND=5
# BATCH_SIZE=256
# MODEL="cnn"
# SAMPLING=2
# NUM_EPOCHS=5
# DATA_PARTITION_SEED=1
# LOG_INTERVAL=10
# STATIC_STEP_SIZE=0
# TRACK_INDIVIDUAL_ACCURACY=0
# OUTPUT_DIR="log_fair_prelim_results_non_iid"

# mkdir -p ${OUTPUT_DIR}


# # Loop lists as variables for flexibility
# Q_VALUES=(0 1 5 10 15 20) # fixed the spacing issue
# RUN_NUMBER_START=1
# RUN_NUMBER_END=5

# # Loop through the values of q
# for q_value in "${Q_VALUES[@]}"; do

#     # Loop through different optimizers
#     for optimizer in "${OPTIMIZERS[@]}"; do

#         # Loop for run_number from 1 to 5
#         for run_number in $(seq $RUN_NUMBER_START $RUN_NUMBER_END); do

#             # Create a new bash script for each configuration
#             script_name="run_${optimizer}_q${q_value}_run${run_number}.sh"
#             echo "#!/bin/bash" > $script_name
#             echo "python -u main.py \\
#                 --dataset=${DATASET} \\
#                 --optimizer=${optimizer} \\
#                 --learning_rate=${LEARNING_RATE} \\
#                 --learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\
#                 --num_rounds=${NUM_ROUNDS} \\
#                 --eval_every=${EVAL_EVERY} \\
#                 --clients_per_round=${CLIENTS_PER_ROUND} \\
#                 --batch_size=${BATCH_SIZE} \\
#                 --q=${q_value} \\
#                 --model=${MODEL} \\
#                 --sampling=${SAMPLING} \\
#                 --num_epochs=${NUM_EPOCHS} \\
#                 --data_partition_seed=${DATA_PARTITION_SEED} \\
#                 --log_interval=${LOG_INTERVAL} \\
#                 --static_step_size=${STATIC_STEP_SIZE} \\
#                 --track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\
#                 --output=\"${OUTPUT_DIR}/${optimizer}_mri_non_iid_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}\" \\
#                 --run_number ${run_number}" >> $script_name

#             # Make the script executable
#             chmod +x $script_name
#             # Run the generated script
#             ./$script_name

#         done
#     done
# done

# echo "All configurations have been processed."

############################################# multi-threading ##############################################3


# #!/bin/bash

# # Global parameters
# DATASET="mri_non_iid"
# OPTIMIZERS=("qffedavg" "qffedsgd") # List of optimizers
# LEARNING_RATE=0.001
# LEARNING_RATE_LAMBDA=0.01
# NUM_ROUNDS=20
# EVAL_EVERY=1
# CLIENTS_PER_ROUND=5
# BATCH_SIZE=512
# MODEL="cnn"
# SAMPLING=2
# NUM_EPOCHS=10
# DATA_PARTITION_SEED=1
# LOG_INTERVAL=10
# STATIC_STEP_SIZE=0
# TRACK_INDIVIDUAL_ACCURACY=0
# OUTPUT_DIR="log_fair_prelim_results_non_iid"

# # Create output directory
# mkdir -p ${OUTPUT_DIR}

# # Define Q values and run number range
# Q_VALUES=(0 1 5 10 15 20) # Different q values for testing
# RUN_NUMBER_START=1
# RUN_NUMBER_END=5
# MAX_JOBS=4  # Maximum number of jobs to run in parallel

# # Loop through the values of q
# for q_value in "${Q_VALUES[@]}"; do
#     # Loop through different optimizers
#     for optimizer in "${OPTIMIZERS[@]}"; do
#         # Loop for run_number from 1 to 5
#         for run_number in $(seq $RUN_NUMBER_START $RUN_NUMBER_END); do
#             # Create a new bash script for each configuration
#             script_name="${OUTPUT_DIR}/run_${optimizer}_q${q_value}_run${run_number}.sh"
#             echo "#!/bin/bash" > $script_name
#             echo "python -u main.py \\" >> $script_name
#             echo "--dataset=${DATASET} \\" >> $script_name
#             echo "--optimizer=${optimizer} \\" >> $script_name
#             echo "--learning_rate=${LEARNING_RATE} \\" >> $script_name
#             echo "--learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> $script_name
#             echo "--num_rounds=${NUM_ROUNDS} \\" >> $script_name
#             echo "--eval_every=${EVAL_EVERY} \\" >> $script_name
#             echo "--clients_per_round=${CLIENTS_PER_ROUND} \\" >> $script_name
#             echo "--batch_size=${BATCH_SIZE} \\" >> $script_name
#             echo "--q=${q_value} \\" >> $script_name
#             echo "--model=${MODEL} \\" >> $script_name
#             echo "--sampling=${SAMPLING} \\" >> $script_name
#             echo "--num_epochs=${NUM_EPOCHS} \\" >> $script_name
#             echo "--data_partition_seed=${DATA_PARTITION_SEED} \\" >> $script_name
#             echo "--log_interval=${LOG_INTERVAL} \\" >> $script_name
#             echo "--static_step_size=${STATIC_STEP_SIZE} \\" >> $script_name
#             echo "--track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> $script_name
#             echo "--output=\"${OUTPUT_DIR}/${optimizer}_mri_non_iid_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}\" \\" >> $script_name
#             echo "--run_number=${run_number}" >> $script_name

#             # Make the script executable
#             chmod +x $script_name

#             # Run the generated script in the background and redirect output to log files
#             ./$script_name > "${OUTPUT_DIR}/${optimizer}_q${q_value}_run${run_number}.log" 2>&1 &

#             # Control number of parallel jobs
#             while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
#                 sleep 1  # Wait for a background job to finish
#             done
#         done
#     done
# done

# # Wait for all background jobs to finish
# wait
# echo "All configurations have been processed."


# #### iid #####

# # Global parameters
# DATASET="mri_iid"
# OPTIMIZERS=("maml") # List of optimizers
# LEARNING_RATE=0.001
# LEARNING_RATE_LAMBDA=0.01
# NUM_ROUNDS=20
# EVAL_EVERY=1
# CLIENTS_PER_ROUND=5
# BATCH_SIZE=512
# MODEL="cnn"
# SAMPLING=2
# NUM_EPOCHS=10
# DATA_PARTITION_SEED=1
# LOG_INTERVAL=10
# STATIC_STEP_SIZE=0
# TRACK_INDIVIDUAL_ACCURACY=0
# OUTPUT_DIR="log_fair_prelim_results_iid"

# # Create output directory
# mkdir -p ${OUTPUT_DIR}

# # Define Q values and run number range
# Q_VALUES=(0 1 5 10 15 20) # Different q values for testing
# RUN_NUMBER_START=1
# RUN_NUMBER_END=5
# MAX_JOBS=10  # Maximum number of jobs to run in parallel

# # Loop through the values of q
# for q_value in "${Q_VALUES[@]}"; do
#     # Loop through different optimizers
#     for optimizer in "${OPTIMIZERS[@]}"; do
#         # Loop for run_number from 1 to 5
#         for run_number in $(seq $RUN_NUMBER_START $RUN_NUMBER_END); do
#             # Output file path
#             output_file="${OUTPUT_DIR}/${optimizer}_mri_iid_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_final_accuracies.csv"
#             output_csv="${OUTPUT_DIR}/${optimizer}_mri_iid_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_final_accuracies.csv_final_variances.csv"

#             # Check if output file exists
#             if [ -f "${output_csv}" ]; then
#                 echo "Output file ${output_file} already exists. Skipping this configuration."
#                 continue
#             fi
            
#             echo "Executing file ${output_file} ! "

#             # Create a new bash script for each configuration
#             script_name="${OUTPUT_DIR}/run_${optimizer}_q${q_value}_run${run_number}.sh"
#             echo "#!/bin/bash" > $script_name
#             echo "python -u main.py \\" >> $script_name
#             echo "--dataset=${DATASET} \\" >> $script_name
#             echo "--optimizer=${optimizer} \\" >> $script_name
#             echo "--learning_rate=${LEARNING_RATE} \\" >> $script_name
#             echo "--learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> $script_name
#             echo "--num_rounds=${NUM_ROUNDS} \\" >> $script_name
#             echo "--eval_every=${EVAL_EVERY} \\" >> $script_name
#             echo "--clients_per_round=${CLIENTS_PER_ROUND} \\" >> $script_name
#             echo "--batch_size=${BATCH_SIZE} \\" >> $script_name
#             echo "--q=${q_value} \\" >> $script_name
#             echo "--model=${MODEL} \\" >> $script_name
#             echo "--sampling=${SAMPLING} \\" >> $script_name
#             echo "--num_epochs=${NUM_EPOCHS} \\" >> $script_name
#             echo "--data_partition_seed=${DATA_PARTITION_SEED} \\" >> $script_name
#             echo "--log_interval=${LOG_INTERVAL} \\" >> $script_name
#             echo "--static_step_size=${STATIC_STEP_SIZE} \\" >> $script_name
#             echo "--track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> $script_name
#             echo "--output=\"${output_file}\" \\" >> $script_name
#             echo "--run_number=${run_number}" >> $script_name

#             # Make the script executable
#             chmod +x $script_name

#             # Run the generated script in the background and redirect output to log files
#             ./$script_name > "${OUTPUT_DIR}/${optimizer}_q${q_value}_run${run_number}.log" 2>&1 &

#             # Control number of parallel jobs
#             while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
#                 sleep 1  # Wait for a background job to finish
#             done
#         done
#     done
# done

# # Wait for all background jobs to finish
# wait
# echo "All configurations have been processed."



#### non iid #####

# Global parameters
DATASET="mnist"  #"mri_non_iid"
OPTIMIZERS=("qffedavg" "qffedsgd" "maml") # List of optimizers
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
OUTPUT_DIR="log_fair-fl_privfair_results_mnist"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Define Q values and run number range
Q_VALUES=(0 1 5 10 15 20) # Different q values for testing
RUN_NUMBER_START=1
RUN_NUMBER_END=5
MAX_JOBS=8  # Maximum number of jobs to run in parallel

# Loop through the values of q
for q_value in "${Q_VALUES[@]}"; do
    # Loop through different optimizers
    for optimizer in "${OPTIMIZERS[@]}"; do
        # Loop for run_number from 1 to 5
        for run_number in $(seq $RUN_NUMBER_START $RUN_NUMBER_END); do
            # Output file path
            output_file="${OUTPUT_DIR}/${optimizer}_mnist_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_final_accuracies.csv"
            output_csv="${OUTPUT_DIR}/${optimizer}_mnist_run${run_number}_q${q_value}_clients${CLIENTS_PER_ROUND}_rounds${NUM_ROUNDS}_epochs${NUM_EPOCHS}_sampling${SAMPLING}_final_accuracies.csv_final_variances.csv"

            # Check if output file exists
            if [ -f "${output_csv}" ]; then
                echo "Output file ${output_file} already exists. Skipping this configuration."
                continue
            fi
            
            echo "Executing file ${output_file} ! "

            # Create a new bash script for each configuration
            script_name="${OUTPUT_DIR}/run_${optimizer}_q${q_value}_run${run_number}.sh"
            echo "#!/bin/bash" > $script_name
            echo "python -u main.py \\" >> $script_name
            echo "--dataset=${DATASET} \\" >> $script_name
            echo "--optimizer=${optimizer} \\" >> $script_name
            echo "--learning_rate=${LEARNING_RATE} \\" >> $script_name
            echo "--learning_rate_lambda=${LEARNING_RATE_LAMBDA} \\" >> $script_name
            echo "--num_rounds=${NUM_ROUNDS} \\" >> $script_name
            echo "--eval_every=${EVAL_EVERY} \\" >> $script_name
            echo "--clients_per_round=${CLIENTS_PER_ROUND} \\" >> $script_name
            echo "--batch_size=${BATCH_SIZE} \\" >> $script_name
            echo "--q=${q_value} \\" >> $script_name
            echo "--model=${MODEL} \\" >> $script_name
            echo "--sampling=${SAMPLING} \\" >> $script_name
            echo "--num_epochs=${NUM_EPOCHS} \\" >> $script_name
            echo "--data_partition_seed=${DATA_PARTITION_SEED} \\" >> $script_name
            echo "--log_interval=${LOG_INTERVAL} \\" >> $script_name
            echo "--static_step_size=${STATIC_STEP_SIZE} \\" >> $script_name
            echo "--track_individual_accuracy=${TRACK_INDIVIDUAL_ACCURACY} \\" >> $script_name
            echo "--output=\"${output_file}\" \\" >> $script_name
            echo "--run_number=${run_number}" >> $script_name

            # Make the script executable
            chmod +x $script_name

            # Run the generated script in the background and redirect output to log files
            ./$script_name > "${OUTPUT_DIR}/${optimizer}_q${q_value}_run${run_number}.log" 2>&1 &

            # Control number of parallel jobs
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 1  # Wait for a background job to finish
            done
        done
    done
done

# Wait for all background jobs to finish
wait
echo "All configurations have been processed."
