#!/bin/bash

# Define an array of optimizers
# optimizers=("ditto")

# # Function to run simulations
# run_simulation() {
#     local optimizer=$1
#     local simulation_type=$2
#     local run_number=$3
#     local base_dir="./log_${simulation_type}/${optimizer}"
#     local output_csv="${base_dir}/${optimizer}_${simulation_type}_run${run_number}_final_accuracies.csv"
#     local log_file="${base_dir}/${optimizer}_${simulation_type}_run${run_number}.log"
#     local script_name="${base_dir}/run_${simulation_type}_${optimizer}_run${run_number}.sh"

#     # Ensure the output directory exists
#     mkdir -p "${base_dir}"

#     # Create script
#     echo "#!/bin/bash" > "$script_name"
#     echo "python -u main.py \\" >> "$script_name"
#     echo "--dataset=${simulation_type} \\" >> "$script_name"
#     echo "--optimizer=${optimizer} \\" >> "$script_name"
#     echo "--learning_rate=0.001 \\" >> "$script_name"
#     echo "--learning_rate_lambda=0.01 \\" >> "$script_name"
#     echo "--num_rounds=20 \\" >> "$script_name"
#     echo "--eval_every=1 \\" >> "$script_name"
#     echo "--clients_per_round=5 \\" >> "$script_name"
#     echo "--batch_size=64 \\" >> "$script_name"
#     echo "--q=0 \\" >> "$script_name"
#     echo "--model='cnn' \\" >> "$script_name"
#     echo "--sampling=2 \\" >> "$script_name"
#     echo "--num_epochs=40 \\" >> "$script_name"
#     echo "--data_partition_seed=1 \\" >> "$script_name"
#     echo "--log_interval=10 \\" >> "$script_name"
#     echo "--static_step_size=0 \\" >> "$script_name"
#     echo "--track_individual_accuracy=0 \\" >> "$script_name"
#     echo "--output=\"${base_dir}/${optimizer}_${simulation_type}_run${run_number}\" \\" >> "$script_name"
#     echo "--run_number ${run_number}" >> "$script_name"
    
#     chmod +x "$script_name"

#     # Check if the output file exists
#     if [ ! -f "$output_csv" ]; then
#         echo "Now making: $output_csv"
#         # Run the script and log output
#         ./$script_name > "$log_file" 2>&1
#     else
#         echo "Skipping as output already exists: $output_csv"
#     fi
# }

# # Max concurrent jobs
# max_jobs=2
# current_jobs=0

# ### MRI IID SIMULATION
# for optimizer in "${optimizers[@]}"; do
#     for i in {1..10}; do
#         run_simulation "$optimizer" "mri_iid" "$i" &
#         ((current_jobs++))
#         if [[ $current_jobs -ge $max_jobs ]]; then
#             wait -n
#             ((current_jobs--))
#         fi
#     done
# done


# ### MRI NON IID SIMULATION
# for optimizer in "${optimizers[@]}"; do
#     for i in {1..10}; do
#         run_simulation "$optimizer" "mri_non_iid" "$i" &
#         ((current_jobs++))
#         if [[ $current_jobs -ge $max_jobs ]]; then
#             wait -n
#             ((current_jobs--))
#         fi
#     done
# done

# #######################################

optimizers=("qffedavg" "afl" "qffedsgd" "maml" "ditto")

# Function to run simulations
run_simulation() {
    local optimizer=$1
    local simulation_type=$2
    local run_number=$3
    local base_dir="./log_${simulation_type}_q5_lr=0.1/${optimizer}"
    local output_csv="${base_dir}/${optimizer}_${simulation_type}_run${run_number}_final_accuracies.csv"
    local log_file="${base_dir}/${optimizer}_${simulation_type}_run${run_number}.log"
    local script_name="${base_dir}/run_${simulation_type}_${optimizer}_run${run_number}.sh"

    # Ensure the output directory exists
    mkdir -p "${base_dir}"

    # Create script
    echo "#!/bin/bash" > "$script_name"
    echo "python -u main.py \\" >> "$script_name"
    echo "--dataset=${simulation_type} \\" >> "$script_name"
    echo "--optimizer=${optimizer} \\" >> "$script_name"
    echo "--learning_rate=0.1 \\" >> "$script_name"
    echo "--learning_rate_lambda=0.1 \\" >> "$script_name"
    echo "--num_rounds=20 \\" >> "$script_name"
    echo "--eval_every=1 \\" >> "$script_name"
    echo "--clients_per_round=5 \\" >> "$script_name"
    echo "--batch_size=64 \\" >> "$script_name"
    echo "--q=5 \\" >> "$script_name"
    echo "--model='cnn' \\" >> "$script_name"
    echo "--sampling=2 \\" >> "$script_name"
    echo "--num_epochs=40 \\" >> "$script_name"
    echo "--data_partition_seed=1 \\" >> "$script_name"
    echo "--log_interval=10 \\" >> "$script_name"
    echo "--static_step_size=0 \\" >> "$script_name"
    echo "--track_individual_accuracy=0 \\" >> "$script_name"
    echo "--output=\"${base_dir}/${optimizer}_${simulation_type}_run${run_number}\" \\" >> "$script_name"
    echo "--run_number ${run_number}" >> "$script_name"
    
    chmod +x "$script_name"

    # Check if the output file exists
    if [ ! -f "$output_csv" ]; then
        echo "Now making: $output_csv"
        # Run the script and log output
        ./$script_name > "$log_file" 2>&1
    else
        echo "Skipping as output already exists: $output_csv"
    fi
}

# Max concurrent jobs
max_jobs=8
current_jobs=0

### MRI IID SIMULATION
for optimizer in "${optimizers[@]}"; do
    for i in {1..10}; do
        run_simulation "$optimizer" "mri_iid" "$i" &
        ((current_jobs++))
        if [[ $current_jobs -ge $max_jobs ]]; then
            wait -n
            ((current_jobs--))
        fi
    done
done


### MRI NON IID SIMULATION
for optimizer in "${optimizers[@]}"; do
    for i in {1..10}; do
        run_simulation "$optimizer" "mri_non_iid" "$i" &
        ((current_jobs++))
        if [[ $current_jobs -ge $max_jobs ]]; then
            wait -n
            ((current_jobs--))
        fi
    done
done

