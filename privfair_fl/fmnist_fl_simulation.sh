#!/bin/bash

# Define an array of optimizers
optimizers=("qffedavg" "afl" "qffedsgd" "maml" "ditto")

# Function to run simulations
run_simulation() {
    local optimizer=$1
    local simulation_type=$2
    local q=$3
    local lr=$4
    local base_dir="./log_${simulation_type}_q${q}_lr${lr}/${optimizer}"
    local output_csv="${base_dir}/${optimizer}_${simulation_type}_final_accuracies.csv"
    local log_file="${base_dir}/${optimizer}_${simulation_type}.log"
    local script_name="${base_dir}/run_${simulation_type}_${optimizer}.sh"

    # Ensure the output directory exists
    mkdir -p "${base_dir}"

    # Create script
    echo "#!/bin/bash" > "$script_name"
    echo "python -u main.py \\" >> "$script_name"
    echo "--dataset=${simulation_type} \\" >> "$script_name"
    echo "--optimizer=${optimizer} \\" >> "$script_name"
    echo "--learning_rate=${lr} \\" >> "$script_name"
    echo "--learning_rate_lambda=${lr} \\" >> "$script_name"
    echo "--num_rounds=20 \\" >> "$script_name"
    echo "--eval_every=1 \\" >> "$script_name"
    echo "--clients_per_round=5 \\" >> "$script_name"
    echo "--batch_size=512 \\" >> "$script_name"
    echo "--q=${q} \\" >> "$script_name"
    echo "--model='cnn' \\" >> "$script_name"
    echo "--sampling=2 \\" >> "$script_name"
    echo "--num_epochs=40 \\" >> "$script_name"
    echo "--data_partition_seed=1 \\" >> "$script_name"
    echo "--log_interval=10 \\" >> "$script_name"
    echo "--static_step_size=0 \\" >> "$script_name"
    echo "--track_individual_accuracy=0 \\" >> "$script_name"
    echo "--output=\"${base_dir}/${optimizer}_${simulation_type}\" \\" >> "$script_name"    
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
q_value=1
lr_value=0.1
dataset="fmnist_non_iid_backdoor"

### MNIST SIMULATION
for optimizer in "${optimizers[@]}"; do
    run_simulation "$optimizer" $dataset $q_value $lr_value &
    ((current_jobs++))
    if [[ $current_jobs -ge $max_jobs ]]; then
        wait -n
        ((current_jobs--))
    fi
done
wait