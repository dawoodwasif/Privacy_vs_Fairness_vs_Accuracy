#!/bin/bash

# Define an array of optimizers (only qffedavg as per your requirement)
optimizers=("qffedavg" "qffedsgd" "maml")

# Define q values
q_values1=(0 1 5 10)    # For the first task
q_values2=(1)         # For the second task

# Define epsilon values for DP (LDP and GDP)
epsilon_values1=(8) # For task 1
# epsilon_values2=(0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10) # For task 2
epsilon_values2=(2 4 8 16) # For task 2


# Define modulus values for HE
modulus_values1=(4096)  # For task 1
modulus_values2=(4096 8192 16384 32768) # For task 2

# Function to run simulations
run_simulation() {
    local optimizer=$1
    local privacy_tech=$2
    local param_value=$3
    local simulation_type=$4
    local q=$5
    local lr=$6
    local base_dir="./fmnist_privfair_experiments_OFFICIAL_LAST/${optimizer}/${privacy_tech}_${param_value}_q${q}_lr${lr}"
    local output_csv="${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}_final_accuracies.csv"
    local log_file="${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}.log"
    local script_name="${base_dir}/run_${privacy_tech}_${optimizer}.sh"

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
    echo "--output=\"${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}\" \\" >> "$script_name"

    if [[ $privacy_tech == "LDP" || $privacy_tech == "GDP" ]]; then
        echo "--dp_flag=True \\" >> "$script_name"
        echo "--dp_delta=1e-5 \\" >> "$script_name"
        echo "--dp_sensitivity=0.01 \\" >> "$script_name"
        echo "--dp_epsilon=${param_value} \\" >> "$script_name"
        echo "--dp_scope=${privacy_tech} \\" >> "$script_name"
        echo "--he_flag=False \\" >> "$script_name"
        echo "--smc_flag=False \\" >> "$script_name"
    elif [[ $privacy_tech == "HE" ]]; then
        echo "--he_flag=True \\" >> "$script_name"
        echo "--he_poly_modulus_degree=${param_value} \\" >> "$script_name"
        echo "--dp_flag=False \\" >> "$script_name"
        echo "--smc_flag=False \\" >> "$script_name"
    fi

    chmod +x "$script_name"

    # Check if the output file exists
    if [ ! -f "$output_csv" ]; then
        echo "Now running: $output_csv"
        # Run the script and log output
        ./$script_name > "$log_file" 2>&1
    else
        echo "Skipping as output already exists: $output_csv"
    fi
}


# Max concurrent jobs
max_jobs=18
current_jobs=0
lr_value=0.1
dataset="fmnist"

# Task 1: q-FedAvg with q=0, 1, 5, 10 for GDP and LDP  and HE (modulus 4k, 8k)
for q in "${q_values1[@]}"; do
    for epsilon in "${epsilon_values1[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            # Run LDP simulation
            run_simulation "$optimizer" "LDP" "$epsilon" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi

            # Run GDP simulation
            run_simulation "$optimizer" "GDP" "$epsilon" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi
        done
    done

    # Run HE simulation
    for modulus in "${modulus_values1[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            run_simulation "$optimizer" "HE" "$modulus" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi
        done
    done
done

# Wait for all Task 1 jobs to finish
wait

# Task 2: q-FedAvg with q=0 and 1 for GDP and LDP (epsilon 0.5 to 10) and HE (modulus 4k, 8k, 16k, 32k)
for q in "${q_values2[@]}"; do
    for epsilon in "${epsilon_values2[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            # Run LDP simulation
            run_simulation "$optimizer" "LDP" "$epsilon" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi

            # Run GDP simulation
            run_simulation "$optimizer" "GDP" "$epsilon" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi
        done
    done

    # Run HE simulation
    for modulus in "${modulus_values2[@]}"; do
        for optimizer in "${optimizers[@]}"; do
            run_simulation "$optimizer" "HE" "$modulus" $dataset $q $lr_value &
            ((current_jobs++))
            if [[ $current_jobs -ge $max_jobs ]]; then
                wait -n
                ((current_jobs--))
            fi
        done
    done
done

# Wait for all Task 2 jobs to finish
wait

