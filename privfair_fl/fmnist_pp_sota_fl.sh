# #!/bin/bash

# # Define an array of optimizers
# optimizers=("qffedavg")

# # Set epsilon value for both LDP and GDP to 8
# epsilon=8

# # Define modulus values for HE (default 4096)
# modulus_values=(4096)

# # Define anonymization techniques (t, k, l)
# anon_techniques=("t" "k" "l")

# # Function to run simulations
# run_simulation() {
#     local optimizer=$1
#     local privacy_tech=$2
#     local param_value=$3
#     local simulation_type=$4
#     local q=$5
#     local lr=$6
#     local anon=$7  # Anonymization technique
#     local base_dir="./mnist_pp_sota_fl_experiments/${optimizer}/${privacy_tech}_${param_value}_q${q}_lr${lr}_${anon}"
#     local output_csv="${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}_final_accuracies.csv"
#     local log_file="${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}.log"
#     local script_name="${base_dir}/run_${privacy_tech}_${optimizer}_${anon}.sh"

#     # Ensure the output directory exists
#     mkdir -p "${base_dir}"

#     # Create script
#     echo "#!/bin/bash" > "$script_name"
#     echo "python -u main.py \\" >> "$script_name"
#     echo "--dataset=${simulation_type} \\" >> "$script_name"
#     echo "--optimizer=${optimizer} \\" >> "$script_name"
#     echo "--learning_rate=${lr} \\" >> "$script_name"
#     echo "--learning_rate_lambda=${lr} \\" >> "$script_name"
#     echo "--num_rounds=20 \\" >> "$script_name"
#     echo "--eval_every=1 \\" >> "$script_name"
#     echo "--clients_per_round=5 \\" >> "$script_name"
#     echo "--batch_size=512 \\" >> "$script_name"
#     echo "--q=${q} \\" >> "$script_name"
#     echo "--model='cnn' \\" >> "$script_name"
#     echo "--sampling=2 \\" >> "$script_name"
#     echo "--num_epochs=40 \\" >> "$script_name"
#     echo "--data_partition_seed=1 \\" >> "$script_name"
#     echo "--log_interval=10 \\" >> "$script_name"
#     echo "--static_step_size=0 \\" >> "$script_name"
#     echo "--track_individual_accuracy=0 \\" >> "$script_name"
#     echo "--output=\"${base_dir}/${optimizer}_${privacy_tech}_${simulation_type}\" \\" >> "$script_name"

# #     if [[ $privacy_tech == "LDP" || $privacy_tech == "GDP" ]]; then
# #         echo "--dp_flag=True \\" >> "$script_name"
# #         echo "--dp_delta=1e-5 \\" >> "$script_name"
# #         echo "--dp_sensitivity=0.01 \\" >> "$script_name"
# #         echo "--dp_epsilon=${epsilon} \\" >> "$script_name"
# #         echo "--dp_scope=${privacy_tech} \\" >> "$script_name"
# #         echo "--he_flag=False \\" >> "$script_name"
# #         echo "--smc_flag=False \\" >> "$script_name"
# #     elif [[ $privacy_tech == "HE" ]]; then
# #         echo "--he_flag=True \\" >> "$script_name"
# #         echo "--he_poly_modulus_degree=${param_value} \\" >> "$script_name"
# #         echo "--dp_flag=False \\" >> "$script_name"
# #         echo "--smc_flag=False \\" >> "$script_name"
# #     fi

# #     # Set data anonymization technique if provided
# #     if [[ $anon == "t" || $anon == "k" || $anon == "l" ]]; then
# #         echo "--data_anon=${anon} \\" >> "$script_name"
# #     fi

# #     chmod +x "$script_name"

# #     # Check if the output file exists
# #     if [ ! -f "$output_csv" ]; then
# #         echo "Now running: $output_csv"
# #         # Run the script and log output
# #         ./$script_name > "$log_file" 2>&1
# #     else
# #         echo "Skipping as output already exists: $output_csv"
# #     fi
# # }

# # # Max concurrent jobs
# # max_jobs=6
# # current_jobs=0
# # q_value=0
# # lr_value=0.1

# # # Run simulations for LDP and GDP with epsilon = 8
# # for optimizer in "${optimizers[@]}"; do
# #     # Run LDP simulation
# #     run_simulation "$optimizer" "LDP" "$epsilon" "mnist" $q_value $lr_value "None" &
# #     ((current_jobs++))
# #     if [[ $current_jobs -ge $max_jobs ]]; then
# #         wait -n
# #         ((current_jobs--))
# #     fi

# #     # Run GDP simulation
# #     run_simulation "$optimizer" "GDP" "$epsilon" "mnist" $q_value $lr_value "None" &
# #     ((current_jobs++))
# #     if [[ $current_jobs -ge $max_jobs ]]; then
# #         wait -n
# #         ((current_jobs--))
# #     fi
# # done

# # # Wait for all LDP and GDP jobs to finish
# # wait

# # # Run simulations for HE with modulus value 4096
# # for modulus in "${modulus_values[@]}"; do
# #     for optimizer in "${optimizers[@]}"; do
# #         run_simulation "$optimizer" "HE" "$modulus" "mnist" $q_value $lr_value "None" &
# #         ((current_jobs++))
# #         if [[ $current_jobs -ge $max_jobs ]]; then
# #             wait -n
# #             ((current_jobs--))
# #         fi
# #     done
# # done

# # # Wait for all HE jobs to finish
# # wait

# # # Run simulations with data anonymization techniques (t, k, l)
# # for anon in "${anon_techniques[@]}"; do
# #     for optimizer in "${optimizers[@]}"; do
# #         run_simulation "$optimizer" "Anonymization" "$anon" "mnist" $q_value $lr_value "$anon" &
# #         ((current_jobs++))
# #         if [[ $current_jobs -ge $max_jobs ]]; then
# #             wait -n
# #             ((current_jobs--))
# #         fi
# #     done
# # done

# # # Wait for all anonymization jobs to finish
# # wait
# lr_value=0.1

# # Configure privacy techniques based on combinations
# case $privacy_tech in
# "LDP")
#     echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     ;;
# "GDP")
#     echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     ;;
# "HE")
#     echo "--he_flag=True --he_poly_modulus_degree=${param_value} \\" >> "$script_name"
#     ;;
# "SMC")
#     echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
#     ;;
# "LDP+HE")
#     echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     echo "--he_flag=True --he_poly_modulus_degree=${param_value} \\" >> "$script_name"
#     ;;
# "GDP+HE")
#     echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     echo "--he_flag=True --he_poly_modulus_degree=${param_value} \\" >> "$script_name"
#     ;;
# "LDP+SMC")
#     echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
#     ;;
# "GDP+SMC")
#     echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
#     echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
#     ;;
# "HE+SMC")
#     echo "--he_flag=True --he_poly_modulus_degree=${param_value} \\" >> "$script_name"
#     echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
#     ;;
# esac

# # Set data anonymization technique if provided
# if [[ $anon == "t" || $anon == "k" || $anon == "l" ]]; then
# echo "--data_anon=${anon} \\" >> "$script_name"
# fi

# chmod +x "$script_name"

# # Run the script if output file doesn't exist
# if [ ! -f "$output_csv" ]; then
# echo "Running simulation: $output_csv"
# ./$script_name > "$log_file" 2>&1 &
# ((current_jobs++))
# if [[ $current_jobs -ge $max_jobs ]]; then
#     wait -n
#     ((current_jobs--))
# fi
# else
# echo "Skipping existing output: $output_csv"
# fi
# }

# # Run simulations for each privacy technique and combination
# for optimizer in "${optimizers[@]}"; do
# run_simulation "$optimizer" "LDP" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "GDP" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "HE" "4096" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "SMC" "0" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "LDP+HE" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "GDP+HE" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "LDP+SMC" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "GDP+SMC" "$epsilon" "mnist" $q_value $lr_value "None"
# run_simulation "$optimizer" "HE+SMC" "4096" "mnist" $q_value $lr_value "None"
# done

# # Wait for all background jobs to finish
# wait
# echo "All simulations complete."


#!/bin/bash

# Define constants
optimizers=("qffedavg")
epsilon=8
modulus_value=4096
learning_rate=0.1
q_value=0
max_concurrent_jobs=6
current_jobs=0

# Function to run simulations
run_simulation() {
    local optimizer="$1"
    local privacy_technique="$2"
    local dataset="$3"
    local reweighting_factor="$4"
    local lr="$5"

    local base_dir="./fmnist_non_iid_pp_sota_fl_experiments_epsilon=8/${optimizer}/${privacy_technique}_q${reweighting_factor}_lr${lr}"
    local output_csv="${base_dir}/${optimizer}_${privacy_technique}_${dataset}_final_accuracies.csv"
    local log_file="${base_dir}/${optimizer}_${privacy_technique}_${dataset}.log"
    local script_name="${base_dir}/run_${privacy_technique}_${optimizer}.sh"

    # Ensure the output directory exists
    mkdir -p "${base_dir}"

    # Create the run script
    echo "#!/bin/bash" > "$script_name"
    echo "python -u main.py \\" >> "$script_name"
    echo "--dataset=${dataset} \\" >> "$script_name"
    echo "--optimizer=${optimizer} \\" >> "$script_name"
    echo "--learning_rate=${lr} \\" >> "$script_name"
    echo "--learning_rate_lambda=${lr} \\" >> "$script_name"
    echo "--num_rounds=20 \\" >> "$script_name"
    echo "--eval_every=1 \\" >> "$script_name"
    echo "--clients_per_round=5 \\" >> "$script_name"
    echo "--batch_size=512 \\" >> "$script_name"
    echo "--q=${reweighting_factor} \\" >> "$script_name"
    echo "--model='cnn' \\" >> "$script_name"
    echo "--sampling=2 \\" >> "$script_name"
    echo "--num_epochs=40 \\" >> "$script_name"
    echo "--data_partition_seed=1 \\" >> "$script_name"
    echo "--log_interval=10 \\" >> "$script_name"
    echo "--static_step_size=0 \\" >> "$script_name"
    echo "--track_individual_accuracy=0 \\" >> "$script_name"
    echo "--output=\"${base_dir}/${optimizer}_${privacy_technique}_${dataset}\" \\" >> "$script_name"

    # Configure privacy techniques based on combinations
    case $privacy_technique in
        "LDP")
            echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            ;;
        "GDP")
            echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            ;;
        "k-anonymity")
            echo "--data_anon=k \\" >> "$script_name"
            ;;
        "l-diversity")
            echo "--data_anon=l \\" >> "$script_name"
            ;;
        "t-closeness")
            echo "--data_anon=t \\" >> "$script_name"
            ;;
        "HE")
            echo "--he_flag=True --he_poly_modulus_degree=${modulus_value} \\" >> "$script_name"
            ;;
        "SMC")
            echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
            ;;
        "LDP+HE")
            echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            echo "--he_flag=True --he_poly_modulus_degree=${modulus_value} \\" >> "$script_name"
            ;;
        "GDP+HE")
            echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            echo "--he_flag=True --he_poly_modulus_degree=${modulus_value} \\" >> "$script_name"
            ;;
        "LDP+SMC")
            echo "--dp_flag=True --dp_scope=LDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
            ;;
        "GDP+SMC")
            echo "--dp_flag=True --dp_scope=GDP --dp_epsilon=${epsilon} --dp_delta=1e-5 --dp_sensitivity=0.01 \\" >> "$script_name"
            echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
            ;;
        "HE+SMC")
            echo "--he_flag=True --he_poly_modulus_degree=${modulus_value} \\" >> "$script_name"
            echo "--smc_flag=True --smc_threshold=7 --smc_num_shares=10 \\" >> "$script_name"
            ;;
    esac

    chmod +x "$script_name"

    # Run the script if output file doesn't exist
    if [ ! -f "$output_csv" ]; then
        echo "Running simulation: $output_csv"
        ./$script_name > "$log_file" 2>&1 &
        ((current_jobs++))
        if [[ $current_jobs -ge $max_concurrent_jobs ]]; then
            wait -n
            ((current_jobs--))
        fi
    else
        echo "Skipping existing output: $output_csv"
    fi
}

# List of privacy techniques to cover all specified scenarios
privacy_techniques=("LDP" "GDP" "k-anonymity" "l-diversity" "t-closeness" "HE" "SMC" 
                    "LDP+HE" "GDP+HE" "LDP+SMC" "GDP+SMC" "HE+SMC")

# Run simulations for each optimizer and privacy technique
for optimizer in "${optimizers[@]}"; do
    for privacy_technique in "${privacy_techniques[@]}"; do
        run_simulation "$optimizer" "$privacy_technique" "fmnist_non_iid" "$q_value" "$learning_rate"
    done
done

# Wait for all background jobs to finish
wait
echo "All simulations complete."

