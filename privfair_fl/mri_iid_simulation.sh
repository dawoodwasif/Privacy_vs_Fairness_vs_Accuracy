#!/bin/bash


### MRI IID SIMULATION
for i in {1..10}; do
    # Create a new bash script for each run
    script_name="run_iid_qffedavg_run${i}.sh"
    echo "#!/bin/bash" > $script_name
    echo "python -u main.py \\
    --dataset=mri_iid \\
    --optimizer=qffedavg \\
    --learning_rate=0.001 \\
    --learning_rate_lambda=0.01 \\
    --num_rounds=20 \\
    --eval_every=1 \\
    --clients_per_round=5 \\
    --batch_size=64 \\
    --q=0 \\
    --model='cnn' \\
    --sampling=2 \\
    --num_epochs=40 \\
    --data_partition_seed=1 \\
    --log_interval=10 \\
    --static_step_size=0 \\
    --track_individual_accuracy=0 \\
    --output=\"./log_mri_iid/qffedavg_iid_run${i}\" \\
    --run_number ${i}" >> $script_name

    # Make the script executable
    chmod +x $script_name
    # Run the generated script
    ./$script_name
done


### MRI NON IID SIMULATION
for i in {1..10}; do
    # Create a new bash script for each run
    script_name="run_non_iid_qffedavg_run${i}.sh"
    echo "#!/bin/bash" > $script_name
    echo "python -u main.py \\
    --dataset=mri_non_iid \\
    --optimizer=qffedavg \\
    --learning_rate=0.001 \\
    --learning_rate_lambda=0.01 \\
    --num_rounds=20 \\
    --eval_every=1 \\
    --clients_per_round=5 \\
    --batch_size=64 \\
    --q=0 \\
    --model='cnn' \\
    --sampling=2 \\
    --num_epochs=40 \\
    --data_partition_seed=1 \\
    --log_interval=10 \\
    --static_step_size=0 \\
    --track_individual_accuracy=0 \\
    --output=\"./log_mri_non_iid/qffedavg_non_iid_run${i}\" \\
    --run_number ${i}" >> $script_name

    # Make the script executable
    chmod +x $script_name
    # Run the generated script
    ./$script_name
done
