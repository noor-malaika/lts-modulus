#!/bin/bash

# Generate a timestamp and log file name
timestamp=$(date +"%Y-%m-%d_%H:%M:%S")
file_name="nohup_$timestamp.out"

# Run the wandb sweep command and redirect output to the log file
nohup wandb sweep --project shell_mgn_sweep_v2 shell_mgn/conf/multi_comp/sweep.yaml > "$file_name" 2>&1 &

# Wait for 5 seconds to ensure the Sweep ID is generated
echo "Waiting for 5 seconds to ensure the Sweep ID is generated..."
sleep 5

# Extract the Sweep ID from the log file
sweep_id=$(grep -oP 'Creating sweep with ID: \K\w+' "$file_name")

# Check if the Sweep ID was found
if [ -z "$sweep_id" ]; then
    echo "Error: Sweep ID not found in log file."
    exit 1
fi

# Print the Sweep ID
echo "Sweep ID: $sweep_id"

# Run the wandb agent command and append output to the same log file
nohup wandb agent --project shell_mgn_sweep_v2 --entity malaikanoor7864-mnsuam "$sweep_id" >> "$file_name" 2>&1 &

# Print the log file location
echo "Log file: $file_name"