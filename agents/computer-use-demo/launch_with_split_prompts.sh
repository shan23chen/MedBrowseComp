#!/bin/bash
set -e

# --- Configuration ---
IMAGE_NAME="computer-use-demo"
NUM_INSTANCES=2  # Number of containers to launch
SHARED_VOLUME_HOST_PATH="$(pwd)/shared_docker_volume" # Host path for the shared volume
MASTER_CSV="prompts_master.csv"  # Master CSV file with all prompts
RESULTS_CSV="results_complete.csv"  # Final merged results CSV file

# Wait for container completion configuration
WAIT_TIMEOUT=36000  # Maximum time to wait for containers (in seconds)
CHECK_INTERVAL=30  # Time between checks (in seconds)

# Default display settings
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=768

# --- Prepare shared volume ---
echo "--- Setting up Shared Volume ---"
if [ ! -d "$SHARED_VOLUME_HOST_PATH" ]; then
    echo "Creating shared volume directory: $SHARED_VOLUME_HOST_PATH"
    mkdir -p "$SHARED_VOLUME_HOST_PATH"
else
    echo "Shared volume directory already exists: $SHARED_VOLUME_HOST_PATH"
fi

# Copy master CSV to shared volume if it's not already there
if [ ! -f "$SHARED_VOLUME_HOST_PATH/$MASTER_CSV" ]; then
    echo "Copying master CSV to shared volume"
    cp "$MASTER_CSV" "$SHARED_VOLUME_HOST_PATH/"
fi

# --- Split the master CSV file into per-container CSVs ---
echo "--- Splitting Prompts CSV for $NUM_INSTANCES Containers ---"
python split_csv.py --input "$SHARED_VOLUME_HOST_PATH/$MASTER_CSV" \
                    --output "$SHARED_VOLUME_HOST_PATH" \
                    --containers "$NUM_INSTANCES" \
                    --prefix "$IMAGE_NAME-instance-"

# Check if split was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to split CSV file. Exiting."
    exit 1
fi

# --- Launch containers using existing script ---
echo "--- Launching $NUM_INSTANCES Containers ---"
./launch_containers.sh -i "$IMAGE_NAME" -w "$DEFAULT_WIDTH" -h "$DEFAULT_HEIGHT" -n "$NUM_INSTANCES" -s "$SHARED_VOLUME_HOST_PATH"

# --- Wait for containers to process prompts ---
echo "--- Waiting for Containers to Process Prompts ---"
echo "Will check every $CHECK_INTERVAL seconds, timeout after $WAIT_TIMEOUT seconds"

start_time=$(date +%s)
completed=0
container_ids=()

# Get list of container IDs
for ((i=1; i<=$NUM_INSTANCES; i++)); do
    container_name="${IMAGE_NAME}-instance-${i}"
    container_id=$(docker ps -q -f name=${container_name})
    
    if [ -n "$container_id" ]; then
        container_ids+=("$container_id")
    else
        echo "Warning: Container $container_name not found!"
    fi
done

# Wait for all containers to complete or timeout
while [ $completed -lt ${#container_ids[@]} ] && [ $(( $(date +%s) - start_time )) -lt $WAIT_TIMEOUT ]; do
    completed=0
    
    # Check for completion marker files
    for ((i=1; i<=$NUM_INSTANCES; i++)); do
        csv_file="$SHARED_VOLUME_HOST_PATH/${IMAGE_NAME}-instance-${i}.csv"
        completed_marker="${csv_file}.completed"
        
        if [ -f "$completed_marker" ]; then
            completed=$((completed + 1))
        fi
    done
    
    # Print status
    time_elapsed=$(( $(date +%s) - start_time ))
    echo "Progress: $completed/$NUM_INSTANCES containers completed processing ($time_elapsed seconds elapsed)"
    
    # If not all containers completed, wait before checking again
    if [ $completed -lt ${#container_ids[@]} ]; then
        sleep $CHECK_INTERVAL
    fi
done

# Check if we timed out
if [ $completed -lt ${#container_ids[@]} ]; then
    echo "WARNING: Timed out waiting for all containers to complete!"
    echo "Proceeding with merging partial results."
else
    echo "All containers have completed processing their prompts!"
fi

# --- Merge results from all containers ---
echo "--- Merging Results from All Containers ---"
python merge_csv.py --input "$SHARED_VOLUME_HOST_PATH" \
                   --output "$SHARED_VOLUME_HOST_PATH/$RESULTS_CSV" \
                   --prefix "$IMAGE_NAME-instance-"

# Check if merge was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to merge CSV results. Check logs for details."
    exit 1
fi

# --- Final Instructions ---
echo
echo "--- Processing Summary ---"
echo "Complete merged results file: $SHARED_VOLUME_HOST_PATH/$RESULTS_CSV"
echo
echo "You can stop all containers launched by this script with:"
echo "  docker stop \$(docker ps -q --filter name=computer-use-demo-instance-)"
echo
echo "To remove the stopped containers:"
echo "  docker rm \$(docker ps -aq --filter status=exited --filter name=computer-use-demo-instance-)"