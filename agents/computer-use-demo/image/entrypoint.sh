#!/bin/bash
set -e

# Initialize the environment
./start_all.sh
./novnc_startup.sh

# Start the HTTP server in the background
python http_server.py > /tmp/server_logs.txt 2>&1 &

# Extract container instance number from hostname if possible
HOSTNAME=$(hostname)
CONTAINER_INSTANCE_NUM=""

# Try to extract instance number from hostname
if [[ $HOSTNAME =~ -([0-9]+)$ ]]; then
    CONTAINER_INSTANCE_NUM="${BASH_REMATCH[1]}"
    echo "Detected container instance number: $CONTAINER_INSTANCE_NUM"
else
    # If hostname doesn't contain the instance number, try to get it from
    # DISPLAY_NUM environment variable which is set during container launch
    if [ ! -z "$DISPLAY_NUM" ]; then
        CONTAINER_INSTANCE_NUM="$DISPLAY_NUM"
        echo "Using DISPLAY_NUM as instance number: $CONTAINER_INSTANCE_NUM"
    else
        echo "Could not determine container instance number"
    fi
fi

# Set shared directory path
SHARED_DIR="/home/computeruse/shared_data"

# Run run_prompts.py as a module with instance number and mark-completed flag
if [ ! -z "$CONTAINER_INSTANCE_NUM" ]; then
    python -m computer_use_demo.run_prompts \
        --instance-num "$CONTAINER_INSTANCE_NUM" \
        --shared-dir "$SHARED_DIR" \
        --mark-completed > /tmp/run_prompts_stdout.log 2>&1 &
    echo "➡️  Looking for prompts for instance: $CONTAINER_INSTANCE_NUM"
else
    python -m computer_use_demo.run_prompts \
        --shared-dir "$SHARED_DIR" \
        --mark-completed > /tmp/run_prompts_stdout.log 2>&1 &
    echo "➡️  Looking for prompts with default settings"
fi

echo "✨ Script is running!"
echo "➡️  Check /tmp/run_prompts_stdout.log for output"

# Keep the container running
tail -f /dev/null