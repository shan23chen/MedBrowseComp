#!/bin/bash

# --- Configuration ---
IMAGE_NAME="computer-use-demo"
DEFAULT_NUM_INSTANCES=2 # Default number of instances if not provided
SHARED_VOLUME_HOST_PATH="$(pwd)/shared_docker_volume" # Host path for the shared volume (created in the script's directory)
SHARED_VOLUME_CONTAINER_PATH="/data" # Mount point inside the container
CSV_FILENAME="shared_data.csv"       # Name of the shared CSV file
STREAMLIT_INTERNAL_PORT=8501         # Define the internal Streamlit port

# Ports required by the container internally that need to be published
# Ensure STREAMLIT_INTERNAL_PORT is included here if it needs mapping
CONTAINER_PORTS=(5900 ${STREAMLIT_INTERNAL_PORT} 6080 8080)
# Remove potential duplicates if STREAMLIT_INTERNAL_PORT was already in the list
CONTAINER_PORTS=($(printf "%s\n" "${CONTAINER_PORTS[@]}" | sort -u))


# --- Helper Functions ---

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print error messages and exit
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Function to get the assigned host port for a container port
# Uses `docker port` and extracts the numeric port after the last colon
get_host_port() {
    local container_id_or_name="$1"
    local container_port="$2"
    local port_info
    local host_port

    # Retry mechanism in case the port info isn't immediately available
    for _ in {1..5}; do
        port_info=$(docker port "$container_id_or_name" "$container_port/tcp" 2>/dev/null) # Specify TCP
        if [ -n "$port_info" ]; then
            # Extract the port number after the last colon (handles IPv4/IPv6)
            host_port=$(echo "$port_info" | awk -F: '{print $NF}' | head -n 1)
            if [[ "$host_port" =~ ^[0-9]+$ ]]; then
                echo "$host_port"
                return 0
            fi
        fi
        sleep 0.5 # Wait briefly before retrying
    done

    echo "N/A" # Return N/A if port couldn't be found after retries
    return 1
}


# --- Sanity Checks ---

echo "--- Running Pre-flight Checks ---"

# Check if Docker is installed
if ! command_exists docker; then
    error_exit "Docker command not found. Please install Docker."
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    error_exit "Docker daemon is not running. Please start Docker."
fi
echo "[OK] Docker is installed and running."

# Check if gcloud is installed (needed for credentials path)
if ! command_exists gcloud; then
    error_exit "gcloud command not found. Please install Google Cloud SDK."
fi
echo "[OK] gcloud command found."

# Check for gcloud application default credentials
GCLOUD_CRED_PATH="$HOME/.config/gcloud/application_default_credentials.json"
if [ ! -f "$GCLOUD_CRED_PATH" ]; then
    error_exit "gcloud application default credentials not found at '$GCLOUD_CRED_PATH'. Run 'gcloud auth application-default login'."
fi
echo "[OK] Found gcloud credentials at '$GCLOUD_CRED_PATH'."

# Check if the Docker image exists
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "WARNING: Docker image '$IMAGE_NAME' not found locally."
    read -p "Do you want to try building it now? (docker build . -t $IMAGE_NAME) [y/N]: " build_confirm
    if [[ "$build_confirm" =~ ^[Yy]$ ]]; then
        echo "Attempting to build '$IMAGE_NAME'..."
        if ! docker build . -t "$IMAGE_NAME"; then
            error_exit "Failed to build Docker image '$IMAGE_NAME'. Please build it manually."
        fi
        echo "[OK] Docker image '$IMAGE_NAME' built successfully."
    else
        error_exit "Docker image '$IMAGE_NAME' not found. Please build it first (e.g., 'docker build . -t $IMAGE_NAME')."
    fi
else
    echo "[OK] Docker image '$IMAGE_NAME' found locally."
fi


# Check for required environment variables
if [ -z "$VERTEX_REGION" ]; then
    read -p "Enter VERTEX_REGION (e.g., us-east5): " VERTEX_REGION
    if [ -z "$VERTEX_REGION" ]; then
        error_exit "VERTEX_REGION environment variable is not set."
    fi
    export VERTEX_REGION # Export it for the 'docker run' commands
fi
echo "[OK] VERTEX_REGION is set to '$VERTEX_REGION'."

if [ -z "$VERTEX_PROJECT_ID" ]; then
    read -p "Enter VERTEX_PROJECT_ID: " VERTEX_PROJECT_ID
    if [ -z "$VERTEX_PROJECT_ID" ]; then
        error_exit "VERTEX_PROJECT_ID environment variable is not set."
    fi
    export VERTEX_PROJECT_ID # Export it for the 'docker run' commands
fi
echo "[OK] VERTEX_PROJECT_ID is set to '$VERTEX_PROJECT_ID'."

echo "--- Pre-flight Checks Passed ---"
echo

# --- Get Number of Instances ---

read -p "How many container instances do you want to launch? [Default: $DEFAULT_NUM_INSTANCES]: " num_instances_input
NUM_INSTANCES=${num_instances_input:-$DEFAULT_NUM_INSTANCES}

# Validate input is a positive integer
if ! [[ "$NUM_INSTANCES" =~ ^[1-9][0-9]*$ ]]; then
    error_exit "Invalid number of instances. Please enter a positive integer."
fi

echo "Will attempt to launch $NUM_INSTANCES instance(s)."
echo "Docker will automatically assign available host ports."
echo "Warning: Ensure your system has enough RAM and resources."
echo

# --- Setup Shared Volume ---

echo "--- Setting up Shared Volume ---"
if [ ! -d "$SHARED_VOLUME_HOST_PATH" ]; then
    echo "Creating shared volume directory: $SHARED_VOLUME_HOST_PATH"
    mkdir -p "$SHARED_VOLUME_HOST_PATH" || error_exit "Failed to create shared volume directory."
else
    echo "Shared volume directory already exists: $SHARED_VOLUME_HOST_PATH"
fi

SHARED_CSV_FILE_PATH="$SHARED_VOLUME_HOST_PATH/$CSV_FILENAME"
if [ ! -f "$SHARED_CSV_FILE_PATH" ]; then
    echo "Creating empty shared CSV file: $SHARED_CSV_FILE_PATH"
    touch "$SHARED_CSV_FILE_PATH" || error_exit "Failed to create shared CSV file."
else
    echo "Shared CSV file already exists: $SHARED_CSV_FILE_PATH"
fi
echo "--- Shared Volume Setup Complete ---"
echo

# --- Launch Containers ---

echo "--- Launching Containers ---"
# Array to store the streamlit URLs for the final summary
declare -a streamlit_urls=()

for (( i=1; i<=NUM_INSTANCES; i++ ))
do
    echo "Launching instance $i of $NUM_INSTANCES..."
    instance_name="${IMAGE_NAME}-instance-${i}"
    port_mappings="" # String to build port flags

    # Add a -p flag for each container port, letting Docker assign the host port
    for container_port in "${CONTAINER_PORTS[@]}"; do
        port_mappings+="-p ${container_port} "
    done

    # Construct the docker run command
    docker_run_cmd="docker run -d --name $instance_name \
        -e API_PROVIDER=vertex \
        -e CLOUD_ML_REGION=$VERTEX_REGION \
        -e ANTHROPIC_VERTEX_PROJECT_ID=$VERTEX_PROJECT_ID \
        -v \"$GCLOUD_CRED_PATH:/home/computeruse/.config/gcloud/application_default_credentials.json:ro\" \
        -v \"$SHARED_VOLUME_HOST_PATH:$SHARED_VOLUME_CONTAINER_PATH\" \
        $port_mappings \
        $IMAGE_NAME"

    echo "Executing: $docker_run_cmd"

    # Execute the command
    container_id=$(eval $docker_run_cmd) # Use eval to handle spaces/quotes correctly

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start container instance $i ($instance_name)." >&2
        streamlit_urls+=("Instance $i ($instance_name): FAILED_TO_START") # Add failure notice
    else
        echo "Successfully launched instance $i:"
        echo "  Container Name: $instance_name"
        echo "  Container ID: $container_id"
        echo "  Fetching assigned ports..."

        # Retrieve and display the dynamically assigned ports
        echo "  Mapped Ports (Host:Container):"
        all_ports_found=true
        streamlit_port_found_for_instance=false
        current_instance_streamlit_url="" # Store URL for this instance

        for container_port in "${CONTAINER_PORTS[@]}"; do
            host_port=$(get_host_port "$container_id" "$container_port")
            if [ "$host_port" == "N/A" ]; then
                echo "    ?:$container_port (Failed to retrieve host port)"
                all_ports_found=false
                # If Streamlit port retrieval failed
                if [ "$container_port" -eq "$STREAMLIT_INTERNAL_PORT" ]; then
                     current_instance_streamlit_url="Instance $i ($instance_name): Streamlit port $STREAMLIT_INTERNAL_PORT mapping retrieval failed"
                     streamlit_port_found_for_instance="failed" # Mark as failed retrieval
                fi
            else
                echo "    $host_port:$container_port"
                # Check if this is the Streamlit port
                if [ "$container_port" -eq "$STREAMLIT_INTERNAL_PORT" ]; then
                    current_instance_streamlit_url="Instance $i ($instance_name): http://localhost:$host_port"
                    streamlit_port_found_for_instance=true # Mark as found
                fi
            fi
        done

        # Add the determined Streamlit URL (or failure message) to the summary array
        if [ "$streamlit_port_found_for_instance" = true ]; then
             streamlit_urls+=("$current_instance_streamlit_url")
        elif [ "$streamlit_port_found_for_instance" = "failed" ]; then
             streamlit_urls+=("$current_instance_streamlit_url")
        else
             # This case should ideally not happen if STREAMLIT_INTERNAL_PORT is in CONTAINER_PORTS
             # but added for robustness
             streamlit_urls+=("Instance $i ($instance_name): Streamlit port $STREAMLIT_INTERNAL_PORT mapping not found in output.")
        fi


        if ! $all_ports_found; then
             echo "  Warning: Could not retrieve all assigned host ports. Check with 'docker port $instance_name'."
        fi

        echo "  Shared Volume: '$SHARED_VOLUME_HOST_PATH' (Host) -> '$SHARED_VOLUME_CONTAINER_PATH' (Container)"
        echo "---"
    fi
    # No sleep needed here usually
done

echo "--- Container Launch Process Complete ---"
echo

# --- Output Streamlit Summary ---
echo "--- Streamlit Access URLs ---"
if [ ${#streamlit_urls[@]} -eq 0 ]; then
    echo "No container instances were attempted or launched."
else
    printf "%s\n" "${streamlit_urls[@]}" # Print each URL/message on a new line
fi
echo "-----------------------------"
echo

# --- Final Instructions ---
echo "You can list running containers using: docker ps"
echo "To see all ports for a specific container: docker port <container_name_or_id>"
echo "To stop a specific container: docker stop <container_name_or_id>"
echo "To stop all containers launched by this script (based on name prefix):"
echo "  docker stop \$(docker ps -q --filter name=${IMAGE_NAME}-instance-)"
echo "To remove stopped containers:"
echo "  docker rm \$(docker ps -aq --filter status=exited --filter name=${IMAGE_NAME}-instance-)"
echo "Shared data is available in: $SHARED_VOLUME_HOST_PATH"

