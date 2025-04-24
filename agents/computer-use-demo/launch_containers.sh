#!/bin/bash

# --- Configuration ---
IMAGE_NAME="computer-use-demo"
DEFAULT_NUM_INSTANCES=2 # Default number of instances if not provided
SHARED_VOLUME_HOST_PATH="$(pwd)/shared_docker_volume" # Host path for the shared volume
SHARED_VOLUME_CONTAINER_PATH="/home/computeruse/shared_data" # Mount point inside the container
CSV_FILENAME="shared_data.csv"       # Name of the shared CSV file

# Default display settings
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=768

# Ports required by the container internally that need to be published
CONTAINER_PORTS=(6080 8000)   # VNC and HTTP ports from your entrypoint.sh

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

# Function to check if a port is available
check_port_available() {
    local port=$1
    if command_exists netstat; then
        if netstat -tuln | grep -q ":$port "; then
            return 1  # Port is in use
        else
            return 0  # Port is available
        fi
    elif command_exists ss; then
        if ss -tuln | grep -q ":$port "; then
            return 1  # Port is in use
        else
            return 0  # Port is available
        fi
    else
        echo "Warning: Neither netstat nor ss found. Port availability check might not be reliable."
        return 0  # Assume port is available
    fi
}

# Function to find an available port starting from a base port
find_available_port() {
    local base_port=$1
    local port=$base_port
    local max_attempts=100
    local attempt=0
    
    while ! check_port_available $port && [ $attempt -lt $max_attempts ]; do
        port=$((port + 1))
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error_exit "Could not find an available port after $max_attempts attempts."
    fi
    
    echo $port
}

# Function to get the assigned host port for a container port
get_host_port() {
    local container_id_or_name="$1"
    local container_port="$2"
    local port_info
    local host_port

    # Retry mechanism in case the port info isn't immediately available
    for _ in {1..5}; do
        port_info=$(docker port "$container_id_or_name" "$container_port/tcp" 2>/dev/null)
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

# Function to show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Launch multiple containers using the same image with automatic port mapping."
    echo
    echo "Options:"
    echo "  -i IMAGE_NAME    Specify the Docker image name (default: $IMAGE_NAME)"
    echo "  -w WIDTH         Set the display width (default: $DEFAULT_WIDTH)"
    echo "  -h HEIGHT        Set the display height (default: $DEFAULT_HEIGHT)"
    echo "  -n NUM           Set the number of containers to launch (default: prompt user)"
    echo "  -s SHARED_PATH   Set the host path for shared volume (default: $SHARED_VOLUME_HOST_PATH)"
    echo "  -b               Build the Docker image before launching containers"
    echo "  -f DOCKERFILE    Path to Dockerfile (default: ./Dockerfile)"
    echo "  -?               Show this help message"
    echo
    exit 0
}

# --- Parse command-line options ---
BUILD_IMAGE=false
DOCKERFILE_PATH="./Dockerfile"
NUM_INSTANCES=""

while getopts "i:w:h:n:s:bf:?" opt; do
    case $opt in
        i) IMAGE_NAME="$OPTARG" ;;
        w) DEFAULT_WIDTH="$OPTARG" ;;
        h) DEFAULT_HEIGHT="$OPTARG" ;;
        n) NUM_INSTANCES="$OPTARG" ;;
        s) SHARED_VOLUME_HOST_PATH="$OPTARG" ;;
        b) BUILD_IMAGE=true ;;
        f) DOCKERFILE_PATH="$OPTARG" ;;
        ?) show_usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

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

# Build the image if requested
if [ "$BUILD_IMAGE" = true ]; then
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        error_exit "Dockerfile not found at $DOCKERFILE_PATH"
    fi
    
    echo "Building Docker image: $IMAGE_NAME"
    if ! docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .; then
        error_exit "Failed to build Docker image."
    fi
    echo "[OK] Docker image built successfully."
fi

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
        error_exit "Docker image '$IMAGE_NAME' not found. Please build it first."
    fi
else
    echo "[OK] Docker image '$IMAGE_NAME' found locally."
fi

echo "--- Pre-flight Checks Passed ---"
echo

# --- Get Number of Instances if not provided via command line ---
if [ -z "$NUM_INSTANCES" ]; then
    read -p "How many container instances do you want to launch? [Default: $DEFAULT_NUM_INSTANCES]: " num_instances_input
    NUM_INSTANCES=${num_instances_input:-$DEFAULT_NUM_INSTANCES}
fi

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
# Arrays to store container information and access URLs
declare -a container_info=()
declare -a vnc_urls=()

# Base ports for services
BASE_VNC_PORT=6080
BASE_HTTP_PORT=8000

for (( i=1; i<=NUM_INSTANCES; i++ ))
do
    echo "Launching instance $i of $NUM_INSTANCES..."
    
    # Find available ports
    vnc_port=$(find_available_port $BASE_VNC_PORT)
    http_port=$(find_available_port $BASE_HTTP_PORT)
    
    # Create a unique container name
    container_name="${IMAGE_NAME}-instance-${i}"
    
    # This is the key change: We set a custom hostname that matches the CSV naming pattern
    container_hostname="$container_name"
    
    echo "  Container Name: $container_name"
    echo "  Container Hostname: $container_hostname"
    echo "  VNC Port: $vnc_port (mapped from 6080)"
    echo "  HTTP Port: $http_port (mapped from 8000)"
    
    # Launch the container with explicit hostname set
    container_id=$(docker run -d \
        --name "$container_name" \
        --hostname "$container_hostname" \
        -e API_PROVIDER=vertex \
        -e CLOUD_ML_REGION=$VERTEX_REGION \
        -e ANTHROPIC_VERTEX_PROJECT_ID=$VERTEX_PROJECT_ID \
        -p "$vnc_port:6080" \
        -p "$http_port:8000" \
        -e DISPLAY_NUM="$i" \
        -e HEIGHT="$DEFAULT_HEIGHT" \
        -e WIDTH="$DEFAULT_WIDTH" \
        -v $HOME/.config/gcloud/application_default_credentials.json:/home/computeruse/.config/gcloud/application_default_credentials.json \
        -v "$SHARED_VOLUME_HOST_PATH:$SHARED_VOLUME_CONTAINER_PATH" \
        "$IMAGE_NAME")
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start container instance $i ($container_name)." >&2
        vnc_urls+=("Instance $i ($container_name): FAILED_TO_START")
    else
        echo "  Container ID: $container_id"
        echo "  Shared Volume: '$SHARED_VOLUME_HOST_PATH' (Host) -> '$SHARED_VOLUME_CONTAINER_PATH' (Container)"
        
        # Store container information
        container_info+=("Container: $container_name | VNC Port: $vnc_port | HTTP Port: $http_port")
        vnc_urls+=("Instance $i ($container_name): http://localhost:$vnc_port/vnc.html")
        
        # Increment base ports for next iteration to reduce search time
        BASE_VNC_PORT=$((vnc_port + 1))
        BASE_HTTP_PORT=$((http_port + 1))
    fi
    
    echo "---"
done

echo "--- Container Launch Process Complete ---"
echo

# --- Output VNC Access URLs ---
echo "--- VNC Access URLs ---"
if [ ${#vnc_urls[@]} -eq 0 ]; then
    echo "No container instances were successfully launched."
else
    printf "%s\n" "${vnc_urls[@]}" # Print each URL on a new line
fi
echo "-----------------------------"
echo

# --- Container Launch Summary ---
if [ ${#container_info[@]} -gt 0 ]; then
    echo "Container Launch Summary:"
    echo "========================="
    for info in "${container_info[@]}"; do
        echo "$info"
    done
    echo "========================="
    echo
fi

# --- Final Instructions ---
echo "You can list running containers using: docker ps"
echo "To see all ports for a specific container: docker port <container_name_or_id>"
echo "To stop a specific container: docker stop <container_name_or_id>"
echo "To stop all containers launched by this script (based on name prefix):"
echo "  docker stop \$(docker ps -q --filter name=${IMAGE_NAME}-instance-)"
echo "To remove stopped containers:"
echo "  docker rm \$(docker ps -aq --filter status=exited --filter name=${IMAGE_NAME}-instance-)"
echo "Shared data is available in: $SHARED_VOLUME_HOST_PATH"