# Path to your Isaac Sim installation (e.g., ~/.local/share/ov/pkg/isaac-sim-5.0.0)
export ISAAC_SIM_PATH="$HOME/isaacsim"

# Path to this repository
export ISAACLAB_ARENA_PATH="$(pwd)"

# Path to Isaac Lab (assumed to be in submodules)
export ISAACLAB_PATH="$ISAACLAB_ARENA_PATH/submodules/IsaacLab"

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] [-u] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install [LIB]  Install the extensions inside Isaac Arena Lab and learning frameworks as extra dependencies. Default is 'all'."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'."
    echo -e "\t-u, --uv [NAME]      Create the uv environment for Isaac Lab. Default name is 'env_isaaclab'."
    echo -e "\n" >&2
}


# extract isaac sim path
extract_isaacsim_path() {
    # Use the sym-link path to Isaac Sim directory
    local isaac_path=${ISAACLAB_PATH}/_isaac_sim
    # If above path is not available, try to find the path using python
    if [ ! -d "${isaac_path}" ]; then
        # Use the python executable to get the path
        local python_exe=$(extract_python_exe)
        # Retrieve the path importing isaac sim and getting the environment path
        if [ $(${python_exe} -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
            local isaac_path=$(${python_exe} -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])")
        fi
    fi
    # check if there is a path available
    if [ ! -d "${isaac_path}" ]; then
        # throw an error if no path is found
        echo -e "[ERROR] Unable to find the Isaac Sim directory: '${isaac_path}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Isaac Sim directory is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim" >&2
        # exit the script
        exit 1
    fi
    # return the result
    echo ${isaac_path}
}

# extract the python from isaacsim
extract_python_exe() {
    # check if using conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # use conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    elif ! [[ -z "${VIRTUAL_ENV}" ]]; then
        # use uv virtual environment python
        local python_exe=${VIRTUAL_ENV}/bin/python
    else
        # use kit python
        local python_exe=${ISAACLAB_PATH}/_isaac_sim/python.sh

    if [ ! -f "${python_exe}" ]; then
            # note: we need to check system python for cases such as docker
            # inside docker, if user installed into system python, we need to use that
            # otherwise, use the python from the kit
            if [ $(python -m pip list | grep -c 'isaacsim-rl') -gt 0 ]; then
                local python_exe=$(which python)
            fi
        fi
    fi
    # check if there is a python path available
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find any Python executable at path: '${python_exe}'" >&2
        echo -e "\tThis could be due to the following reasons:" >&2
        echo -e "\t1. Conda or uv environment is not activated." >&2
        echo -e "\t2. Isaac Sim pip package 'isaacsim-rl' is not installed." >&2
        echo -e "\t3. Python executable is not available at the default path: ${ISAACLAB_PATH}/_isaac_sim/python.sh" >&2
        exit 1
    fi
    # return the result
    echo ${python_exe}
}

# find pip command based on virtualization
extract_pip_command() {
    # detect if we're in a uv environment
    if [ -n "${VIRTUAL_ENV}" ] && [ -f "${VIRTUAL_ENV}/pyvenv.cfg" ] && grep -q "uv" "${VIRTUAL_ENV}/pyvenv.cfg"; then
        pip_command="uv pip install"
    else
        # retrieve the python executable
        python_exe=$(extract_python_exe)
        pip_command="${python_exe} -m pip install"
    fi

    echo ${pip_command}
}

extract_pip_uninstall_command() {
    # detect if we're in a uv environment
    if [ -n "${VIRTUAL_ENV}" ] && [ -f "${VIRTUAL_ENV}/pyvenv.cfg" ] && grep -q "uv" "${VIRTUAL_ENV}/pyvenv.cfg"; then
        pip_uninstall_command="uv pip uninstall"
    else
        # retrieve the python executable
        python_exe=$(extract_python_exe)
        pip_uninstall_command="${python_exe} -m pip uninstall -y"
    fi

    echo ${pip_uninstall_command}
}

# check if input directory is a python extension and install the module
install_isaaclab_arena_extension() {
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    pip_command=$(extract_pip_command)

    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\t module: $1"
        $pip_command --editable "$1"
    fi
}

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 0
fi

# pass the arguments
while [[ $# -gt 0 ]]; do
    # read the key
    case "$1" in
        -i|--install)
            # install the python packages in IsaacLab-Arena/submodules directory
            echo "[INFO] Installing extensions inside the Isaac Lab Arena repository..."
            python_exe=$(extract_python_exe)
            pip_command=$(extract_pip_command)
            pip_uninstall_command=$(extract_pip_uninstall_command)
            export -f extract_python_exe
            export -f extract_pip_command
            export -f extract_pip_uninstall_command
            if [ -z "$2" ]; then
                framework_name="all"
            elif [ "$2" = "none" ]; then
                framework_name="none"
                shift # past argument
            else
                framework_name=$2
                shift # past argument
            fi
            $ISAACLAB_PATH/isaaclab.sh -i framework_name
            ${pip_command} -e "${ISAACLAB_ARENA_PATH}/submodules/Isaac-GR00T"
            # Install IsaacLab Arena
            pip install -e .
            # unset local variables
            unset extract_python_exe
            unset extract_pip_command
            unset extract_pip_uninstall_command
            shift # past argument
            ;;
        -c|--conda)
            # use default name if not provided
            if [ -z "$2" ]; then
                conda_env_name="env_isaaclab_arena"
            else
                conda_env_name=$2
                shift # past argument
            fi
            # setup the conda environment for Isaac Lab
            $ISAACLAB_PATH/isaaclab.sh --conda ${conda_env_name}
            shift # past argument
            ;;
        -u|--uv)
            # use default name if not provided
            if [ -z "$2" ]; then
                uv_env_name="env_isaaclab_arena"
            else
                uv_env_name=$2
                shift # past argument
            fi
            # setup the uv environment for Isaac Lab
            $ISAACLAB_PATH/isaaclab.sh --uv ${conda_env_name}
            shift # past argument
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
    esac
done