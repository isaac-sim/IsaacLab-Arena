# Path to this repository
export ISAACLAB_ARENA_PATH="$(pwd)"

# Path to Isaac Lab (assumed to be in submodules)
export ISAACLAB_PATH="$ISAACLAB_ARENA_PATH/submodules/IsaacLab"

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] [-u] -- Utility to manage Isaac Lab."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install [LIB]  Install the extensions inside Isaac Lab Arena and learning frameworks as extra dependencies. Default is 'all'."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Isaac Lab Arena. Default name is 'env_isaaclab_arena'."
    echo -e "\t-u, --uv [NAME]      Create the uv environment for Isaac Lab Arena. Default name is 'env_isaaclab_arena'."
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

# setup uv environment for Isaac Lab Arena
setup_uv_env() {
    # get environment name from input
    local env_name="$1"
    local python_path="$2"

    # check uv is installed
    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv could not be found. Please install uv and try again."
        echo "[ERROR] uv can be installed here:"
        echo "[ERROR] https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    # check if _isaac_sim symlink exists and isaacsim-rl is not installed via pip
    if [ ! -L "${ISAACLAB_PATH}/_isaac_sim" ] && ! python -m pip list | grep -q 'isaacsim-rl'; then
        echo -e "[WARNING] _isaac_sim symlink not found at ${ISAACLAB_PATH}/_isaac_sim"
        echo -e "\tThis warning can be ignored if you plan to install Isaac Sim via pip."
        echo -e "\tIf you are using a binary installation of Isaac Sim, please ensure the symlink is created before setting up the conda environment."
    fi

    # check if the environment exists
    local env_path="${ISAACLAB_ARENA_PATH}/${env_name}"
    if [ ! -d "${env_path}" ]; then
        echo -e "[INFO] Creating uv environment named '${env_name}'..."
        uv venv --clear --python "${python_path}" "${env_path}"
        # Install pip so isaaclab.sh can use python -m pip
        echo -e "[INFO] Installing pip in uv environment..."
        uv pip install --python "${env_path}/bin/python" pip
    else
        echo "[INFO] uv environment '${env_name}' already exists."
    fi

    # define root path for activation hooks
    local isaaclab_root="${ISAACLAB_ARENA_PATH}"

    # cache current paths for later
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH

    # ensure activate file exists
    touch "${env_path}/bin/activate"

     # add variables to environment during activation
    cat >> "${env_path}/bin/activate" <<EOF
export ISAACLAB_PATH="${ISAACLAB_PATH}"
alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"
alias isaaclab_arena="${ISAACLAB_ARENA_PATH}/isaaclab_arena.sh"
export RESOURCE_NAME="IsaacSim"

if [ -f "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh" ]; then
    . "${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh"
fi
EOF

    # add information to the user about alias
    echo -e "[INFO] Added 'isaaclab' alias to uv environment for 'isaaclab.sh' script."
    echo -e "[INFO] Created uv environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                source ${env_name}/bin/activate."
    echo -e "\t\t2. To install Isaac Lab Arena extensions, run:      isaaclab_arena -i"
    echo -e "\t\t3. To perform formatting, run:                      isaaclab -f"
    echo -e "\t\t4. To deactivate the environment, run:              deactivate"
    echo -e "\n"
}


# update the pycharm settings from template and isaac sim settings
update_pycharm_settings() {
    echo "[INFO] Setting up PyCharm settings..."
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # path to setup_pycharm.py
    setup_pycharm_script="${ISAACLAB_ARENA_PATH}/.idea/tools/setup_pycharm.py"
    # check if the file exists before attempting to run it
    if [ -f "${setup_pycharm_script}" ]; then
        ${python_exe} "${setup_pycharm_script}"
    else
        echo "[WARNING] Unable to find the script 'setup_pycharm.py'. Aborting PyCharm settings setup."
    fi

    # Update run configurations with environment variables
    update_run_configs_script="${ISAACLAB_ARENA_PATH}/.idea/tools/update_run_configs.py"
    if [ -f "${update_run_configs_script}" ]; then
        echo "[INFO] Updating PyCharm run configurations with environment variables..."
        ${python_exe} "${update_run_configs_script}"
    else
        echo "[WARNING] Unable to find the script 'update_run_configs.py'. Run configurations may not have proper environment variables."
    fi
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
            # Install IsaacLab
            $ISAACLAB_PATH/isaaclab.sh -i $framework_name
            # Install Isaac-GR00T
            ${pip_command} -e "${ISAACLAB_ARENA_PATH}/submodules/Isaac-GR00T"
            # Install IsaacLab Arena
            ${pip_command} -e .
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
                echo "[INFO] Using default uv environment name: env_isaaclab_arena"
                uv_env_name="env_isaaclab_arena"
            else
                echo "[INFO] Using uv environment name: $2"
                uv_env_name=$2
                shift # past argument
            fi
            # setup the uv environment for Isaac Lab
            setup_uv_env ${uv_env_name}
            shift # past argument
            ;;
        -y|--pycharm)
            # update the pycharm settings
            update_pycharm_settings
            shift # past argument
            # exit neatly
            break
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
    esac
done