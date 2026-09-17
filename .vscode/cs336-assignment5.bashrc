# Preserve the user's usual interactive Bash configuration.
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Remove any Python environment inherited from another project.
if declare -F deactivate >/dev/null 2>&1; then
    deactivate
fi
unset VIRTUAL_ENV

# Avoid loading the system CUDA toolkit ahead of the CUDA libraries bundled
# with the Assignment 5 PyTorch environment.
unset LD_LIBRARY_PATH

# Keep external traffic on the configured proxy while sending local vLLM
# requests directly to the server running in this terminal session.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"

# Enter Assignment 5 and activate its project environment.
if cd "$CS336_ASSIGNMENT5_ROOT"; then
    if [ -f ".venv/bin/activate" ]; then
        source ".venv/bin/activate"
    else
        echo "Warning: Assignment 5 .venv not found; run: uv sync --extra gpu"
    fi
else
    echo "Warning: Assignment 5 directory not found: $CS336_ASSIGNMENT5_ROOT"
fi
