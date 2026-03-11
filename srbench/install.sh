#!/bin/bash
# Install pyjessamine and its dependencies for SRBench
#
# SRBench requirement: "Do not include your source code.
# Use install.sh to pull it from a stable source repository."

set -e

REPO_URL="https://github.com/riordanaa/pyjessamine.git"
INSTALL_DIR="${CONDA_PREFIX:-$HOME}/.pyjessamine_src"

# Install Python dependencies (juliacall handles Julia installation)
pip install numpy scikit-learn sympy "juliacall>=0.9.14"

# Clone the repository (or pull latest if already cloned)
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing clone at $INSTALL_DIR"
    cd "$INSTALL_DIR" && git pull
else
    echo "Cloning $REPO_URL to $INSTALL_DIR"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Install pyjessamine package
pip install -e "$INSTALL_DIR/python"

# Pre-compile Jessamine.jl to avoid first-run latency during benchmarks
python -c "
import os
os.environ['JESSAMINE_NO_PYCALL'] = '1'
import juliacall
jl = juliacall.Main
jl.seval('using Pkg')
repo_root = '$INSTALL_DIR'.replace('\\\\', '/')
jl.seval(f'Pkg.activate(\"{repo_root}\")')
jl.seval('Pkg.instantiate()')
jl.seval('using Jessamine')
print('Jessamine.jl precompilation complete')
"
