#!/bin/bash
# Install pyjessamine and its dependencies for SRBench

set -e

# Install Python dependencies (juliacall handles Julia installation)
pip install numpy scikit-learn sympy "juliacall>=0.9.14"

# Install pyjessamine from the repo
pip install -e "$(dirname "$0")/../python"

# Pre-compile Jessamine.jl to avoid first-run latency during benchmarks
python -c "
import os
os.environ['JESSAMINE_NO_PYCALL'] = '1'
import juliacall
jl = juliacall.Main
jl.seval('using Pkg')
repo_root = os.path.normpath(os.path.join('$(dirname \"$0\")', '..'))
jl.seval(f'Pkg.activate(\"{repo_root}\")')
jl.seval('Pkg.instantiate()')
jl.seval('using Jessamine')
print('Jessamine.jl precompilation complete')
"
