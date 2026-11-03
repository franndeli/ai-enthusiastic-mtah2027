#!/bin/sh
# Create or update the project Conda environment in ./.conda-env from environment.yml.
# Never installs into base or the system Python.
set -eu
cd "$(dirname "$0")/.."
if [ -d .conda-env ]; then
  conda env update --prefix ./.conda-env --file environment.yml --prune
else
  conda env create --prefix ./.conda-env --file environment.yml
fi
# macOS: the torch wheel bundles its own OpenMP runtime, which aborts when the env's conda OpenMP
# is already loaded (OMP Error #15). Point torch at the env's runtime instead.
if [ "$(uname)" = "Darwin" ]; then
  lib=$(./.conda-env/bin/python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('torch').origin), 'lib'))")
  if [ -f "$lib/libomp.dylib" ] && [ ! -L "$lib/libomp.dylib" ] && [ -f ./.conda-env/lib/libomp.dylib ]; then
    mv "$lib/libomp.dylib" "$lib/libomp.dylib.torch"
    ln -s "$(python3 -c "import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))" "$(pwd)/.conda-env/lib/libomp.dylib" "$lib")" "$lib/libomp.dylib"
  fi
fi
./.conda-env/bin/python -c "import numpy, pandas, kloppy, matplotlib, sklearn, torch, torch_geometric; print('env ok: numpy', numpy.__version__, '| pandas', pandas.__version__, '| kloppy', kloppy.__version__, '| torch', torch.__version__, '| torch_geometric', torch_geometric.__version__)"
