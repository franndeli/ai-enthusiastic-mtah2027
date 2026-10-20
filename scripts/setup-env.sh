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
./.conda-env/bin/python -c "import numpy, pandas, kloppy, matplotlib; print('env ok: numpy', numpy.__version__, '| pandas', pandas.__version__, '| kloppy', kloppy.__version__)"
