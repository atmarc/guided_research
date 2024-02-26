#!/bin/sh
set -e -u

blockMesh
touch fluid-openfoam.foam

../../tutorials/tools/run-openfoam.sh "$@"
. ../../tutorials/tools/openfoam-remove-empty-dirs.sh && openfoam_remove_empty_dirs

