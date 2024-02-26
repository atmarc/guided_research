#!/bin/sh

date 
ccx flap > /dev/null
python ccx2paraview-3.1.0/ccx2paraview.py flap.frd vtu
mkdir output/$1
mv *.vtu output/$1


# usage() { echo "Usage: run.sh [-modal]" 1>&2; exit 1; }

# # There must be either 0 arguments or 1, which is modal.
# # Send an error otherwise
# if [ $# -ge 2 ] || { [ $# -eq 1 ] && [ "$1" != "-modal" ]; }; then
#     usage
# fi

# # No arg => regular simulation. Otherwise, it's modal
# if [ $# -eq 0 ]; then
#     ccx_preCICE -i flap -precice-participant Solid
# else
#     ccx_preCICE -i frequency
#     mv frequency.eig flap_modal.eig
#     ccx_preCICE -i flap_modal -precice-participant Solid
# fi