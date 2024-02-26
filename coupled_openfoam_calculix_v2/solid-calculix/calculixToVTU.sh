#!/bin/sh

python ccx2paraview-3.1.0/ccx2paraview.py flap.frd vtu
mkdir output/$1
mv *.vtu output/$1