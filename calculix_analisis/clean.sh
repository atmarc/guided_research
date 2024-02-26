echo "--- Cleaning up CalculiX case in $(pwd)"
set -e -u
rm -fv ./*.cvg ./*.dat ./*.frd ./*.sta ./*.12d spooles.out dummy
rm -fv WarnNodeMissMultiStage.nam
rm -fv ./*.eig
# clean_precice_logs .

# rm output/*