Commands
Create:
vtune -collect hotspots -result-dir ./vtune_results -- ./openmp
Visualize:
vtune-gui openmp/vtune_results/vtune_results.vtune

make run USER=valentin.carauleanu MAIN_DIR=serial EXEC=serial