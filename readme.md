Commands
Create:
vtune -collect hotspots -result-dir ./vtune_results -- ./openmp
Visualize:
vtune-gui openmp/vtune_results/vtune_results.vtune
