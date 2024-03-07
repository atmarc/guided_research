# Review of higher-order time stepping schemes in open-source solvers 

This is the repository containing the code of my Guided Research project, conducted at the chair of Scientific Computing in TUM, under the supervision of M.Sc. Benjamin Rodenberg and Prof. Hans-Joachim Bungartz.

We performed several convergence studies on open-source simulation solvers that are compatible with the [preCICE](https://precice.org/) coupling library. The scripts to run the simulations, process the output and create the plots can be found in this repository, organized in the following way:

* `calculix_analisis`: convergence study of the [Calculix](www.calculix.de) solver, using the perpenicular flap scenario applying a constant force.
* `openfoam_analisis`: convergence study of the [OpenFOAM](https://www.openfoam.com/) solver, using the Taylor-Green vortex scenario.
* `coupled_openfoam_calculix_v2`: convergence study of the [perpendicular flap](https://precice.org/tutorials-perpendicular-flap.html) scenario, found in the tutorial cases of preCICE. In this folder we test the entire FSI simulation, coupling the two previously evaluated solvers with preCICE.
* `coupled_openfoam_calculix_v3`: in this folder we can find the same study as the previous folder, but utilizing the version 3 of the preCICE library, and of the corresponding [openfoam-adapter](https://github.com/precice/openfoam-adapter/) and [calculix-adapter](https://github.com/precice/calculix-adapter). In addition, it contains a `fake-fluid` script that allows us to test the scenario excluding the fluid component, to verify the correct behaviour of the calculix-adapter.

Every scenario contains a `run_simulations.py` script that is easily configurable, to run multiple simulations automatically. In addition, there is a `plots.py` script tht creates the figures found on the report, by processing the data of the simulations. To reproduce the results one should open the `run_simulations.py` program, define the parameters to investigate (mainly the $\Delta t$), and afterwards run the plots script.

In the folder `report` one can find the final report containing the results and conclusions extracted from this work.

