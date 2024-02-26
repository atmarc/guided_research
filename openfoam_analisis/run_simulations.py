from jinja2 import Environment, select_autoescape, FileSystemLoader
from pathlib import Path
import subprocess
import os
import numpy as np
from vels_gen import write_init_vels


def render(template_path, output_path, config_params):
    base_path = template_path.parent.absolute()
    env = Environment(loader=FileSystemLoader(base_path))
    template = env.get_template(template_path.name)

    with open(output_path, "w") as file:
        file.write(template.render(config_params))


def setup_simulation(config):
    print("Setting up simulation with parameters", config)
    templates_fldr = Path("templates")
    system_flder = Path("system")
    write_init_vels(out_filename=Path("0") / "init_vels", N=config['Nx'])

    render(templates_fldr / "template-controlDict",   system_flder / "controlDict",   config)
    render(templates_fldr / "template-fvSchemes",     system_flder / "fvSchemes",     config)
    render(templates_fldr / "template-blockMeshDict", system_flder / "blockMeshDict", config)
    render(templates_fldr / "template-fvSolution",    system_flder / "fvSolution",    config)


def run_simulation(options, base_path="out", runfile="run.sh"):
    print("Running simulation...")
    os.system(f'./{runfile}')

    # Make sure the directory exists, or create it
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True) 
    
    # Last computed file
    latest_time = subprocess.check_output("foamListTimes -latestTime", shell=True, text=True)
    vels_file = Path(latest_time.strip()) / "U"
    
    # Move the last U folder to defined output 
    file_name = f'dt{options["timeStep"]}_N{options["Nx"]}_E{options["endTime"]}_C{options["ddtFactor"]}'
    output_f = base_dir / file_name
    print("Moving output to", output_f)
    vels_file.rename(output_f)  


def clean_results(runfile="clean.sh"):
    print("Cleaning previous output files")
    os.system(f'./{runfile}')


if __name__ == "__main__":
    # timesteps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # timesteps = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    timesteps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    base_path = Path("output") / "Uvels_N150_C1"

    for dt in timesteps:
        clean_results()

        options = {
            'timeStep': dt,
            'endTime': 1,
            'writeInterval': round(1 / dt),
            'ddtScheme': 'CrankNicolson',
            'ddtFactor': 1,
            'Nx': 150,
            'Ny': 150,
            'pTolerance': 1e-16,
            'uTolerance': 1e-16,
            'writeFormat': 'ascii',
            'writePrecision': 14,
            'timePrecision': 14,
            'nCorrectors': 2
        }

        setup_simulation(options)

        run_simulation(options, base_path=base_path)

