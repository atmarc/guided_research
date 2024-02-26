from jinja2 import Environment, select_autoescape, FileSystemLoader
from pathlib import Path
import subprocess
import os
import numpy as np
import time


def clean_results():
    print("Cleaning previous output files")
    subprocess.run("./clean.sh", cwd="fluid-openfoam", stdout=subprocess.DEVNULL)
    subprocess.run("./clean.sh", cwd="solid-calculix", stdout=subprocess.DEVNULL)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution took {round(elapsed_time,2)} seconds to execute")
        return result
    return wrapper


def render(template_path, output_path, config_params):
    base_path = template_path.parent.absolute()
    env = Environment(loader=FileSystemLoader(base_path))
    template = env.get_template(template_path.name)

    with open(output_path, "w") as file:
        file.write(template.render(config_params))


def setup_simulation(config):
    print("Setting up simulation parameters", config)

    templates_fldr = Path("templates")
    fluid_fldr = Path("fluid-openfoam/system")
    solid_fldr = Path("solid-calculix")

    render(templates_fldr / "controlDict-template",   fluid_fldr / "controlDict", config)
    render(templates_fldr / "flap-template",          solid_fldr / "flap.inp",    config)
    render(templates_fldr / "preciceConfig-template", "precice-config.xml",       config)

@timing_decorator
def run_simulation(options, out_folder="out"):
    print("Running simulation...")

    with open("output_openfoam.log", "w") as outfile1:
        p1 = subprocess.Popen("./run.sh", cwd="fluid-openfoam", stdout=outfile1)

    with open("output_calculix.log", "w") as outfile2:
        p2 = subprocess.Popen(["./run.sh"], cwd="solid-calculix", stdout=outfile2)

    p1.wait()
    p2.wait()

    print("Simulation execution finished!")

    print("Postprocessing results...")
    subprocess.run(["./calculixToVTU.sh", out_folder], cwd="solid-calculix", stdout=subprocess.DEVNULL)
    print("Done!")


if __name__ == "__main__":
    timesteps = [
        # 0.1, 
        # 0.05, 0.04, 0.02, 0.01,
        # 0.005, 0.004, 0.002, 0.001
        0.0005, 0.0004, 0.0002, 0.0001
    ]

    for dt in timesteps:
        clean_results()

        n_samples = 10

        options = {
            'timestep': dt,
            'endTime': 2,
            'maxTime': 1,
            'writeInterval': round(1 / (n_samples * dt)),
        }

        setup_simulation(options)

        out_folder = f'E{options["endTime"]}-dt{options["timestep"]}'
        run_simulation(options, out_folder=out_folder)