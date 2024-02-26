from jinja2 import Environment, select_autoescape, FileSystemLoader
from pathlib import Path
import subprocess
import os
import numpy as np


def render(template_path, output_path, config_params):
    base_path = template_path.parent.absolute()
    env = Environment(loader=FileSystemLoader(base_path))
    template = env.get_template(template_path.name)

    with open(output_path, "w") as file:
        file.write(template.render(config_params))



def setup_simulation(config):
    print("Setting up simulation with parameters", config)
    templates_fldr = Path("templates")
    system_flder = Path(".")

    render(templates_fldr / "flap.inp", system_flder / "flap.inp", config)


def clean_results(runfile="clean.sh"):
    print("Cleaning previous output files")
    os.system(f'./{runfile}')


def run_simulation(config, output, runfile="run.sh"):
    setup_simulation(config)
    os.system(f'./{runfile} {output}')


def main():
    dts = [
        # 0.5, 0.4, 0.3, 0.2, 0.15, 0.1,
        # 0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
        # 0.005, 0.004, 0.003, 0.002, 0.0015, 0.001,
        # 0.0005, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001,
        # 0.00005, 0.00004, 0.00003, 0.00002, 0.000015, 0.00001,
    ]
    

    dts = [
        0.125, 0.6, 0.7, 0.8, 0.9,
        0.0125, 0.06, 0.07, 0.08, 0.09,
        0.00125, 0.006, 0.007, 0.008, 0.009,
        0.0006, 0.0007, 0.0008, 0.0009,
    ]


    for dt in dts:
        config = {
            "timestep": dt,
            "endtime": 1,
            "frequency": 1,
            "xForce": 0.5,
            "yForce": 0.0,
            "zForce": 0.0
        }

        foldername = f'E{config["endtime"]}-dt{config["timestep"]}'

        run_simulation(config, foldername)



if __name__ == "__main__":
    main()

