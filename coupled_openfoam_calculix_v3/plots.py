import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def load_vel_vtu(filename, N=100):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    vtk_dataset = reader.GetOutput()
    data_arrays = vtk_dataset.GetPointData()
    U_vtk = data_arrays.GetArray("U")

    U = vtk_to_numpy(U_vtk)
    return U


def get_sorted_files(base_path):
    list_files = os.listdir(base_path)
    
    if len(list_files) <= 1: return list_files

    def my_key(fn):
        num = fn.split('.')[1]
        left_zeros = len(num) - len(num.lstrip('0'))
        return float(num) * 10 ** (-left_zeros)

    return sorted(list_files, key=my_key)


def get_top_values(base_path):
    list_files = get_sorted_files(base_path)
    top_values = []
    for file in list_files:
        arr = load_vel_vtu(base_path / file)
        top_value = arr[491][0] # value at (0, 1, 0)
        top_values.append(top_value)

    return top_values


def plot_endtime():
    dts =  [
        # 0.1, 0.05, 
        0.04, 0.02, 0.01,
        0.005, 0.004, 0.002, 0.001,
        0.0005, 0.0004, 0.0002, 0.0001
    ]
    
    tend = 2
    top_values = []
    for dt in dts:
        base_path = Path(f'solid-calculix/output/E{tend}-dt{dt}/')
        last_file = get_sorted_files(base_path)[-1]
        arr = load_vel_vtu(base_path / last_file)
        top_value = arr[491][0] # value at (0, 1, 0)
        top_values.append(top_value)
        print(dt, top_value)

    

    diffs = [abs(y - top_values[-1]) for y in top_values]

    # plt.plot(dts, diffs)
    # plt.plot(dts, top_values, 'x')
    plt.plot(dts, diffs, '^', label="preCICEv3 Euler")

    # Convergence lines
    ai, bi = 2, -3
    ddt = (dts[ai]/dts[bi])
    m = 0
    # plt.plot([dts[ai], dts[bi]], [diffs[bi] * ddt + m,    diffs[bi] + m], '--')
    plt.plot([dts[ai], dts[bi]], [diffs[bi] * ddt**2 + m, diffs[bi] + m], '--')

    ####################################################################
    # Previous results
    dts_old =  [
        0.1, 
        0.05, 0.04, 0.02, 0.01,
        0.005, 0.004, 0.002, 0.001
    ]
    
    tend = 1
    top_values_v2 = []
    for dt in dts_old:
        base_path = Path(f'../coupled_openfoam_calculix_v2/solid-calculix/output/E{tend}-dt{dt}/')
        last_file = get_sorted_files(base_path)[-1]
        arr = load_vel_vtu(base_path / last_file)
        top_value = arr[491][0] # value at (0, 1, 0)
        top_values_v2.append(top_value)
        print(dt, top_value)

    diffs_old = [abs(y - top_values_v2[-1]) for y in top_values_v2]
    plt.plot(dts_old, diffs_old, 'o', label="preCICEv2")

    ####################################################################
    # Crank nickelson values
    
    dts_C =  [
        # 0.1, 0.05, 
        0.04, 0.02, 0.01,
        0.005, 0.004, 0.002, 0.001,
        0.0005, 
        # 0.0004, 
        0.0002, 0.0001
    ]
    
    tend = 1
    top_values_C = []
    for dt in dts_C:
        base_path = Path(f'solid-calculix/output/C0.9-E{tend}-dt{dt}/')
        last_file = get_sorted_files(base_path)[-1]
        arr = load_vel_vtu(base_path / last_file)
        top_value = arr[491][0] # value at (0, 1, 0)
        top_values_C.append(top_value)
        print(dt, top_value)

    

    diffs_C = [abs(y - top_values_C[-1]) for y in top_values_C]

    # plt.plot(dts_C, diffs)
    # plt.plot(dts_C, top_values, 'x')
    plt.plot(dts_C, diffs_C, 's', label="preCICEv3 C0.9")

    # Convergence lines
    ai, bi = 2, -1
    ddt = (dts_C[ai]/dts_C[bi])
    m = 0
    # plt.plot([dts_C[ai], dts_C[bi]], [diffs_C[bi] * ddt + m,    diffs_C[bi] + m], '--')
    # plt.plot([dts_C[ai], dts_C[bi]], [diffs[bi] * ddt**2 + m, diffs[bi] + m], '--')

    ####################################################################
    # General config

    plt.xscale("log")
    plt.yscale("log")
    # plt.xticks(dts,dts)
    plt.grid()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    plot_endtime()