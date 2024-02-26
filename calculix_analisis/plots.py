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


def plot_oscilation():
    # dts = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    dts = [
        # 0.5, 
        0.4, 0.3, 0.2, 0.15, 0.1,
        0.05, 0.04, 0.03, 0.02, 0.015, 0.01,
        0.005, 0.004, 0.003, 0.002, 0.0015, 0.001,
        # 0.0005, 0.0004, 0.0003, 0.0002, 0.00015, 0.0001,
        # 0.00005, 0.00004, 0.00003, 0.00002, 0.000015, 0.00001,
    ]

    tend = 1
    freq = 1
    for dt in dts:
        base_path = Path(f'output/E{tend}-dt{dt}/')
        print("Reading", base_path)
        top_values = get_top_values(base_path)
        X = np.arange(dt, tend+dt, dt * freq)
        plt.plot(X, top_values, label=dt)

    plt.legend()
    plt.show()


def plot_endtime():
    dts = [
        0.00005,0.00004, #0.00003, 0.00002, 0.000015, 0.00001,
        0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
        0.001, 0.00125, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
        0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,

        # 0.0001, 0.0002, 0.0004, 0.0005, 
        # 0.001, 0.002, 0.004, 0.005, 
        # 0.01, 0.02, 0.04, 0.05, 
        # 0.1, 0.2, 0.4, 0.5,
    ]

    tend = 1
    freq = 1
    top_values = np.zeros(len(dts), dtype=np.float64)
    for i, dt in enumerate(dts):
        base_path = Path(f'output/E{tend}-dt{dt}/')
        last_file = get_sorted_files(base_path)[-1]
        arr = load_vel_vtu(base_path / last_file)
        top_value = arr[491][0] # value at (0, 1, 0)
        top_values[i] = arr[491][0]
        print(dt, format(arr[491][0], '.15f'))

    

    diffs = [abs(y - top_values[0]) for y in top_values]

    # plt.plot(dts, diffs)
    plt.plot(dts, diffs, 'x')
    
    import pandas as pd
    df = pd.DataFrame(data={"timestep": dts, "displacement": top_value, "diff with dt=5e-5": diffs})
    df.to_csv("calculix-data.csv", index=False)

    # Convergence lines
    ai, bi = 10, -4
    ddt = (dts[ai]/dts[bi])
    m = 2e-6
    # plt.plot([dts[ai], dts[bi]], [diffs[bi] * ddt + m,    diffs[bi] + m], '--')
    plt.plot([dts[ai], dts[bi]], [diffs[bi] * ddt**2 + m, diffs[bi] + m], '--')

    plt.xscale("log")
    plt.yscale("log")
    # plt.xticks(dts,dts)
    plt.grid()
    # plt.legend()
    plt.show()



if __name__ == "__main__":
    # plot_oscilation()
    plot_endtime()