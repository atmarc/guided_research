import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import pandas as pd
import Ofpp


def load_vel_vtu(filename, N=100):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    vtk_dataset = reader.GetOutput()
    data_arrays = vtk_dataset.GetCellData()
    U_vtk = data_arrays.GetArray("U")

    U = vtk_to_numpy(U_vtk)
    U_grid = U.reshape(N, N, 3)
    return U_grid


def load_vel_vtm(filename: Path, N=100):
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(filename))
    reader.Update()

    output = reader.GetOutput()
    block = output.GetBlock(0) # Get first block "internal.vtu", where vel is
    data_arrays = block.GetCellData()
    U_vtk = data_arrays.GetArray("U")
    
    U = vtk_to_numpy(U_vtk)
    U_grid = U.reshape(N, N, 3)
    return U_grid


def get_vtm_files_from_series(folder, series_filename="time_stepping_review.vtm.series"):
    file_to_read = Path(folder).joinpath(series_filename)
    with open(file_to_read, 'r') as f:
        series = json.load(f)

    return [Path(folder).joinpath(file["name"]) for file in series["files"]]


def plot_image(mat):
    plt.imshow(mat)
    plt.colorbar()
    plt.show()


def taylor_green(N=100, L=2*np.pi, t=0, nu=0.1, v_0=1):
    def u(x,y,t):
        return v_0 * -np.cos(x) * np.sin(y) * np.exp(-2*nu*t)

    def v(x,y,t):
        return v_0 * np.sin(x) * np.cos(y) * np.exp(-2*nu*t)

    dx = L / N
    
    sol = np.zeros((N,N,2))

    for i, x in enumerate(np.arange(0 + dx/2, L, dx)):
        for j, y in enumerate(np.arange(0 + dx/2, L, dx)):
            sol[i, j, 0] = u(x, y, t)
            sol[i, j, 1] = v(x, y, t)

    return sol


def velocity_mag(vels):
    return np.sqrt(np.sum(np.square(vels), axis=2))


def plots():
    pass
    # Plot of different values for different timesteps, at t=10s 
    # for x, y in [(50,50)]:
    #     pos_x_y = [vels[x][y] for vels in magnitudes]
    #     sol_x_y = sol_mag[x][y]
    #     # sol_x_y = magnitudes[-1][x][y]
        
    #     fig, ax = plt.subplots()
    #     ax.set_title(f'Value at position ({x},{y})')
    #     ax.plot(dt, abs(pos_x_y - sol_x_y), 'x', label="Simulated values")
    #     # ax.plot([dt[0], dt[-1]], [sol_x_y, sol_x_y], label=f'Analytical solution: {round(sol_x_y, 4)}')

    #     error_0005 = abs(magnitudes[0][x][y] - sol_x_y)
    #     ax.plot([dt[0], dt[-1]], [error_0005, dt[-1]/dt[0] * error_0005], "--")
    #     ax.plot([dt[0], dt[-1]], [error_0005, (dt[-1]/dt[0])**2 * error_0005], "--")
        
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.set_xticks(dt)
    #     plt.legend()
    #     plt.grid()
    #     plt.show()


    # Plot of RMSE of every case, at t=10s
    # MSEs = [np.sqrt(np.mean(np.square(abs(sol_mag - vels)))) for vels in magnitudes]
    # fig, ax = plt.subplots()
    # ax.set_title(f'Root mean square errors')
    # ax.plot(dt, MSEs, 'x', label="mse")
    # ax.set_xticks(dt)
    # ax.set_xscale('log')
    # plt.show()

    # for t, error in zip(dt, MSEs):
    #     print(f'{t}:\t{error}')


def load_vels_N100():
    dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    base_path = Path("output")

    velocities_N100_data = []
    for dt in dts:
        folder_name = base_path.joinpath(f'dt{dt}_N100_E10_C1')
        vtm_files = get_vtm_files_from_series(folder_name)
        last_timestep = load_vel_vtm(vtm_files[-1], N=100)
        velocities_N100_data.append(last_timestep)

    return velocities_N100_data


def load_vels_N100_C09():
    base_path = Path("output")
    dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    velocities_N100_C09 = []

    for dt in dts:
        folder_name = base_path.joinpath(f'dt{dt}_N100_E10_C0.9')
        vtm_files = get_vtm_files_from_series(folder_name)
        last_timestep = load_vel_vtm(vtm_files[-1], N=100)
        velocities_N100_C09.append(last_timestep)

    return velocities_N100_C09


def load_vels_N100_E5():
    base_path = Path("output")
    dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    velocities_N100_E5 = []

    for dt in dts:
        folder_name = base_path.joinpath(f'dt{dt}_N100_E5_C1')
        vtm_files = get_vtm_files_from_series(folder_name)
        last_timestep = load_vel_vtm(vtm_files[-1], N=100)
        velocities_N100_E5.append(last_timestep)

    return velocities_N100_E5


def load_vels_N150():
    base_path = Path("output")
    velocities_N150_data = []
    dts = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    for dt in dts:
        folder_name = base_path.joinpath(f'dt{dt}_N150_E10_C1')
        vtm_files = get_vtm_files_from_series(folder_name)
        last_timestep = load_vel_vtm(vtm_files[-1], N=150)
        velocities_N150_data.append(last_timestep)

    return velocities_N150_data


def load_vels_N100_high_res():
    base_path = Path("output/high_tol")
    velocities_N150_data = []
    dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for dt in dts:
        folder_name = base_path / f'dt{dt}_N100_E10_C1'
        vtm_files = get_vtm_files_from_series(folder_name)
        last_timestep = load_vel_vtm(vtm_files[-1], N=100)
        velocities_N150_data.append(last_timestep)

    return velocities_N150_data


def plot_compare_errors_N100_N150():
    # Reading last values at t=10s from the simulations 
    velocities_N100_data = load_vels_N100()
    velocities_N100_C09 = load_vels_N100_C09()
    velocities_N100_E5 = load_vels_N100_E5()
    velocities_N150_data = load_vels_N150()
    velocities_N100_high_res = load_vels_N100_high_res()

    dts = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    n100_dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Get velocity magnitudes
    magnitudes_N100 = list(map(velocity_mag, velocities_N100_data))
    magnitudes_N100_C09 = list(map(velocity_mag, velocities_N100_C09))
    magnitudes_N100_E5 = list(map(velocity_mag, velocities_N100_E5))
    magnitudes_N150 = list(map(velocity_mag, velocities_N150_data))
    
    magnitudes_N100_high_res = list(map(velocity_mag, velocities_N100_high_res))

    # Reference analytical solution
    analytical_sol_N100 = taylor_green(t=10)
    sol_mag_N100 = velocity_mag(analytical_sol_N100)

    analytical_sol_N100_E5 = taylor_green(t=5)
    sol_mag_N100_E5 = velocity_mag(analytical_sol_N100_E5)

    analytical_sol_N150 = taylor_green(t=10, N=150)
    sol_mag_N150 = velocity_mag(analytical_sol_N150)

    # Plotting values at a given point for each case
    x, y = (50, 50)
    err_x_y_N100     = [abs(vels[x][y] - magnitudes_N100[0][x][y]) for vels in magnitudes_N100]
    err_x_y_N100_C09 = [abs(vels[x][y] - magnitudes_N100_C09[0][x][y]) for vels in magnitudes_N100_C09]
    err_x_y_N100_E5  = [abs(vels[x][y] - sol_mag_N100_E5[x][y]) for vels in magnitudes_N100_E5]
    err_x_y_N150     = [abs(vels[75][75] - magnitudes_N150[0][75][75]) for vels in magnitudes_N150]
    err_x_y_N100_high_res = [abs(vels[x][y] - magnitudes_N100_high_res[0][x][y]) for vels in magnitudes_N100_high_res]
    
    # sol_x_y = magnitudes[-1][x][y]
    markers = iter(['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H'])
    fig, ax = plt.subplots()
    ax.plot(n100_dts, err_x_y_N100, next(markers), label="N100 C1")
    ax.plot(n100_dts, err_x_y_N100_C09, next(markers), label="N100 C0.9")
    # ax.plot(n100_dts, err_x_y_N100_E5, next(markers), label="N100 E5")
    ax.plot(dts, err_x_y_N150, next(markers), label="N150 C1")
    ax.plot(n100_dts, err_x_y_N100_high_res, next(markers), label="N100 high")
    

    # error_0005 = abs(magnitudes_N150[0][x][y] - sol_x_y)
    ax.plot([n100_dts[0], n100_dts[-1]], [err_x_y_N100[-1] * n100_dts[0]/n100_dts[-1], err_x_y_N100[-1]], "--")
    ax.plot([n100_dts[0], n100_dts[-1]], [err_x_y_N100[-1] * (n100_dts[0]/n100_dts[-1])**2, err_x_y_N100[-1]], "--")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(n100_dts)
    plt.xlabel("Time step size")
    plt.ylabel("absolute error")
    plt.legend()
    plt.title("Absolute error compared to analytical solution")
    plt.grid()
    plt.show()


def plot_MSEs():
    # Reading last values at t=10s from the simulations 
    velocities_N100_data = load_vels_N100()
    velocities_N100_C09 = load_vels_N100_C09()
    velocities_N150_data = load_vels_N150()

    dts = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    n100_dts = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Get velocity magnitudes
    magnitudes_N100 = list(map(velocity_mag, velocities_N100_data))
    magnitudes_N100_C09 = list(map(velocity_mag, velocities_N100_C09))
    magnitudes_N150 = list(map(velocity_mag, velocities_N150_data))

    # Reference analytical solution
    analytical_sol_N100 = taylor_green(t=10, N=100)
    sol_mag_N100 = velocity_mag(analytical_sol_N100)

    analytical_sol_N150 = taylor_green(t=10, N=150)
    sol_mag_N150 = velocity_mag(analytical_sol_N150)

    # Compute MSE
    MSE_N100     = [np.mean(np.square(vels - sol_mag_N100)) for vels in magnitudes_N100]
    MSE_N100_C09 = [np.mean(np.square(vels - sol_mag_N100)) for vels in magnitudes_N100_C09]
    MSE_N150     = [np.mean(np.square(vels - sol_mag_N150)) for vels in magnitudes_N150]

    # Plot values
    fig, ax = plt.subplots()

    ax.plot(n100_dts, MSE_N100, 'x', label="N100 C1")
    ax.plot(n100_dts, MSE_N100_C09, 'x', label="N100 C0.9")
    ax.plot(dts, MSE_N150, 'x', label="N150 C1")

    ax.set_title(f'MSE')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(n100_dts)
    plt.xlabel("Time step size")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.show()


def load_values(dts, N, C, base_path):
    velocities = []
    for dt in dts:
        file_name = Path(base_path) / f'dt{dt}_N{N}_E1_C{C}'
        U = Ofpp.parse_internal_field(str(file_name))
        velocities.append(U.reshape(N, N, 3))

    return velocities


def plot_abs_error_N50():
    base_path = Path("output")
    dts = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    velocities_C1 = load_values(dts, 50, 1, base_path / "Uvels_N50")
    velocities_C0 = load_values(dts, 50, 0, base_path / "Uvels_N50_C0")
    velocities_N100_C1 = load_values(dts[3:], 100, 1, base_path / "Uvels_N100_C1")
    velocities_N100_C0 = load_values(dts[3:], 100, 0, base_path / "Uvels_N100_C0")
    velocities_N150_C1 = load_values(dts[3:], 150, 1, base_path / "Uvels_N150_C1")
    
    magnitudes_C1 = list(map(velocity_mag, velocities_C1))
    magnitudes_C0 = list(map(velocity_mag, velocities_C0))
    magnitudes_N100_C1 = list(map(velocity_mag, velocities_N100_C1))
    magnitudes_N100_C0 = list(map(velocity_mag, velocities_N100_C0))
    magnitudes_N150_C1 = list(map(velocity_mag, velocities_N150_C1))

    # analytical_sol = taylor_green(t=1, N=50)
    # sol_mag = velocity_mag(analytical_sol)

    x, y = (25, 25)
    ref_sol = magnitudes_C1[REF_ID][x][y]
    err_x_y_C1 = [abs(vels[x][y] - ref_sol) for vels in magnitudes_C1]
    err_x_y_C0 = [abs(vels[x][y] - ref_sol) for vels in magnitudes_C0]
    err_x_y_N100_C1 = [abs(vels[50][50] - magnitudes_N100_C1[REF_ID][50][50]) for vels in magnitudes_N100_C1]
    err_x_y_N100_C0 = [abs(vels[50][50] - magnitudes_N100_C1[REF_ID][50][50]) for vels in magnitudes_N100_C0]
    err_x_y_N150_C1 = [abs(vels[75][75] - magnitudes_N150_C1[REF_ID][75][75]) for vels in magnitudes_N150_C1]

    fig, ax = plt.subplots()
    markers = iter(['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H'])

    ax.plot(dts[3:], err_x_y_C1[3:], next(markers), label="N50", color="orange")
    # ax.plot(dts, err_x_y_C0, next(markers), label="N50 C0")
    ax.plot(dts[3:], err_x_y_N100_C1, next(markers), label="N100", color="blue")
    # ax.plot(dts[3:], err_x_y_N100_C0, next(markers), label="N100 C0", color="orange")
    ax.plot(dts[3:], err_x_y_N150_C1, next(markers), label="N150", color="green")

    # First and second order lines
    # ax.plot([dts[3], dts[-1]], [err_x_y_N100_C0[0], err_x_y_N100_C0[0] * dts[-1]/dts[3]], "--", color="orange")
    ax.plot([dts[3], dts[-1]], [err_x_y_N100_C1[-1] * (dts[3]/dts[-1])**2, err_x_y_N100_C1[-1]], "--", color="grey", label="2nd ord")
    # ax.plot([dts[3], dts[-1]], [err_x_y_N150_C1[-1] * (dts[3]/dts[-1])**2, err_x_y_N150_C1[-1]], "--", color="green")

    # Plot parameters 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(dts[3:])
    plt.xlabel("$\\Delta t$")
    plt.ylabel("$|\\tilde{u} - u|$")
    plt.legend()
    plt.title("Convergence analysis of timestep size")
    plt.grid()
    plt.show()

    return {
        "dt": dts,
        "diff-N50": err_x_y_C1,
        "diff-N100": err_x_y_N100_C1 + ['-', '-', '-'],
        "diff-N150": err_x_y_N150_C1 + ['-', '-', '-'],
    }

def plot_MSEs_N50():
    base_path = Path("output")
    dts = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    velocities_C1 = load_values(dts, 50, 1, base_path / "Uvels_N50")
    velocities_C0 = load_values(dts, 50, 0, base_path / "Uvels_N50_C0")
    velocities_N100_C1 = load_values(dts[3:], 100, 1, base_path / "Uvels_N100_C1")
    velocities_N100_C0 = load_values(dts[3:], 100, 0, base_path / "Uvels_N100_C0")
    velocities_N150_C1 = load_values(dts[3:], 150, 1, base_path / "Uvels_N150_C1")

    magnitudes_C1 = list(map(velocity_mag, velocities_C1))
    magnitudes_C0 = list(map(velocity_mag, velocities_C0))
    magnitudes_N100_C1 = list(map(velocity_mag, velocities_N100_C1))
    magnitudes_N100_C0 = list(map(velocity_mag, velocities_N100_C0))
    magnitudes_N150_C1 = list(map(velocity_mag, velocities_N150_C1))

    analytical_sol = taylor_green(t=1, N=50)
    sol_mag = velocity_mag(analytical_sol)

    analytical_sol_N100 = taylor_green(t=1, N=100)
    sol_mag_N100 = velocity_mag(analytical_sol_N100)
    
    analytical_sol_N150 = taylor_green(t=1, N=150)
    sol_mag_N150 = velocity_mag(analytical_sol_N150)

    # ref_sol = magnitudes_C1[REF_ID]
    ref_sol = sol_mag
    MSE_C1 = [np.sqrt(np.mean(np.square(vels - ref_sol))) for vels in magnitudes_C1]
    MSE_C0 = [np.sqrt(np.mean(np.square(vels - ref_sol))) for vels in magnitudes_C0]
    MSE_N100_C1 = [np.sqrt(np.mean(np.square(vels - sol_mag_N100))) for vels in magnitudes_N100_C1]
    MSE_N100_C0 = [np.sqrt(np.mean(np.square(vels - sol_mag_N100))) for vels in magnitudes_N100_C0]
    MSE_N150_C1 = [np.sqrt(np.mean(np.square(vels - sol_mag_N150))) for vels in magnitudes_N150_C1]

    # MSE_C1 = [np.sqrt(np.mean(np.square(vels - sol_mag))) for vels in magnitudes_C1]
    # MSE_C0 = [np.sqrt(np.mean(np.square(vels - sol_mag))) for vels in magnitudes_C0]

    fig, ax = plt.subplots()
    markers = iter(['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H'])

    # First and second order lines
    dt_diff = dts[9]/dts[-1]
    ax.plot([dts[9], dts[-1]], [MSE_C1[-1] * dt_diff**2, MSE_C1[-1]], "--", color="grey")

    ax.plot([dts[0], dts[-1]], [MSE_C1[3], MSE_C1[3]], "--", color="orange")
    ax.plot([dts[0], dts[-1]], [MSE_N100_C1[0], MSE_N100_C1[0]], "--", color="blue")
    ax.plot([dts[0], dts[-1]], [MSE_N150_C1[0], MSE_N150_C1[0]], "--", color="green")


    ax.plot(dts, MSE_C1, next(markers), label="N50", color="orange")
    # ax.plot(dts, MSE_C0, next(markers), label="N50 C0")
    ax.plot(dts[3:], MSE_N100_C1, next(markers), label="N100", color="blue")
    # ax.plot(dts[3:], MSE_N100_C0, next(markers), label="N100 C0")
    ax.plot(dts[3:], MSE_N150_C1, next(markers), label="N150", color="green")


    # Plot parameters 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(dts)
    plt.xlabel("$\Delta t$")
    plt.ylabel("$RMSE$")
    plt.legend()
    plt.title("RMSE comparing to analytical solution")
    plt.grid()
    plt.show()

    return {
        "rmse-N50": MSE_C1,
        "rmse-N100": MSE_N100_C1 + ['-', '-', '-'],
        "rmse-N150": MSE_N150_C1 + ['-', '-', '-'],
    }


REF_ID = 0

if __name__ == "__main__":
    data1 = plot_abs_error_N50()
    data2 = plot_MSEs_N50()
    data = {**data1, **data2}
    
    print(data)

    df = pd.DataFrame(data=data)
    df.to_csv("openfoam-data.csv", index=False)

    # plot_abs_error_N50()
    # plot_compare_errors_N100_N150()