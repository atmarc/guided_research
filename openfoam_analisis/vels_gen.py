import numpy as np

# u(x,y,t) = -cos(x) * sin(y) * exp(-2vt)
# v(x,y,t) =  sin(x) * cos(y) * exp(-2vt)
# p(x,y,t) = -1/4(cos(2x) + cos(2y)) * exp(-4vt)

def write_init_vels(out_filename="0/init_vels", nu=0.1, v_0=1, N=100, L=2*np.pi):
    dx = L / N

    def u(x,y,t):
        return v_0 * -np.cos(x) * np.sin(y) * np.exp(-2*nu*t)


    def v(x,y,t):
        return v_0 * np.sin(x) * np.cos(y) * np.exp(-2*nu*t)


    def p(x,y,t):
        return -0.25 * (np.cos(2*x) + np.cos(2*y)) * np.exp(-4*nu*t)


    with open(out_filename, "w") as f:
        f.write("init_vels (\n")

        for x in np.arange(0 + dx/2, L, dx):
            for y in np.arange(0 + dx/2, L, dx):
                f.write(f'({u(x, y, 0)} {v(x, y, 0)} 0)\n')


        f.write(");\n")


if __name__ == "__main__":
    write_init_vels(N=100)