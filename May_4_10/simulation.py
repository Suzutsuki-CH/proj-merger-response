import numpy as np
import matplotlib.pyplot as plt

def rk4_second_order(f, T0, R0, V0, M, dh):
    """
    Implementation of the fourth-order Runge-Kutta method
    to solve a second-order ordinary differential equation.

    Args:
    - f: Function representing the second-order ODE d^2y/dt^2 = f(t, y, z)
    - T0: Initial time
    - R0: Initial position of two halos
    - V0: Initial velocities of two halos
    - dh: Step size

    Returns:
    - t: time at one step after the initial time
    - y: position at time t + dh
    - v: velocity at time t + dh
    """

    t = T0

    # Initial Conditions of Dark Matter Halos

    r = R0.copy()
    v = V0.copy()

    # k1 coefficients
    k1_r = dh * v
    k1_v = dh * f(t, r, M)

    # k2 coefficients
    k2_r = dh * (v + k1_v/2)
    k2_v = dh * f(t + dh/2, r + k1_r/2, M)


    k3_r = dh * (v + k2_v/2)
    k3_v= dh * f(t + dh/2, r + k2_r/2, M)


    k4_r = dh * (v + k3_v)
    k4_v = dh * f(t + dh, r + k3_r,  M)


    r += dh * (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
    v += dh * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    
    t += dh

    # print(t, r)

    return [t, r, v]


# Setting up initial conditions
              # x     y      z
v0 = np.array([1.,    1.,     1., 
               -1.,  -1.,     -1.])

r0 = np.array([-1.,   -0.4,     -1., 
                1.,    0.4,      1.])

G = 0.65
M = [3, 5]

def f(T, R0, M):

    r1 = np.array(R0[:3])
    r2 = np.array(R0[3:])

    dr_vec = r1-r2
    dr = np.sum((r1-r2)**2)

    fx1 = -G * M[1] / (dr ** 3) * dr_vec[0]
    fy1 = -G * M[1] / (dr ** 3) * dr_vec[1]
    fz1 = -G * M[1] / (dr ** 3) * dr_vec[2]

    fx2 = G * M[0] / (dr ** 3) * dr_vec[0]
    fy2 = G * M[0] / (dr ** 3) * dr_vec[1]
    fz2 = G * M[0] / (dr ** 3) * dr_vec[2]

    F = np.array([fx1, fy1, fz1, fx2, fy2, fz2])

    return F

t0 = 0
dh = 0.005
num_steps = 100000  # number of steps

T = [t0]
R = [r0]
V = [v0]

t0 = 0

for i in range(0, num_steps):
    t0, r0, v0 = rk4_second_order(f, t0, r0, v0, M, dh)
    
    T.append(t0)
    R.append(r0)
    V.append(v0)

x1 = [RR[0] for RR in R]
x2 = [RR[3] for RR in R]

y1 = [RR[1] for RR in R]
y2 = [RR[4] for RR in R]

z1 = [RR[2] for RR in R]
z2 = [RR[5] for RR in R]


ax = plt.figure().add_subplot(projection='3d')
ax.plot(x1,y1,z1, label = r"Halo 1, $M = 3$")
ax.plot(x2,y2,z2, label = r"Halo 2, $M = 5$")
ax.scatter(x1[0], y1[0], z1[0], marker="o", color="black", label="initial Positions")
ax.scatter(x2[0], y2[0], z2[0], marker="o", color="black")
ax.legend()
plt.show()
