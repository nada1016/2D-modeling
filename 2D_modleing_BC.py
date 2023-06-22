import numpy as np
import matplotlib.pyplot as plt

# padding
def padding(vp, mnx, mnz, abs_thick, order, USE_PML_ZMAX):
    if USE_PML_ZMAX == 1:
        ivp = np.zeros((mnz + abs_thick * 2, mnx + abs_thick * 2))
        ivp[abs_thick:abs_thick + mnz, abs_thick:abs_thick + mnx] = vp

        for ii in range(abs_thick):
            ivp[ii, :] = ivp[abs_thick, :]
            ivp[abs_thick + mnz + ii, :] = ivp[abs_thick + mnz, :]
            ivp[:, ii] = ivp[:, abs_thick]
            ivp[:, abs_thick + mnx + ii] = ivp[:, abs_thick + mnx]

    else:
        ivp = np.zeros((mnz + abs_thick + order, mnx + abs_thick * 2))
        ivp[order:order + mnz, abs_thick:abs_thick + mnx] = vp

        for ii in range(order):
            ivp[ii, :] = ivp[order, :]

        for ii in range(abs_thick):
            ivp[:, ii] = ivp[:, abs_thick]
            ivp[:, abs_thick + mnx + ii] = ivp[:, abs_thick + mnx]

        for ii in range(abs_thick):
            ivp[mnz + ii + order, :] = ivp[order + mnz, :]

    return ivp

# Load data from "marmousi_cut.mat" file
mnz = 200
mnx = 200
vp = np.ones((mnz, mnx)) * 1500

abs_thick = 50
abs_rate = 0.20 / abs_thick

order = 2
iabs_thick = abs_thick + order
nx = 2 * iabs_thick + mnx
nz = iabs_thick + mnz

dx = 1 
dt = 0.00001
dz = dx  
time = 0.01  # sec
nt = round(time / dt) + 1
t = np.arange(1,dt,dt*nt)

rec_x = np.array([150 + iabs_thick])
rec_z = np.array([2 + order])

shot_x = np.array([50 + iabs_thick])
shot_z = np.array([2 + order])
nshot = len(shot_x)

nrec = len(rec_x)

USE_PML_ZMAX = False  # false == 0, true == 1

vp_padding = padding(vp.T, mnx, mnz, iabs_thick, order, USE_PML_ZMAX)
T_vp = vp_padding.T

# Source
f0 = 20.0  # dominant frequency of the wavelet
t0 = 1.20 / f0  # excitation time
factor = 1e10  # amplitude coefficient
angle_force = 90.0  # spatial orientation
t = np.arange(nt) * dt
a = np.pi ** 2 * f0 ** 2
dt2rho_src = dt ** 2

fmax = f0 * 3
source_term = factor * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)

force_x = np.sin(angle_force * np.pi / 180) * source_term * dt2rho_src / (dx * dx)
force_x = force_x / np.max(np.abs(force_x))

xx = np.arange(1 + iabs_thick, mnx + iabs_thick)
zz = np.arange(1 + order, mnz + order)

u1 = np.zeros((nx, nz))
u2 = np.zeros((nx, nz))
u3 = np.zeros((nx, nz))

u3_1 = np.zeros((mnx, mnz))
u3_2 = np.zeros((mnx, mnz))
u3_3 = np.zeros((mnx, mnz))

trace_1 = np.zeros((nt, 1))
trace_2 = np.zeros((nt, 1))
trace_3 = np.zeros((nt, 1))

shot = np.zeros((nx, nz))
shot[shot_x, shot_z] = 1

u1 = np.zeros((nx, nz))
u2 = np.zeros((nx, nz))
u3 = np.zeros((nx, nz))

vir_s = np.zeros((mnx, mnz))
for it in range(nt):
    if it % 100 == 1:
        print('Time step =', it)

    for ix in range(1, nx - 1):
        for iz in range(1, nz - 1):
            u3[ix, iz] = 2 * u2[ix, iz] - u1[ix, iz] + (T_vp[ix, iz] ** 2) * (dt ** 2) + force_x[it] * shot[ix, iz]
            vir_s[ix, iz] = 2 / vp[ix, iz] ** 3 * (u3[ix, iz] - 2 * u2[ix, iz] + u1[ix, iz]) / dx ** 2

    if it % 100 == 1:
        plt.figure(1)
        plt.imshow(u3[iabs_thick:mnx + iabs_thick, order:mnz + order].T, cmap='gray', origin='lower')
        plt.colorbar()
        #plt.clim(-10e-2, 10e-2)
        plt.show()

    trace_1[it, 0] = u3[rec_x, rec_z]

del u1, u2, u3

# Jacobian check!
dv = 1
vp2 = np.copy(vp)
vp2[100, 100] += dv

vp_padding2 = padding(vp2.T, mnx, mnz, iabs_thick, order, USE_PML_ZMAX)
T_vp2 = vp_padding2.T

u1 = np.zeros((nx, nz))
u2 = np.zeros((nx, nz))
u3 = np.zeros((nx, nz))

for it in range(nt):
    if it % 100 == 1:
        print('Time step =', it)

    for ix in range(1, nx - 1):
        for iz in range(1, nz - 1):
            u3[ix, iz] = 2 * u2[ix, iz] - u1[ix, iz] + (T_vp2[ix, iz] ** 2) * (dt ** 2) + force_x[it] * shot[ix, iz]
    trace_2[it, 0] = u3[rec_x, rec_z]

del u1, u2, u3

J1 = (trace_2[100, 100] - trace_1[100, 100]) / dv

# Virtual source
shot_x = np.array([100 + iabs_thick])
shot_z = np.array([100 + order])
shot = np.zeros((nx, nz))
shot[shot_x, shot_z] = 1
force_x = vir_s[shot_x, shot_z]

u1 = np.zeros((nx, nz))
u2 = np.zeros((nx, nz))
u3 = np.zeros((nx, nz))

for it in range(nt):
    if it % 100 == 1:
        print('Time step =', it)

    for ix in range(1, nx - 1):
        for iz in range(1, nz - 1):
            u3[ix, iz] = 2 * u2[ix, iz] - u1[ix, iz] + (T_vp[ix, iz] ** 2) * (dt ** 2) + force_x * shot[ix, iz]
    trace_3[it, 0] = u3[rec_x, rec_z]

del u1, u2, u3

J2 = trace_3

aa = J2 - J1
plt.plot(t, J1, label='J1')
plt.plot(t, J2, label='J2')
plt.plot(t, aa, label='diff')
plt.legend()
plt.show()