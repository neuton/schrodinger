#!/usr/bin/env python3
import numpy as np
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

box_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
options = [arg for arg in sys.argv[1:] if arg.startswith('-')]

box = list(map(int, box_args))
ndim = len(box)
filename = ''

dx, dt = 5e-3, 4/np.pi * 1e-5
std = 10 * dx # dispersion of psi(t=0)
momentum = 2*np.pi / dx * .2

print_help = lambda: print('Usage: ' + sys.argv[0] +
    ' [<box_height>] <box_width> [-1slit|-2slits|-barrier] [-tunneling] [-oscillatory] [-ellipse]' +
    ' [-dispersion=<particle_std>] [-momentum=<particle_momentum>] [-save=<filename>]')

if ndim == 0 or ndim > 2: print_help(); exit(1)

grid = np.array(np.meshgrid(*(np.linspace(-n*dx/2, n*dx/2, n) for n in reversed(box))))
walls, potential = np.zeros_like(grid[0]), np.zeros_like(grid[0])

abs2 = lambda psi: psi.real**2 + psi.imag**2
normed = lambda psi: psi / (abs2(psi).sum() * dx**ndim) ** .5
gaussian = lambda grid, shift=np.zeros_like(box), std=dx: normed(
    np.exp(-np.sum((grid - np.array(shift[:ndim])[(slice(None),)+(None,)*ndim])**2, axis=0) / (2 * std**2)))

kinetic_term = lambda psi: -.5 / dx**2 * laplace(psi, mode='constant') # fixed boundary condition
hamiltonian_term = lambda psi: kinetic_term(psi) + potential * psi
avr_energy = lambda psi: (hamiltonian_term(psi).ravel() @ psi.conjugate().ravel() * dx**ndim).real
dpsi_dt = lambda t, psi: -1j * hamiltonian_term(psi) * np.logical_not(walls) # Schroedinger equation

dpsi_dt_raveled = lambda t, psi_raveled: dpsi_dt(t, psi_raveled.reshape(box)).ravel()
solve_forward = lambda t, psi: solve_ivp(dpsi_dt_raveled, (0, t), psi.ravel(), method='DOP853').y[..., -1].reshape(box)

for option in options:
    if option.startswith('-1'): # 1 slit
        if ndim < 2: print('Must be 2D.'); exit(1)
        h, w = box; a, c = 2, 4
        walls[:, w//2 - a : w//2 + a] = 1; walls[h//2 - c//2 : h//2 + c//2, :] = 0

    elif option.startswith('-2'): # 2 slits
        if ndim < 2: print('Must be 2D.'); exit(1)
        h, w = box; a, b, c = 2, 8, 4
        walls[:, w//2 - a : w//2 + a] = 1;
        walls[h//2 - b - c : h//2 - b, :] = 0; walls[h//2 + b : h//2 + b + c, :] = 0

    elif option.startswith('-b'): # barrier
        if ndim < 2: print('Must be 2D.'); exit(1)
        h, w = box; a, b = 2, 8
        walls[h//2 - b : h//2 + b, w//2 - a : w//2 + a] = 1

    elif option.startswith('-e'): # elliptic walls
        walls[np.sum((grid * np.array(box)[:,None,None])**2, axis=0) > (np.prod(box)**(2/ndim) * dx / 2)**2] = 1

    elif option.startswith('-d'): # initial state dispersion
        std = float(option.split('=')[1]) * dx

    elif option.startswith('-m'): # initial state momentum
        momentum = 2*np.pi * float(option.split('=')[1]) / 2 / dx

    elif option.startswith('-s'): # save animation to file
        try:
            filename = option.split('=')[1]
        except IndexError:
            print_help(); exit(1)

# initial wave function at t=0:
psi = gaussian(grid, shift=[-box[-1]/4*dx, -box[0]/50*dx], std=std) * np.exp(1j * grid[0] * momentum)
psi *= np.logical_not(walls)

for option in options:
    if option.startswith('-t'): # potential tunneling
        # potential += gaussian(grid[0]) * avr_energy(psi) / np.max(gaussian(grid[0])) * 1.1
        potential += (np.abs(grid[0]) < dx * 2) * avr_energy(psi) * 1.1

    elif option.startswith('-o'): # elliptic oscillatory potential
        potential += 5e5 * avr_energy(psi) * np.sum(
            (grid / np.array(list(reversed(box)))[(slice(None),)+(None,)*ndim])**2, axis=0)

fig, ax = plt.subplots()

if ndim == 1:
    plot, = ax.plot(grid[0], abs2(psi))
    if np.any(potential):
        ax.plot(grid[0], potential*5e-4, color='black')
    # # compare to exact solution (requires box == [200]):
    # from wave_vs_schrodinger import Schrodinger
    # box = [200]; walls[0] = walls[-1] = 1
    # exact_psi = Schrodinger(lambda x: psi[grid[0] == x - .5], n=200, nx_C_int=200-1, T=4/np.pi)
    # plot2, = ax.plot(grid[0], abs2(np.array([exact_psi(x, 0) for x in grid[0] + .5])))
elif ndim == 2:
    plot = ax.imshow(abs2(psi), vmin=0, vmax=np.max(abs2(psi)) * .3, cmap='magma', interpolation='bilinear')
    if np.any(walls) or np.any(potential):
        ax.imshow(np.ones_like(grid[0]), cmap='gray', vmin=0, vmax=1,
            alpha=np.max([np.array(walls, dtype=float), 1 - np.exp(-2e-5 * potential)], axis=0))

fig.tight_layout()

def update(step):
    psi[...] = normed(solve_forward(dt, psi))
    if ndim == 1:
        plot.set_data([grid[0], abs2(psi)])
        # plot2.set_data([grid[0], abs2(np.array([exact_psi(x, step*dt) for x in grid[0] + .5]))])
    elif ndim == 2:
        plot.set_data(abs2(psi))

anim = FuncAnimation(fig, update, interval=10, save_count=0, frames=240 if filename else None)

if filename:
    anim.save(filename, fps=30)
else:
    plt.show()
