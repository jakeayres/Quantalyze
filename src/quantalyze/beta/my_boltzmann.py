import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import time
from drude import simple_drude
from Tl2201 import compute_fermi_surface_tl2201, modify_vf_for_Bi2201
from model import compute_fermi_surface_model
from LSCO import tight_LSCO
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from scipy.optimize import curve_fit

###############
# Global
###############

E = 1.602176487e-19
HBAR = 1.054571628e-34
ME = 9.10938215e-31
KB = 1.38E-23

A = 3.777e-10  # From Cava1987, table 2/3
C = 13.226e-10 / 2


# Compiling is the default, my benchmark show a 150x speedup
# Not compiling can be useful if errors occur or new code is added.
COMPILE = True
if not COMPILE:
    def njit():
        def wrapper(func):
            return func
        return wrapper

###############
# Part 1
###############


#@njit()
def precompute(nr, *fsargs):
    """ Load all angles, kx, ky, vx, vy, tau """

    assert(nr % 2 == 0)
    pphi = np.linspace(0, 2 * np.pi * (nr - 1) / nr, nr)
    r = compute_FS(pphi, *fsargs)
    return pphi, r[0], r[1], r[2], r[3], r[4]


#@njit()
def stepanalysis(pphi, *fsargs):
    """ Return this sigma component for the given B values.
    Also returns an array matching pphi with omegac values at B=1
    """

    # Determine exps [survival odds to go from each point to the next at B=1]
    # and tsteps [time it takes from one point to the next at B=1]
    N = len(pphi)
    exps = np.zeros(N)
    tsteps = np.zeros(N)
    for i, phi in enumerate(pphi):
        if i < N - 1:
            next_phi = pphi[i + 1]
        else:
            next_phi = pphi[0] + 2 * np.pi  # should just be 2pi
        tsteps[i], exps[i] = get_w_and_dt(phi, next_phi, *fsargs)

    # Just to make sure this is reasonable. Usually 1-100 ps or so.
    circulation_time = np.sum(tsteps)
    # ROEMERS VALUES
    assert(1e-20 < circulation_time < 1e-8) 

    omegac = np.zeros(len(pphi))
    omegac[:-1] = (pphi[1:] - pphi[:-1]) / tsteps[:-1]
    omegac[-1] = (pphi[0] - pphi[-1] + 2 * np.pi) / tsteps[-1]
    return tsteps, exps, omegac


###############
# Part 2
###############


@njit()
def integrate_phi(pphi, vvx, vvy, kkf, ww, ddt, ttau, is_xx):
    """ Perform the phi integral and outsource the internal integrals. """

    # Even because element 0 is also the last element.
    assert(len(pphi) % 2 == 0 and "has to be *even* length for Simpson")
    assert(len(vvx) == len(pphi))
    assert(len(vvy) == len(pphi))
    assert(len(kkf) == len(pphi))
    assert(len(ww) == len(pphi))
    assert(len(ddt) == len(pphi))

    # Note that usually there is 1424242424242...41
    # Phi is cyclic, the first element is the last element,
    # so we simply visit everything once and let it be 242424242...4
    total = 0
    for i, phi in enumerate(pphi):
        value = integrate_time(i, pphi, vvx, vvy, kkf, ww, ddt, ttau, is_xx)
        if i == 0:
            factor = 2
        elif i % 2:  # odd indices
            factor = 4
        else:
            factor = 2
        total += value * factor

    assert(factor == 4)  # guarantee that we finish with 4 before we wrap back.
    dphi = pphi[1] - pphi[0]

    return total * dphi / 3


@njit()
def integrate_time(index, pphi, vvx, vvy, kkf, ww, ddt, ttau, is_xx):
    """ Get the time integral for the phi value at index. """

    v0 = vvx[index] if is_xx else -vvy[index]
    vf0 = np.sqrt(vvx[index]**2 + vvy[index]**2)
    if ww[index] < 1e-4:
        if is_xx:
            return v0 / vf0 * kkf[index] * vvx[index] * ttau[index]
        else:
            return 0

    N = len(pphi)
    survivalodds = 1
    position = index

    # Run down the orbit. Terminate on 2 conditions:
    #   1) The particle is dead. Negligible survival odds.
    #   2) We come full circle.
    # Use trapezoidal integration.
    integral = 0
    while survivalodds > 1e-8 and (position != index or integral == 0):
        # Previous position
        leftbound = vvx[position] * survivalodds

        # New position
        # Note that the ww and ddt array are 1 less in length,
        # [position] there dictates what happens between [position and position+1]
        # in the longer arrays of vx, vy, kx, ky. Hence ask for them
        # *before* incrementing.
        survivalodds *= ww[position]
        time_step = ddt[position]
        position = (position + 1) % N

        rightbound = vvx[position] * survivalodds
        integral += 0.5 * (leftbound + rightbound) * time_step

    # Relevant for when full orbits are completed
    integral /= 1 - survivalodds

    wc = (pphi[(index + 1) % N] - pphi[index]) / ddt[index]
    real = -ttau[index] / (1 + wc**2 * ttau[index]**2)
    real *= wc * ttau[index] * np.sin(pphi[index]) - np.cos(pphi[index])
    real *= np.sqrt(vvx[index]**2 + vvy[index]**2)

    # Velocity at time 0, determines if we do sigma_xx or sigma_xy
    # Really sigma_yx so we add the minus from Onsager's relations
    # using sigma_xy = -sigma_yx here right away to get that over with.
    integral *= v0 * kkf[index] / vf0
    return integral


#@njit()
def get_w_and_dt(phi1, phi2, *fsargs):
    """ Obtain the time t and survival oods exp(integral(-dt/tau)) """

    assert(0 < phi2 - phi1 < 0.1)

    ssubphi = np.linspace(phi1, phi2, 15)
    kkx, kky, vvx, vvy, ttau = compute_FS(ssubphi, *fsargs)

    timer = 0  # integral dt = dk/kdot
    exponent = 0  # integral dk/(kdot * tau)
    for index in range(len(ssubphi) - 1):
        dk = np.sqrt((kkx[index] - kkx[index + 1])**2 +
                     (kky[index] - kky[index + 1])**2)

        wc1 = E / HBAR * np.sqrt(vvx[index]**2 + vvy[index]**2)  # B=1
        wct1 = wc1 * ttau[index]
        wc2 = E / HBAR * np.sqrt(vvx[index + 1]**2 + vvy[index + 1]**2)
        wct2 = wc2 * ttau[index + 1]
        dt = dk * 0.5 * (1 / wc1 + 1 / wc2)

        timer += dt
        exponent += dk * 0.5 * (1 / wct1 + 1 / wct2)
    return timer, np.exp(-exponent)


@njit()
def prefactors():
    return E**2 / (2 * np.pi**2 * C * HBAR)


def sigma(Bvals, pphi, kkx, kky, vvx, vvy, ttau, exps, tsteps, is_xx):
    """ Compute sigma_xx or sigma_xy for all magnetic field values given. """

    kkf = np.sqrt(kkx**2 + kky**2)
    results = np.zeros(len(Bvals))
    st = time.time()
    for i, B in enumerate(Bvals):
        ww = exps ** (1 / B)
        ddt = tsteps / B

        results[i] = prefactors() * integrate_phi(pphi, vvx, vvy, kkf,
                                                  ww, ddt, ttau, is_xx)

        elapsed = time.time() - st
        print(f'Finished B={B:.3f}. '
              f'Busy for {elapsed:.3f} s, '
              f'expect another {elapsed * (len(Bvals) - i - 1) / (i + 1):.2f} s')

    return results


###############
# Execution
###############


def run_Drude_test():

    global compute_FS
    compute_FS = simple_drude
    compute_FS.recompile()

    KF = 0.75e10
    n = KF**2 / (2 * np.pi * C)
    print(f'Isotropic FS with p={n * A**2 * C:.2f} and mass 5m0.')

    # The entire computation is these few lines
    Bvals = np.linspace(0.001, 100, 131)
    pphi, kkx, kky, vvx, vvy, ttau = precompute(3000, KF, 5 * ME)
    ttsteps, eexps, wwc = stepanalysis(pphi, KF, 5 * ME)
    sxx = sigma(Bvals, pphi, kkx, kky, vvx,
                vvy, ttau, eexps, ttsteps, True)
    sxy = sigma(Bvals, pphi, kkx, kky, vvx, vvy,
                ttau, eexps, ttsteps, False)

    # Test that wc is isotropic and the right value
    assert(all(np.abs(wwc - E / 5 / ME) < 1e-6 * E / 5 / ME))

    # Test that sigmaxx and sigmaxy have the right values at all fields
    drude = n * E**2 * ttau[0] / (5 * ME)
    print(
        f'Situation is n={n:.2e} /m3 and sxxDrude={drude/1e8:.2e} (muOhmcm)^-1 at B=0')
    wct = E * Bvals * ttau[0] / (5 * ME)
    drude_xx = drude / (1 + wct**2)
    drude_xy = drude_xx * wct

    # Stringent for normal field
    wh = Bvals > 2
    assert(all(np.abs(drude_xx[wh] - sxx[wh]) < 1e-4 * drude_xx[wh]))
    assert(all(np.abs(drude_xy[wh] - sxy[wh]) < 1e-4 * max(drude_xx[wh])))

    # A bit more relaxed for tiny fields
    # This is below wc=0.07 at B=2, where the algorithm becomes a bit less accurate
    # only to recover at B~0. The reason for this is the change from
    # survival between points with trapezoidal integration (which is
    # bad if you survive for like 2 points) to an exact form if
    # you take wctau=0 limit (which works better and better as B decreases)
    #
    # All of this can be improved by increasing the number of points on the FS.
    wh = Bvals <= 2
    assert(all(np.abs(drude_xx[wh] - sxx[wh]) < 1e-3 * drude_xx[wh]))
    assert(all(np.abs(drude_xy[wh] - sxy[wh]) < 1e-3 * max(drude_xx)))
    wh = Bvals <= 0.1
    assert(all(np.abs(drude_xx[wh] - sxx[wh]) < 1e-4 * drude_xx[wh]))
    assert(all(np.abs(drude_xy[wh] - sxy[wh]) < 1e-4 * max(drude_xx)))

    # # Make sure we really tested low as well as high field limit
    assert(wwc[0] * ttau[0] * min(Bvals) < 0.01)
    assert(wwc[0] * ttau[0] * max(Bvals) > 1)

    plt.figure('Situation')
    plt.plot(pphi, kkx / max(kkx), label='kx')
    plt.plot(pphi, kky / max(kky), label='ky')
    plt.plot(pphi, vvx / max(vvx) + 0.1, label='vx + 0.1')
    plt.plot(pphi, vvy / max(vvy) + 0.2, label='vy + 0.2')
    plt.plot(pphi, ttau / max(ttau) + 0.3, label='tau + 0.3')
    plt.plot(pphi, wwc / max(wwc) + 0.4, label='wc + 0.4')
    plt.xlabel('angle (rad)')
    plt.ylabel('normed')
    plt.legend()

    plt.figure('Conductivity')
    plt.plot(Bvals, sxx / 1e8, label='sxx', lw=4, color='red')
    plt.plot(Bvals, sxy / 1e8, label='sxy', lw=4, color='blue')
    plt.plot(Bvals, drude_xx / 1e8, label='drude xx',
             dashes=[6, 2], color='black')
    plt.plot(Bvals, drude_xy / 1e8, label='drude xy',
             dashes=[6, 2], color='white')
    plt.legend()
    plt.xlabel('Field (T)')
    plt.ylabel('Conductivity ($\mu\Omega cm^{-1}$)')

    plt.figure('MR')
    plt.plot(Bvals, sxx / (sxx**2 + sxy**2), label='rho_xx')
    plt.plot(Bvals, sxy / (sxx**2 + sxy**2), label='rho_xy')

    plt.show()


def simple_LSCO():
    """ Some code to make sure this tighth binding is well integrated. """

    global compute_FS
    compute_FS = tight_LSCO
    compute_FS.recompile()

    from LSCO import lee_hone_parameters

    e0, t0, t1, t2 = lee_hone_parameters(0.24)


    # Parameters from lee hone at p=0.25 from Caitlin's code
    # e0 = 0.2262368400674367
    # t0 = 0.2500000000369835
    # t1 = -0.03165153137549793
    # t2 = 0.015825765687869406
    A = 3.777e-10  # From Cava1987, table 2/3


     # FORMATTING
    rows, cols = 2, 3
    height_ratios, width_ratios = [3, 1], [1]*cols
    width, height = 8, 5
    top, bottom, left, right = 0.92, 0.15, 0.10, 0.95
    hspace, wspace = 0.3, 0.45

    # THE FIGURE
    fig = plt.figure(figsize=(width, height), dpi=120)
    gs = GridSpec(rows, cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    ax = [

        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[:, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),

    ]

    temperatures = [4, 20, 50, 80]
    Bvals = np.linspace(1, 100, 50)

    for i, temperature in enumerate(temperatures):
        print(f'TEMPERATURE:  {temperature}')

        pphi, kkx, kky, vvx, vvy, ttau = precompute(10000, A, e0, t0, t1, t2, temperature)
        ttsteps, eexps, wwc = stepanalysis(pphi, A, e0, t0, t1, t2, temperature)
        print('SIGMA')
        sxx = sigma(Bvals, pphi, kkx, kky, vvx,
                vvy, ttau, eexps, ttsteps, True)
        sxy = sigma(Bvals, pphi, kkx, kky, vvx, vvy,
                ttau, eexps, ttsteps, False)

        rxx = sxx / (sxx**2 + sxy**2) * 1e8
        rxy = sxy / (sxx**2 + sxy**2) * 1e8

        df = pd.DataFrame({
            'field': Bvals,
            'rxx': rxx,
            'rxy': rxy,
            })

        df['drxx'] = df['rxx'].diff()/df['field'].diff()
        df['RH'] = df['rxy']/df['field'] * 10

        # THE PARAMETERS DATAFRAME
        if i == 0:
            df_params = pd.DataFrame({
                'phi': pphi,
                'kx': kkx,
                'ky': kky,
                'kf': np.sqrt(kkx*kkx + kky*kky),
                'vx': vvx,
                'vy': vvy,
                'vf': np.sqrt(vvx*vvx + vvy*vvy),
                'taux 0': ttau*np.cos(pphi),
                'tauy 0': ttau*np.sin(pphi),
                'tau 0': ttau,
                })

            df_data = pd.DataFrame({
                'field': Bvals,
                f'rxx {temperature}': df['rxx'],
                f'rxy {temperature}': df['rxy'],
                f'drxx {temperature}': df['drxx'],
                f'RH {temperature}': df['RH'],
                })

        else:
            df_params[f'vx {temperature}'] = vvx
            df_params[f'vy {temperature}'] = vvy
            df_params[f'vf {temperature}'] = np.sqrt(vvx*vvx + vvy*vvy)

            df_params[f'taux {temperature}'] = ttau*np.cos(pphi)
            df_params[f'tauy {temperature}'] = ttau*np.sin(pphi)
            df_params[f'tau {temperature}'] = ttau

            df_data[f'rxx {temperature}'] = df['rxx']
            df_data[f'rxy {temperature}'] = df['rxy']
            df_data[f'drxx {temperature}'] = df['drxx']
            df_data[f'RH {temperature}'] = df['RH']




        ax[0].plot(df['field'], df['rxx'], label=f'{temperature} K')
        ax[1].plot(df['field'], df['RH'], label='RH')
        ax[2].plot(pphi, 1/ttau * 1e-12)

        ax[3].plot(df['field'], df['drxx'])

    df_params.to_csv('data/new_params.csv', sep='\t')
    df_data.to_csv('data/new_data.csv', sep='\t')



    ax[0].legend(frameon=False, fontsize=7)


    ax[0].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$\rho_{xx}$',
        )
    ax[1].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$R_{H}$',
        )
    ax[2].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\theta$ ($\mathrm{rad}$)',
        ylabel=r'$\tau^{-1}$ ($\mathrm{ps^{-1}}$)',
        )

    ax[3].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$d\rho_{xx}/dH$'
        )

    ax[4].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$T$ ($\mathrm{K}$)',
        ylabel=r'$R_H$'
        )


    plt.show()



def simple_Tl2201():

    global compute_FS
    compute_FS = compute_fermi_surface_tl2201

    # FORMATTING
    rows, cols = 2, 3
    height_ratios, width_ratios = [3, 1], [1]*cols
    width, height = 8, 5
    top, bottom, left, right = 0.92, 0.15, 0.10, 0.95
    hspace, wspace = 0.3, 0.45

    # THE FIGURE
    fig = plt.figure(figsize=(width, height), dpi=120)
    gs = GridSpec(rows, cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    ax = [

        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[:, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),

    ]

    temperatures = [0, 25, 50, 75, 100]
    Bvals = np.linspace(1, 50, 50)

    for i, temperature in enumerate(temperatures):
        print(f'TEMPERATURE:  {temperature}')

        pphi, kkx, kky, vvx, vvy, ttau = precompute(50000, temperature)
        #_, vvx, vvy = modify_vf_for_Bi2201(pphi, vvx, vvy)

        ttsteps, eexps, wwc = stepanalysis(pphi, temperature)
        sxx = sigma(Bvals, pphi, kkx, kky, vvx,
                    vvy, ttau, eexps, ttsteps, True)
        sxy = sigma(Bvals, pphi, kkx, kky, vvx, vvy,
                    ttau, eexps, ttsteps, False)

        sxx /= (23.133e-10/13.226e-10)    # Correct the caxis value
        sxy /= (23.133e-10/13.226e-10)    # manually here (LSCO to Tl2201)

        rxx = sxx / (sxx**2 + sxy**2) * 1e8
        rxy = sxy / (sxx**2 + sxy**2) * 1e8

        df = pd.DataFrame({
            'field': Bvals,
            'rxx': rxx,
            'rxy': rxy,
            })

        df['drxx'] = df['rxx'].diff()/df['field'].diff()
        df['RH'] = df['rxy']/df['field'] * 10

        # THE PARAMETERS DATAFRAME
        if i == 0:
            df_params = pd.DataFrame({
                'phi': pphi,
                'kx': kkx,
                'ky': kky,
                'kf': np.sqrt(kkx*kkx + kky*kky),
                'vx': vvx,
                'vy': vvy,
                'vf': np.sqrt(vvx*vvx + vvy*vvy),
                'taux 0': ttau*np.cos(pphi),
                'tauy 0': ttau*np.sin(pphi),
                'tau 0': ttau,
                })

            df_data = pd.DataFrame({
                'field': Bvals,
                f'rxx {temperature}': df['rxx'],
                f'rxy {temperature}': df['rxy'],
                f'drxx {temperature}': df['drxx'],
                f'RH {temperature}': df['RH'],
                })

        else:
            df_params[f'vx {temperature}'] = vvx
            df_params[f'vy {temperature}'] = vvy
            df_params[f'vf {temperature}'] = np.sqrt(vvx*vvx + vvy*vvy)

            df_params[f'taux {temperature}'] = ttau*np.cos(pphi)
            df_params[f'tauy {temperature}'] = ttau*np.sin(pphi)
            df_params[f'tau {temperature}'] = ttau

            df_data[f'rxx {temperature}'] = df['rxx']
            df_data[f'rxy {temperature}'] = df['rxy']
            df_data[f'drxx {temperature}'] = df['drxx']
            df_data[f'RH {temperature}'] = df['RH']

        f_fit = lambda x, a, b, c: a + b*x + c*x*x

        df_fit = df.where(df['field']<10).dropna()
        popt, _ = curve_fit(f_fit, df_fit['field'], df_fit['RH'])
        RH0 = popt[0]

        df_fit = df.where((df['field']>30) & (df['field']<40)).dropna()
        popt, _ = curve_fit(f_fit, df_fit['field'], df_fit['RH'])
        RH35 = f_fit(35, *popt)

        ax[0].plot(df['field'], df['rxx'], label=f'{temperature} K')
        ax[1].plot(df['field'], df['RH'], label='RH')
        ax[2].plot(pphi, 1/ttau * 1e-12)

        ax[3].plot(df['field'], df['drxx'])
        ax[4].plot(temperature, RH0, 'k.')
        ax[4].plot(temperature, RH35, 'r.')


    df_params.to_csv('data/new_params.csv', sep='\t')
    df_data.to_csv('data/new_data.csv', sep='\t')



    ax[0].legend(frameon=False, fontsize=7)


    ax[0].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$\rho_{xx}$',
        )
    ax[1].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$R_{H}$',
        )
    ax[2].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\theta$ ($\mathrm{rad}$)',
        ylabel=r'$\tau^{-1}$ ($\mathrm{ps^{-1}}$)',
        )

    ax[3].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$d\rho_{xx}/dH$'
        )

    ax[4].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$T$ ($\mathrm{K}$)',
        ylabel=r'$R_H$'
        )


    plt.show()



def simple_model():

    global compute_FS
    compute_FS = compute_fermi_surface_model

    # FORMATTING
    rows, cols = 2, 3
    height_ratios, width_ratios = [3, 1], [1]*cols
    width, height = 8, 5
    top, bottom, left, right = 0.92, 0.15, 0.10, 0.95
    hspace, wspace = 0.3, 0.45

    # THE FIGURE
    fig = plt.figure(figsize=(width, height), dpi=120)
    gs = GridSpec(rows, cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    ax = [

        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[:, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),

    ]

    temperatures = [0, 25, 50, 75, 100]
    Bvals = np.linspace(1, 50, 50)

    for i, temperature in enumerate(temperatures):
        print(f'TEMPERATURE:  {temperature}')

        pphi, kkx, kky, vvx, vvy, ttau = precompute(100000, temperature)

        ttsteps, eexps, wwc = stepanalysis(pphi, temperature)
        sxx = sigma(Bvals, pphi, kkx, kky, vvx,
                    vvy, ttau, eexps, ttsteps, True)
        sxy = sigma(Bvals, pphi, kkx, kky, vvx, vvy,
                    ttau, eexps, ttsteps, False)

        sxx /= (23.133e-10/13.226e-10)    # Correct the caxis value
        sxy /= (23.133e-10/13.226e-10)    # manually here (LSCO to Tl2201)

        rxx = sxx / (sxx**2 + sxy**2) * 1e8
        rxy = sxy / (sxx**2 + sxy**2) * 1e8

        df = pd.DataFrame({
            'field': Bvals,
            'rxx': rxx,
            'rxy': rxy,
            })

        df['drxx'] = df['rxx'].diff()/df['field'].diff()
        df['RH'] = df['rxy']/df['field'] * 10

        # THE PARAMETERS DATAFRAME
        if i == 0:
            df_params = pd.DataFrame({
                'phi': pphi,
                'kx': kkx,
                'ky': kky,
                'kf': np.sqrt(kkx*kkx + kky*kky),
                'vx': vvx,
                'vy': vvy,
                'vf': np.sqrt(vvx*vvx + vvy*vvy),
                'taux 0': ttau*np.cos(pphi),
                'tauy 0': ttau*np.sin(pphi),
                'tau 0': ttau,
                })

            df_data = pd.DataFrame({
                'field': Bvals,
                f'rxx {temperature}': df['rxx'],
                f'rxy {temperature}': df['rxy'],
                f'drxx {temperature}': df['drxx'],
                f'RH {temperature}': df['RH'],
                })

        else:
            df_params[f'vx {temperature}'] = vvx
            df_params[f'vy {temperature}'] = vvy
            df_params[f'vf {temperature}'] = np.sqrt(vvx*vvx + vvy*vvy)

            df_params[f'taux {temperature}'] = ttau*np.cos(pphi)
            df_params[f'tauy {temperature}'] = ttau*np.sin(pphi)
            df_params[f'tau {temperature}'] = ttau

            df_data[f'rxx {temperature}'] = df['rxx']
            df_data[f'rxy {temperature}'] = df['rxy']
            df_data[f'drxx {temperature}'] = df['drxx']
            df_data[f'RH {temperature}'] = df['RH']

        f_fit = lambda x, a, b, c: a + b*x + c*x*x

        df_fit = df.where(df['field']<10).dropna()
        popt, _ = curve_fit(f_fit, df_fit['field'], df_fit['RH'])
        RH0 = popt[0]

        df_fit = df.where((df['field']>30) & (df['field']<40)).dropna()
        popt, _ = curve_fit(f_fit, df_fit['field'], df_fit['RH'])
        RH35 = f_fit(35, *popt)

        ax[0].plot(df['field'], df['rxx'], label=f'{temperature} K')
        ax[1].plot(df['field'], df['RH'], label='RH')
        ax[2].plot(pphi, 1/ttau * 1e-12)

        ax[3].plot(df['field'], df['drxx'])
        ax[4].plot(temperature, RH0, 'k.')
        ax[4].plot(temperature, RH35, 'r.')


    df_params.to_csv('data/new_params.csv', sep='\t')
    df_data.to_csv('data/new_data.csv', sep='\t')



    ax[0].legend(frameon=False, fontsize=7)


    ax[0].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$\rho_{xx}$',
        )
    ax[1].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$R_{H}$',
        )
    ax[2].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\theta$ ($\mathrm{rad}$)',
        ylabel=r'$\tau^{-1}$ ($\mathrm{ps^{-1}}$)',
        )

    ax[3].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$\mu_0 H$ ($\mathrm{T}$)',
        ylabel=r'$d\rho_{xx}/dH$'
        )

    ax[4].set(
        ylim=[0, None],
        xlim=[0, None],
        xlabel=r'$T$ ($\mathrm{K}$)',
        ylabel=r'$R_H$'
        )


    plt.show()




compute_FS = None
# run_Drude_test()
print('\n-------------------\n')
simple_LSCO()
#simple_Tl2201()
#simple_model()