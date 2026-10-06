# Surfaces and rate models

A fit needs two models: the **Fermi surface**, which says where the carriers are and how
fast they move, and the **scattering rate** 1/τ at every point of it. This page covers
fixed Fermi surfaces and rate models; [Fitting the Fermi surface](fermi_surface.md)
covers surfaces with parameters.

??? example "Example data used on this page"

    The running sample: a hole pocket about the zone corner, with 1/τ = Γ₀ + Γ₁ cos²2φ, and
    its prepared `data` (see [Preparing your data](prepare.md)).

    ```python
    --8<-- "beta/boltzmann_fitting/_data.py"
    ```

## The argument-name rule

A rate is a Python function that returns 1/τ in s⁻¹. **Its argument names say what each
argument is:**

- an argument named after a **column of the surface DataFrame** receives that column, as an
  array with one value per point: `kx`, `ky`, `vx`, `vy`, or any column you add;
- **every other argument is a parameter** to fit, and needs a starting value in `p0`;
- an argument with a **default value** is held at it, unless `p0` names it.

```python
def rate(phi, gamma0, gamma1):          # phi is a column; gamma0 and gamma1 are fitted
    return gamma0 + gamma1 * np.cos(2 * phi) ** 2

def rate(a, b, c):                      # no columns: an isotropic rate a + b + c
    return a + b + c

def rate(phi, gamma0, height, width=0.3):  # width is held at 0.3 unless p0 gives it
    ...
```

Add columns for whatever your rate depends on. `bzf.polar_angle(surface, center=...)` gives
the angle φ about a pocket's centre; the distance to a hot spot, the local |v|, or the
k_z of a slice work the same way.

Write rates rather than τ: independent scattering processes add as rates
(Matthiessen's rule), so `gamma_impurity + gamma_inelastic(phi)` is a natural model.

## Choosing a rate model

**Fit several candidate forms to the same data and compare χ²/ν.**

```python
--8<-- "beta/boltzmann_fitting/rate_forms.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/rate_forms.txt"
```

![The fitted scattering rate against angle for five models, with the anisotropic and harmonic forms lying on the truth](../../examples/beta/boltzmann_fitting/rate_forms.png#only-light)
![The fitted scattering rate against angle for five models, with the anisotropic and harmonic forms lying on the truth](../../examples/beta/boltzmann_fitting/rate_forms-dark.png#only-dark)

- **`anisotropic`** is the form the data were made with: χ²/ν ≈ 1 and the true rates.
- **`harmonics`** fits exactly as well, because it is the same model written differently:
  Γ₀ + Γ₁cos²2φ = (Γ₀ + Γ₁/2)(1 + a₄cos 4φ) with a₄ = Γ₁/(2Γ₀ + Γ₁) = 0.5. Different
  parametrisations of one model give one χ².
- **`hot_spots`** has the right symmetry but the wrong shape, and χ²/ν = 7.7 says so.
- **`isotropic`** and **`mean_free_path`** (1/τ = |v|/ℓ, using the `vx` and `vy` columns)
  cannot fit at all.

Use only terms the crystal allows. On a pocket with C₄ symmetry and mirror planes, the
allowed harmonics in φ are cos 4nφ; terms like sin 4φ cannot be determined (see
[What the data can't tell you](limits.md#things-no-data-can-fix)).

## Your own Fermi surface

**Any DataFrame with columns `kx`, `ky`, `vx`, `vy` (SI units) and a `tau` placeholder is a
surface.** The points go in order around the pocket, and v is the group velocity, normal to
the contour. Here the surface is built from a table of k_F and |v_F| at 90 angles, as ARPES
might give, interpolated smoothly onto 512 points.

```python
--8<-- "beta/boltzmann_fitting/surface_from_points.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/surface_from_points.txt"
```

For a band you can write down, `bz.generators.from_dispersion` and `tight_binding` do this
for you, as in the running sample; see [Boltzmann transport](../boltzmann.md).

## Several pockets

**Pass a list of surfaces, and a list of rates in the same order.** σ is summed over the
pockets before inverting, which is the correct way to combine bands. Give each pocket's
rate its own parameter names, or share a name to share a parameter.

```python
--8<-- "beta/boltzmann_fitting/several_pockets.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/several_pockets.txt"
```

One rate (not a list) applies to every pocket; `None` in the list keeps that pocket's own
`tau` column.

## Open sheets

**Pass `period=` exactly as to `bz.conductivity`.** Everything `bz.conductivity` accepts
(`period`, `kz`, `charge`, `scattering_kernel`, `extrapolate`) goes through
`fit_transport` and `transport` unchanged. Open sheets have no centre, so write the rate in
terms of k.

```python
--8<-- "beta/boltzmann_fitting/open_sheets.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann_fitting/open_sheets.txt"
```

!!! warning "Watch out"

    - **Keep the rate positive.** A rate that goes negative at some point stops the model;
      `fit_transport` turns such steps back, but `bounds` (e.g. `{"a4": (-0.95, 0.95)}`)
      are faster and safer.
    - **Give starting values of the right size** (1e12 for a rate in s⁻¹, 1e-8 for a length
      in m): they set each parameter's scale for the optimiser.
    - **Don't fit a channel that is zero by symmetry.** The Hall resistivity of a symmetric
      pair of open sheets is zero to rounding; weighting that rounding noise by 0.2% of
      itself gives nonsense. Pass `hall_resistivity=None`.
    - **A column argument must be a column of every pocket its rate applies to.** Rename the
      argument, or add the column, if not.
