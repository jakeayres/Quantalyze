# Boltzmann transport

Magnetoconductivity of quasi-2D metals from your own Fermi surface. Give it a Fermi
contour (points around the pocket, the group velocity at each, and a relaxation time),
and it solves the Boltzmann equation in the relaxation-time approximation with the
Shockley–Chambers tube integral. The field is along ẑ (the c axis).

The solver is **exact in ω_cτ**: every earlier orbit is included, so it is as
accurate at 1000 T as at 0.1 T. Its only error comes from sampling the contour with N
points, and that falls as N⁻². It is checked against textbook results (Drude, two-band,
anisotropic τ) and an independent brute-force calculation.

```python
from quantalyze.beta import boltzmann as bz
```

| Function | What it does | Returns |
|---|---|---|
| [`bz.conductivity`](#conductivity) | σ(B) of one pocket, or several (summed) | a `DataFrame` of σ_xx, σ_xy, σ_yx, σ_yy |
| [`bz.resistivity`](#resistivity) | ρ = σ⁻¹ | a `DataFrame` of ρ_xx, ρ_xy, ρ_yx, ρ_yy |
| [`bz.hall_coefficient`](#hall_coefficient) | R_H = ½(ρ_yx − ρ_xy)/B | a `Series` (m³/C) |
| [`bz.magnetoresistance`](#magnetoresistance) | (ρ_xx(B) − ρ_xx(0))/ρ_xx(0) | a `Series` |
| [`bz.carrier_density`](#carrier_density) | n from the area a pocket encloses | a float (m⁻³) |
| [`bz.generators`](#generators) | Build contours from your own band, or test shapes | `DataFrame`s |

Everything is in SI units: k in m⁻¹, v in m/s, τ in s, B in T. `bz.units` converts from
Å⁻¹, eV and eV·Å.

??? example "Example data used on this page"

    The examples below use these contours: `circle` is a circular electron pocket (a
    Drude metal), and `anisotropic` is the same circle with a fourfold scattering rate.

    ```python
    --8<-- "beta/boltzmann/_data.py"
    ```

## `conductivity` { #conductivity }

**Compute σ(B) for a Fermi contour.** A contour is a DataFrame with columns `kx`, `ky`,
`vx`, `vy` and `tau`, one row per point, in order around the pocket (either direction,
any starting point). `layer_spacing` is required.

```python
--8<-- "beta/boltzmann/conductivity_basic.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/conductivity_basic.txt"
```

![rho_xx and the Hall coefficient of a circular pocket are both flat in field](../examples/beta/boltzmann/conductivity_basic.png#only-light)
![rho_xx and the Hall coefficient of a circular pocket are both flat in field](../examples/beta/boltzmann/conductivity_basic-dark.png#only-dark)

For a circular pocket with a constant τ this is the Drude model, so there is no
magnetoresistance and R_H = −1/(ne) at every field: both lines are flat. That is a
good first check of your own set-up.

### Anisotropic scattering

Let τ vary around the pocket, here with the fourfold model from `bz.scattering`. Now
the field matters: the Hall coefficient crosses over from ⟨τ²⟩/⟨τ⟩² × 1/(nq) at low
field (Ong's result) to exactly 1/(nq) at high field, and the magnetoresistance
saturates.

```python
--8<-- "beta/boltzmann/anisotropic_tau.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/anisotropic_tau.txt"
```

![Hall coefficient falling from 1.25 to 1 and resistivity rising to 1.25 as omega_c tau grows](../examples/beta/boltzmann/anisotropic_tau.png#only-light)
![Hall coefficient falling from 1.25 to 1 and resistivity rising to 1.25 as omega_c tau grows](../examples/beta/boltzmann/anisotropic_tau-dark.png#only-dark)

### Several pockets

Pass a list of contours, one per pocket. Their conductivities are added, which is the
correct way to combine bands (averaging resistivities is not). A hole pocket is one
whose velocities point inwards; keep `charge` at its default, −e.

```python
--8<-- "beta/boltzmann/two_band.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/two_band.txt"
```

![Magnetoresistance of a compensated metal growing as B squared](../examples/beta/boltzmann/two_band.png#only-light)
![Magnetoresistance of a compensated metal growing as B squared](../examples/beta/boltzmann/two_band-dark.png#only-dark)

Equal numbers of electrons and holes give magnetoresistance that grows as B² without
saturating (doubling B quadruples it), while R_H stays constant.

!!! warning "Watch out"

    - **`layer_spacing` is the distance between conducting layers, not always c.** In a
      body-centred cell (e.g. Tl₂Ba₂CuO₆, La₂₋ₓSrₓCuO₄) it is c/2. Getting it wrong
      scales σ by the same factor.
    - **Velocities are the group velocity ∇ε/ħ in m/s,** not unit vectors and not in
      eV·Å. The solver warns if they are not normal to the contour, which usually means
      they are in the wrong form (or the contour is too coarsely sampled).
    - **Use enough points.** The error falls as N⁻²; N = 512 gives about 3 × 10⁻⁵ on a
      simple pocket. The same error shows up as an apparent magnetoresistance of order
      10⁻⁵ where the exact answer has none, so don't read physics into MR that small.
    - **Contours must be closed, unless you pass `period`** (see [open orbits](#open-orbits)).
      Don't repeat the first point at the end; it is dropped if you do.
    - **Symmetrisation is on by default** (`symmetrize=True`). It enforces
      σ(−B) = σ(B)ᵀ and removes a small discretisation error in the low-field Hall
      coefficient. Leave it on unless you are checking the raw numbers.

### Open orbits { #open-orbits }

A Fermi sheet that crosses the Brillouin zone never closes. Give one period of it, with
`period=(G_x, G_y)`: the reciprocal-lattice vector that takes the last point on to the
first. Carriers then drift along the sheet forever, so the magnetoresistance does not
saturate in the direction across it.

```python
--8<-- "beta/boltzmann/open_orbits.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/open_orbits.txt"
```

![Open-orbit magnetoresistance: none along x, growing without limit along y](../examples/beta/boltzmann/open_orbits.png#only-light)
![Open-orbit magnetoresistance: none along x, growing without limit along y](../examples/beta/boltzmann/open_orbits-dark.png#only-dark)

The sheets run along k_y, so the field sweeps electrons along them and they keep moving
along x in real space. Here v_x is ±v₀ everywhere on the sheets, so σ_xx does not change
with field at all, while σ_yy falls as 1/B² at high field and ρ_yy grows without limit.

!!! warning "Watch out"

    - **Give exactly one period,** without repeating the first point shifted by G (it is
      dropped if you do). Either sign of G works.
    - **Mix open and closed contours** by passing a list for `period`, aligned with the
      list of contours, with `None` for each closed pocket.
    - **Drift removal is off for open orbits,** because their drift is physical. Passing
      `remove_drift=True` together with `period` raises an error.

## `resistivity` { #resistivity }

**Invert σ to get ρ at each field.** It takes the output of `conductivity`, so combine
pockets there first. `conductivity_basic.py` above shows it in use.

## `hall_coefficient` { #hall_coefficient }

**R_H = ½(ρ_yx − ρ_xy)/B.** Using the antisymmetric part of ρ removes any contribution
that is even in B. At high field a single closed pocket gives exactly 1/(nq): −1/(ne)
for an electron pocket, +1/(ne) for a hole pocket, whatever its shape or scattering.
R_H is NaN at B = 0, where it is undefined.

## `magnetoresistance` { #magnetoresistance }

**(ρ_xx(B) − ρ_xx(0))/ρ_xx(0).** The fields you passed to `conductivity` must include
0; it raises an error otherwise.

## `carrier_density` { #carrier_density }

**n = g_s A/(4π²d) from the area A the pocket encloses.** The area follows the curve
between points using the velocities (which fix the tangent direction), so it is
accurate to about 10⁻¹⁰ with 512 points even when they are unevenly spaced.

```python
--8<-- "beta/boltzmann/own_band.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/own_band.txt"
```

![A hole pocket about the zone corner with its inward-pointing velocities](../examples/beta/boltzmann/own_band.png#only-light)
![A hole pocket about the zone corner with its inward-pointing velocities](../examples/beta/boltzmann/own_band-dark.png#only-dark)

The high-field limit is R_H = 1/(ne) = 6.9 × 10⁻¹⁰ m³/C for this hole pocket; at 10 T it
is already within 3% of that.

## Building contours { #generators }

`bz.generators` makes contour DataFrames:

- **From your own band:** `from_dispersion` traces ε(k) = 0 for any ε(k) and ∇ε you
  supply (above), and `polar` builds a pocket of any shape from k_F(φ).
- **Test shapes with known answers:** `circle`, `ellipse`, `tight_binding` and
  `open_sheets`.

`tau` can be a number or a function of the angle φ around the pocket; `bz.scattering`
has `constant`, `cos4phi` and `hot_spot`, and any function of φ works.

If your contour comes from elsewhere (ARPES, DFT), put it in a DataFrame with those
five columns, converting units with `bz.units` first.

!!! warning "Watch out"

    - **`from_dispersion` needs a star-shaped pocket:** every ray from `center` must cross
      the Fermi surface exactly once within `max_radius`. It raises an error otherwise.
    - **`energy` is measured from the Fermi level,** and must accept NumPy arrays.

## `FermiSurface` { #fermisurface }

**The original interface, now running on the same solver.** Give it k_F(θ), m*(θ) and
τ(θ) on a grid of angles; its velocities are ħk_F/m* along the normal.

```python
--8<-- "beta/boltzmann/fermi_surface_class.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/fermi_surface_class.txt"
```

!!! note "Changed"

    `FermiSurface.calculate_conductivity` now returns the exact result, so its numbers
    differ from earlier versions. Those kept only one orbit of history, which is wrong
    once ω_cτ ≳ 1 (σ_xx of a circle was 33% too low at ω_cτ = 5). The options
    `start_phi`, `end_phi`, `phi_points` and `exponent_points` are accepted but no
    longer do anything.

## API reference

??? info "`bz.conductivity`"

    ::: quantalyze.beta.boltzmann.conductivity
        options:
          heading_level: 3

??? info "`bz.resistivity`, `bz.hall_coefficient`, `bz.magnetoresistance`"

    ::: quantalyze.beta.boltzmann.resistivity
        options:
          heading_level: 3

    ::: quantalyze.beta.boltzmann.hall_coefficient
        options:
          heading_level: 3

    ::: quantalyze.beta.boltzmann.magnetoresistance
        options:
          heading_level: 3

??? info "`bz.carrier_density`"

    ::: quantalyze.beta.boltzmann.carrier_density
        options:
          heading_level: 3

??? info "`bz.conductivity_tensor` (arrays in, arrays out)"

    ::: quantalyze.beta.boltzmann.conductivity_tensor
        options:
          heading_level: 3

??? info "`bz.generators`"

    ::: quantalyze.beta.boltzmann.generators
        options:
          heading_level: 3
          show_root_heading: false

??? info "`bz.scattering`"

    ::: quantalyze.beta.boltzmann.scattering
        options:
          heading_level: 3
          show_root_heading: false

??? info "`bz.units`"

    ::: quantalyze.beta.boltzmann.units
        options:
          heading_level: 3
          show_root_heading: false

??? info "`bz.FermiSurface`"

    ::: quantalyze.beta.boltzmann.FermiSurface
        options:
          heading_level: 3
