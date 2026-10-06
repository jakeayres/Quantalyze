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

To go the other way, fitting a scattering rate or a Fermi surface to measured ρ_xx(B) and
Hall data, see [Boltzmann fitting](boltzmann_fitting/index.md).

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

### Extrapolating in N { #extrapolate }

**`extrapolate=True` cancels the N⁻² error.** The solver also computes σ on every other
point and returns (4σ_N − σ_{N/2})/3 (Richardson extrapolation), so the error falls as
N⁻⁴ instead, at 1.5 times the cost.

```python
--8<-- "beta/boltzmann/extrapolate.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/extrapolate.txt"
```

![Error of sigma_xx falling as N to the minus 2 as computed and as N to the minus 4 extrapolated](../examples/beta/boltzmann/extrapolate.png#only-light)
![Error of sigma_xx falling as N to the minus 2 as computed and as N to the minus 4 extrapolated](../examples/beta/boltzmann/extrapolate-dark.png#only-dark)

At 512 points the error of σ_xx drops from 4 × 10⁻⁵ to 3 × 10⁻⁹. The same holds for R_H,
for σ at any field, and for open orbits and k_z-warped surfaces (each slice is
extrapolated on its own). `conductivity_tensor` takes the same option.

!!! warning "Watch out"

    - **`layer_spacing` is the distance between conducting layers, not always c.** In a
      body-centred cell (e.g. Tl₂Ba₂CuO₆, La₂₋ₓSrₓCuO₄) it is c/2. Getting it wrong
      scales σ by the same factor.
    - **Velocities are the group velocity ∇ε/ħ in m/s,** not unit vectors and not in
      eV·Å. The solver warns if they are not normal to the contour, which usually means
      they are in the wrong form (or the contour is too coarsely sampled).
    - **Use enough points.** The error falls as N⁻² at every field; N = 512 gives about
      5 × 10⁻⁵ on a simple pocket, and more where τ or the curvature changes quickly
      between neighbouring points. The discretisation adds no magnetoresistance of its
      own: at low field σ(B) − σ(0) grows as B², as it physically must, so the shape of
      the low-field MR is right even with few points and only its size carries the
      O(N⁻²) error. To check a result, rerun it with twice the points, or use
      [`extrapolate=True`](#extrapolate).
    - **Resolve narrow hot spots.** If τ changes by more than about 1.5× between
      neighbouring points, the solver warns: a hot spot only a point or two wide is
      under-resolved, and the magnetoresistance can be off by several per cent (by a
      quarter when the hot spot is narrower than the point spacing). Add points there.
    - **`extrapolate` needs smoothly sampled points,** an even number of them and at
      least 32. It assumes the error is c/N² with the same c on every other point, which
      holds when the points follow a smooth contour at smoothly varying spacing (the
      generators, a band-structure calculation). On noisy or irregular points, such as
      a measured ARPES contour, it can make things worse, so compare with and without.
      It improves the magnetoresistance only a few-fold below ω_cτ ≈ 2π/N, where a
      small error of order |B|/N remains.
    - **Contours must be closed, unless you pass `period`** (see [open orbits](#open-orbits)).
      Don't repeat the first point at the end; it is dropped if you do.
    - **Onsager's relation σ(−B) = σ(B)ᵀ holds by construction.** With
      `symmetrize=True` (the default) the solver also runs each orbit the other way
      round and warns if the two ever disagree beyond rounding, which would mean a bug.
      `symmetrize=False` skips that check, and is up to twice as fast for positive fields.

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

### Open sheets from your own band { #open-sheets-band }

**`bz.generators.open_sheets_from_dispersion` traces the open sheets of any band ε(k).**
Give the reciprocal-lattice vector G that the sheets repeat along as `period`, and a range
`across` them to search. It returns one period of each sheet, with v = ∇ε/ħ at every
node, together with the `period` to pass on to `conductivity`. With `n_kz` it also slices a
band warped along k_z, as `from_dispersion_3d` does for pockets.

```python
--8<-- "beta/boltzmann/open_sheets_band.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/open_sheets_band.txt"
```

![Magnetoresistance growing without limit across the chains, saturating between the layers, and negligible along the chains](../examples/beta/boltzmann/open_sheets_band.png#only-light)
![Magnetoresistance growing without limit across the chains, saturating between the layers, and negligible along the chains](../examples/beta/boltzmann/open_sheets_band-dark.png#only-dark)

The field along z sweeps carriers along the sheets through k_x. v_x changes sign on the
way and averages out, so ρ_xx grows without limit: this is the open-orbit
magnetoresistance. The part of v_z that comes from hopping straight up (t_z) is the
same all along each orbit and survives any field. As a result ρ_zz saturates, here
towards (t_d/t_z)² = 1. Along the chains v_y hardly changes, and neither does ρ_yy.

!!! warning "Watch out"

    - **Every line across the sheets must cross the Fermi surface the same number of times
      within `across`.** A closed pocket in the range, or a range that cuts through a
      sheet, raises an error. The sheets are numbered in order along n̂, which is G
      turned by +90°.
    - **`period` must be a period of the band,** ε(k + G) = ε(k). The function checks
      this and raises an error if the sheets do not repeat after G.
    - **`tau` is a function of k here:** τ(k_x, k_y), or τ(k_x, k_y, k_z) with `n_kz`, not
      of the angle φ that the pocket generators use.
    - **Only B ∥ z is supported,** as for [warping along k_z](#kz-warping).

### Warping along k_z { #kz-warping }

A layered metal's Fermi surface is a corrugated cylinder: its cross-section changes
with k_z. Give it as slices at evenly spaced k_z (one period, 2π/d) with `kx`, `ky`,
`kz`, `vx`, `vy`, `vz` columns, and pass `kz="kz"`. With the field along c each carrier
stays in its slice, so each slice is an ordinary orbit carrying v_z with it, and the
result is the full 3×3 tensor, including the interlayer σ_zz.

```python
--8<-- "beta/boltzmann/kz_warping.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/kz_warping.txt"
```

![Interlayer magnetoresistance growing strongly while the in-plane one stays near zero](../examples/beta/boltzmann/kz_warping.png#only-light)
![Interlayer magnetoresistance growing strongly while the in-plane one stays near zero](../examples/beta/boltzmann/kz_warping-dark.png#only-dark)

Here v_z changes sign around every orbit (the interlayer hopping vanishes on the
diagonals), so the orbital motion averages it away and the interlayer resistance rises
steeply with field, while the nearly circular in-plane pocket has almost none.

!!! warning "Watch out"

    - **The slices must be evenly spaced over exactly one period of k_z,** 2π/d (with d
      the `layer_spacing`); `bz.generators.from_dispersion_3d` makes them that way. A
      handful of slices per period of the warping is usually enough, because averaging
      evenly spaced slices of a periodic function is very accurate.
    - **Only B ∥ c is supported.** Tilted fields would move carriers between slices.
    - **`magnetoresistance(sigma, component="zz")`** gives the interlayer
      magnetoresistance; `carrier_density(df, ..., kz="kz")` averages the slices.

### Scattering beyond the relaxation time { #scattering-kernels }

**Pass `scattering_kernel=P` to include where scattered carriers go, not only how often
they leave.** A relaxation time removes carriers from the current and returns nothing.
Real scattering sends a carrier from k to particular states k′, and the carriers
arriving there carry current too: this in-scattering is the current vertex correction.
Give it as a kernel P(k, k′), the rate for scattering from k into states near k′ per
unit density of states per spin (J·m³/s). `conductivity` then solves the full
linearised Boltzmann equation on every contour at once, still exactly in ω_cτ. `tau`
becomes the background relaxation time (np.inf for none), and the kernel adds its own
out-scattering rate Γ(k) = ∫dμ′ P(k, k′).

```python
--8<-- "beta/boltzmann/scattering_kernel.py:example"
```

```text title="Output"
--8<-- "beta/boltzmann/scattering_kernel.txt"
```

![Magnetoresistance across the chains with the scattering kernel, and larger than with a relaxation time of the same rates](../examples/beta/boltzmann/scattering_kernel.png#only-light)
![Magnetoresistance across the chains with the scattering kernel, and larger than with a relaxation time of the same rates](../examples/beta/boltzmann/scattering_kernel-dark.png#only-dark)

Spin-density-wave fluctuations scatter carriers between the two sheets. Where the
nesting is best they are hottest, twice as fast as where t′ spoils it. A relaxation time
with the same rates misses where the carriers go. Scattering onto the other sheet
reverses their velocity along the chains, so it relaxes that current about twice as
fast: σ_yy(0) is roughly halved. It also changes how the field reshapes the current
across the chains, and the magnetoresistance there grows more than twice as large.

!!! warning "Watch out"

    - **`tau` is the background only** when a kernel is given. Do not also fold the
      kernel's rate into it; `out_scattering_rate` shows what the kernel adds.
    - **Units:** P is a rate per unit density of states *per spin*, per unit volume. A
      kernel written against the density of states per layer (per area) must be
      multiplied by d. Calibrate it with `out_scattering_rate`, or, for an isotropic kernel,
      with `density_of_states` (P = Γ / N(E_F)).
    - **The kernel must be finite, non-negative, symmetric** (P(k, k′) = P(k′, k), detailed
      balance) **and periodic in the reciprocal lattice.** For a kernel peaked at a
      momentum transfer Q, include both ±Q and reduce k − k′ ∓ Q into the first zone, as
      `bz.scattering.spin_fluctuation_kernel` does. An asymmetric kernel raises an error.
    - **Give the whole Fermi surface** (both open sheets, every k_z slice) where the kernel
      conserves particles and there is no background: otherwise the net current would
      never relax, and it raises an error.
    - **Resolve the kernel:** it must vary smoothly from node to node. A warning says so
      when the out-scattering rate changes on every other node; use more nodes. The same
      warning catches a kernel that is not periodic along an open sheet.
    - **Pockets coupled by the kernel are solved together,** so their conductivities do
      not add, and the cost grows as the cube of the nodes coupled: about 0.4 s per field
      for 2048 nodes on a laptop. `extrapolate=True` keeps the accuracy with fewer nodes.
    - **The kernel does not depend on B**, and the distribution is taken at the Fermi
      level (k_BT ≪ E_F). For a field-dependent kernel, call once per field.

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

## `density_of_states` { #density_of_states }

**N(E_F) per spin per volume, summed over the contours.** It uses the same node weights
as `conductivity`, so it calibrates an isotropic kernel exactly: P = Γ / N(E_F). For a
circle it is m/(2πħ²d). It is per spin, so the specific heat coefficient is
γ = (π²/3) k_B² g_s N(E_F).

## `out_scattering_rate` { #out_scattering_rate }

**The rate Γ(k) = ∫dμ′ P(k, k′) a kernel gives at every node,** aligned with each
DataFrame's rows. Use it to scale a kernel to a target rate, as in the example above. Pass
every contour the kernel couples: Γ integrates over all of them.

## `mean_free_path` { #mean_free_path }

**The vector mean free path L at zero field at every node.** In the relaxation-time
approximation it is vτ. With a kernel it includes the in-scattering: it lengthens where
scattering is forward, and shortens or rotates where carriers are sent across the Fermi
surface. σ(0) = g_s e² ∫dμ v ⊗ L. The weak-field Hall conductivity is set by the area L
sweeps out around the Fermi surface (Ong's construction), which then includes the vertex
corrections.

## Building contours { #generators }

`bz.generators` makes contour DataFrames:

- **From your own band:** `from_dispersion` traces ε(k) = 0 for any ε(k) and ∇ε you
  supply (above), `from_dispersion_3d` does the same for a k_z-warped surface, slice by
  slice, `open_sheets_from_dispersion` traces [open sheets](#open-sheets-band), and
  `polar` builds a pocket of any shape from k_F(φ).
- **Test shapes with known answers:** `circle`, `ellipse`, `tight_binding` and
  `open_sheets`.

`tau` can be a number or a function of the angle φ around the pocket; `bz.scattering`
has `constant`, `cos4phi` and `hot_spot`, and any function of φ works. It also has
`spin_fluctuation_kernel`, a [scattering kernel](#scattering-kernels) peaked at an
ordering wavevector Q.

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

??? info "`bz.density_of_states`, `bz.out_scattering_rate`, `bz.mean_free_path`"

    ::: quantalyze.beta.boltzmann.density_of_states
        options:
          heading_level: 3

    ::: quantalyze.beta.boltzmann.out_scattering_rate
        options:
          heading_level: 3

    ::: quantalyze.beta.boltzmann.mean_free_path
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
