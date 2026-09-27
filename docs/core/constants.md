# Constants

Physical and mathematical constants in SI units, available as `qz.constants`. Values are CODATA 2018, which are exact where the 2019 SI redefinition fixed them.

```python
--8<-- "core/constants/constants_basic.py:example"
```

```text title="Output"
--8<-- "core/constants/constants_basic.txt"
```

## All constants

| Name | Short name | Value | Unit |
|---|---|---|---|
| `PI` | | 3.141592653589793 | |
| `SPEED_OF_LIGHT` | | 299 792 458 | m s⁻¹ |
| `GRAVITATIONAL_CONSTANT` | | 6.67430 × 10⁻¹¹ | m³ kg⁻¹ s⁻² |
| `PLANCK_CONSTANT` | | 6.62607015 × 10⁻³⁴ | J s |
| `REDUCED_PLANCK_CONSTANT` | `HBAR` | `PLANCK_CONSTANT / (2 * PI)` ≈ 1.054571818 × 10⁻³⁴ | J s |
| `ELEMENTARY_CHARGE` | `E` | 1.602176634 × 10⁻¹⁹ | C |
| `BOLTZMANN_CONSTANT` | `KB` | 1.380649 × 10⁻²³ | J K⁻¹ |
| `AVOGADRO_CONSTANT` | | 6.02214076 × 10²³ | mol⁻¹ |
| `GAS_CONSTANT` | | 8.314462618 | J mol⁻¹ K⁻¹ |
| `ELECTRON_MASS` | `EM` | 9.1093837015 × 10⁻³¹ | kg |
| `PROTON_MASS` | | 1.67262192369 × 10⁻²⁷ | kg |
| `NEUTRON_MASS` | | 1.67492749804 × 10⁻²⁷ | kg |

!!! warning "Watch out"

    **`E` is the elementary charge,** not Euler's number. For e = 2.718…, use `numpy.e` or `math.e`.
