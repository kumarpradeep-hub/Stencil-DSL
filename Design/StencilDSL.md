# StencilDSL — Python DSL Design Specification

## 1. Purpose

StencilDSL is a Python-hosted domain-specific language for expressing stencil computations. The language describes the grid, neighborhood, point computation, update policy, ordering, boundaries, scalar types, and validation rules.

---

## 2. Language Categories

To keep the DSL terminology consistent, every public language element belongs to one of the following categories.

| Category | Meaning | Examples |
|---|---|---|
| **TYPE** | DSL data or domain object | `Grid` |
| **ATTRIBUTE** | Property of a DSL object | `shape`, `dtype`, `ndim`, `name` |
| **CONSTRUCT** | Defines stencil/program structure | `@stencil`, signature, `return` |
| **OPERATOR** | Performs access or symbolic computation | `n[-1, 0]`, `+`, `sqrt`, `mean`, `cast` |
| **DESCRIPTOR** | Describes a neighborhood | `star`, `box`, `asymmetric`, `per_axis`, `offsets` |
| **POLICY** | Defines semantic behavior | `Jacobi`, `GaussSeidel`, `SOR`, `Periodic` |
| **LANGUAGE RULE** | Static rule enforced by the DSL | type promotion, offset-rank validation |

---

## 3. Core DSL Structure

| Area | Category | Main elements |
|---|---|---|
| Grid | **TYPE** | `Grid(shape, dtype, name)` |
| Grid metadata | **ATTRIBUTE** | `shape`, `dtype`, `ndim`, `name` |
| Stencil | **CONSTRUCT** | `@stencil(...)`, signature, scalar return |
| Neighborhood | **DESCRIPTOR** | `star`, `box`, `asymmetric`, `per_axis`, `offsets` |
| Computation | **OPERATOR** | neighborhood/grid access, arithmetic, math, comparison, select, reduction, cast |
| Update | **POLICY** | `Jacobi`, `GaussSeidel`, `SOR(omega)` |
| Ordering | **POLICY** | `Lexicographic`, `ReverseLexicographic`, `RedBlack` |
| Boundary | **POLICY** | `Periodic`, `Dirichlet`, `Replicate`, `Mirror` |
| Validation | **LANGUAGE RULE** | grid, neighborhood, signature, update, ordering, boundary and type checks |

---

## 4. Grid

### 4.1 Grid — TYPE

`Grid` represents the logical discrete domain used by a stencil.

```python
u = Grid(shape=(N, M), dtype=f32, name="temperature")
```

Its attributes are `shape`, `dtype`, `ndim`, and optional `name`. `ndim` is derived from `shape`.

Dimensionality is implicit in the shape: `Grid((N,), ...)` is 1D, `Grid((N, M), ...)` is 2D, `Grid((N, M, K), ...)` is 3D, and the same rule extends to N-D. Separate `Grid1D`, `Grid2D`, or `Grid3D` types are unnecessary.

---

## 5. Stencil — CONSTRUCT

A stencil defines the point-wise computation that produces one scalar result at one logical grid point. `@stencil(...)` marks the Python function as a DSL stencil definition rather than an ordinary Python function.

```python
@stencil(
    neighborhood=star(radius=1),
    update=Jacobi,
    boundary=Periodic(),
)
def smooth(u: Grid[f32], n: Neighborhood["u"]) -> f32:
    return mean(n)
```

A stencil consists of its signature, neighborhood description, point computation, update policy, optional ordering, boundary policy, and one scalar point output. The function body is the **stencil body**: it describes the computation for one logical point. Calling the decorated function applies that stencil over the logical domain and produces one output grid; callers provide grid and scalar inputs, while bound `Neighborhood` parameters are supplied by the DSL.

---

## 6. Stencil Signature

### 6.1 Inputs

A stencil may accept one or more `Grid` inputs and zero or more scalar inputs. V1 scalar types are `f16`, `f32`, `f64`, `i32`, `i64`, and `bool`.

```python
def heat(u: Grid[f32], alpha: f32, dt: f32) -> f32:
    ...
```

For V1, all grid inputs must have compatible rank and shape; grid dtypes may differ. The first `Grid` parameter is the **primary grid**: its logical domain determines the output domain, and it is the update target for order-dependent policies. Additional grid inputs are read-only auxiliary fields.

### 6.2 Neighborhood Parameter

If a grid has a declared neighborhood, the stencil receives a bound `Neighborhood` object for that grid.

```python
@stencil(neighborhood=star(radius=1))
def smooth(u: Grid[f32], n: Neighborhood["u"]) -> f32:
    return mean(n)
```

Here `u` is the grid input and `n` represents the values of `u` at the offsets selected by the descriptor.

### 6.3 Output

V1 supports exactly one scalar output per stencil point. The returned scalar type must be compatible with the declared output type. Applying the stencil over the domain produces one logical output grid with the primary grid's domain and the declared scalar return type as its element type.

---

## 7. Neighborhood — DESCRIPTOR

A neighborhood specifies which relative grid points are available to the point computation. Semantically, every descriptor denotes a statically known set of relative offsets.

### 7.1 `star(radius, center=False)`

`star` selects axis-aligned neighbors. In 2D, `star(radius=1)` resolves to `(-1, 0)`, `(1, 0)`, `(0, -1)`, `(0, 1)`. With `center=True`, `(0, 0)` is also included.

### 7.2 `box(radius, center=True)`

`box` selects every offset in the dense N-D window. In 2D, `box(radius=1)` is the full 3×3 neighborhood. For N dimensions and radius `r`, a center-inclusive box contains `(2r + 1)^N` offsets.

### 7.3 `asymmetric(backward, forward, axis=None, center=False)`

`asymmetric` allows unequal reach in opposite directions along one axis. Used on its own, `axis` is required. For example, `asymmetric(backward=2, forward=1, axis=0)` represents offsets `-2`, `-1`, and `+1` along axis 0, plus the center only when `center=True`. When nested inside `per_axis(...)`, the axis is supplied by the mapping key and may be omitted.

### 7.4 `per_axis(...)`

`per_axis` defines an axis-aligned neighborhood with independent reach on each dimension. The mapping key supplies the axis:

```python
per_axis({
    0: asymmetric(backward=2, forward=1),
    1: asymmetric(backward=1, forward=3),
})
```

This example produces axis-aligned offsets with reach 2 backward/1 forward on axis 0 and 1 backward/3 forward on axis 1. Arbitrary non-axis-aligned patterns should use `offsets([...])`.

### 7.5 `offsets([...])`

`offsets` is the fully explicit form.

```python
offsets([(-2, 0), (-1, 0), (0, 0), (1, 1)])
```

There is no `center` flag for `offsets`; the center is included only if `(0, ..., 0)` is explicitly listed.

### 7.6 Pattern Rules

`radius`, `backward`, and `forward` must be non-negative; an axis must exist in the associated grid; every explicit offset must contain exactly `Grid.ndim` components; duplicate offsets are invalid. Pattern dimensionality is inferred from the grid, so APIs such as `star2d` or `box3d` are unnecessary.

---

## 8. Neighborhood Object

A declared descriptor creates a neighborhood object such as `n`. `Neighborhood` is a **symbolic view** over the values at the descriptor's declared offsets; it does not imply that a concrete Python container of values is materialized. It supports:

| Element | Category | Meaning |
|---|---|---|
| `n[offset]` | **OPERATOR** | Access a specific declared offset |
| `for value in n` | **CONSTRUCT** | Statically iterate over neighborhood values |
| `sum(n)`, `mean(n)`, `min(n)`, `max(n)` | **OPERATOR** | Reduce the neighborhood to one scalar |
| `n.offsets` | **ATTRIBUTE** | Resolved static offset set |
| `n.count` | **ATTRIBUTE** | Number of active offsets |

`n[offset]` is valid only when the offset belongs to the declared neighborhood. Positional indexing such as `n[0]` is not supported in V1 because offset-based access is clearer and independent of internal ordering.

Neighborhood iteration ranges only over the statically declared offsets; it is not general dynamic loop support.

---

## 9. Point Computation — OPERATORS

### 9.1 Access

Explicit relative grid access is written as `u[offset]`; declared-neighborhood access is written as `n[offset]`. Both are symbolic relative accesses at the current stencil point, and offset rank must match `Grid.ndim`.

```python
u[-2, 0]
n[-1, 0]
```

### 9.2 Arithmetic and Math

Arithmetic operators are `+`, `-`, `*`, `/`, and unary `-`. Math operators are `abs(x)`, `sqrt(x)`, `exp(x)`, `log(x)`, and `pow(x, y)`.

### 9.3 Comparison and Selection

Comparison operators are `<`, `>`, `<=`, `>=`, `==`, and `!=`; they produce `bool`. Conditional scalar selection is expressed as:

```python
select(condition, true_value, false_value)
```

### 9.4 Reduction

V1 reductions are `sum(n)`, `mean(n)`, `min(n)`, and `max(n)`. Each consumes a neighborhood collection and returns one scalar.

### 9.5 Cast

Explicit conversion is written as `cast(value, type)`. Narrowing conversions require an explicit cast.

---

## 10. Update — POLICY

The update policy defines which state a grid access observes while the stencil is being applied.

| Policy | Semantics | Ordering |
|---|---|---|
| `Jacobi` | Every read observes the previous grid state; newly produced values are not visible during the same application | Not allowed |
| `GaussSeidel` | Reads may observe values already updated in the current sweep | Required |
| `SOR(omega)` | Gauss-Seidel visibility plus relaxation | Required |

For `SOR(omega)`, the stencil body returns the candidate update value and the DSL applies:

```text
result = (1 - omega) * old_center + omega * candidate
```

`omega` is part of the update policy. A suspicious numerical value may produce a warning rather than a structural error.

---

## 11. Ordering — POLICY

V1 supports `Lexicographic()` (forward coordinate order), `ReverseLexicographic()` (reverse coordinate order), and `RedBlack()` (parity-based two-phase updates). `RedBlack()` is valid only when every non-center dependency offset changes parity; otherwise same-color points may depend on one another and the stencil is rejected. Ordering is part of stencil semantics for order-dependent updates.

---

## 12. Boundary — POLICY

Boundary policy defines the meaning of an access outside the logical grid domain.

| Policy | Meaning |
|---|---|
| `Periodic()` | Wrap the coordinate around the domain |
| `Dirichlet(value)` | Return a fixed external value |
| `Replicate()` | Use the nearest valid edge value |
| `Mirror()` | Reflect the coordinate back into the domain without repeating the edge value |

Boundary is attached to the stencil, not the grid. A single policy can apply to every axis:

```python
boundary=Periodic()
```

or policies can be specified per axis:

```python
boundary={0: Periodic(), 1: Dirichlet(0.0)}
```

When an out-of-domain access violates multiple axes, violated axes are resolved in increasing axis order. Coordinate-transforming policies (`Periodic`, `Replicate`, `Mirror`) continue resolution; `Dirichlet(value)` returns the boundary value immediately. `Mirror()` uses reflection without repeating the edge point, so in 1D `-1` maps to `1` and `N` maps to `N-2`; repeated reflection is used for farther out-of-range coordinates. For an axis of extent 1, every mirrored access resolves to its only valid coordinate, `0`.

---

## 13. Multiple Grid Inputs

Each grid input may have its own neighborhood descriptor and neighborhood object. Decorator mappings use **parameter names as strings**, because Python function parameters do not exist yet when the decorator expression is evaluated.

```python
@stencil(
    neighborhood={
        "u": star(radius=1),
        "v": box(radius=1),
    },
    update=Jacobi,
    boundary=Periodic(),
)
def combine(
    u: Grid[f32],
    nu: Neighborhood["u"],
    v: Grid[f32],
    nv: Neighborhood["v"],
) -> f32:
    return mean(nu) + max(nv)
```

V1 requires compatible rank and shape across grid inputs, allows different dtypes, allows a distinct neighborhood per grid, and applies one update and boundary policy to the whole stencil. The first grid input is the **primary grid**; additional grids are read-only auxiliary fields. Under `GaussSeidel` or `SOR`, only accesses to the primary grid may observe values already updated in the current sweep. A grid without a declared neighborhood may still use explicit `grid[offset]` access.

---

## 14. Expression Type Rules — LANGUAGE RULE

V1 scalar types are `f16`, `f32`, `f64`, `i32`, `i64`, and `bool`.

| Rule | Behavior |
|---|---|
| Access | Grid and neighborhood access preserve the grid element type |
| Promotion | `f16 < f32 < f64`; integer–floating-point arithmetic promotes to floating-point |
| Literals | Numeric literals are context-typed |
| Comparison / select | Comparisons return `bool`; `select(condition, a, b)` requires a Boolean condition and compatible branch types |
| Math | `abs` preserves numeric type; `sqrt`, `exp`, and `log` require floating-point input, with integer input promoted to `f32`; `pow` follows operand promotion |
| Reduction | `sum`, `min`, and `max` preserve the promoted element type; `mean` preserves floating-point type and returns `f32` for integer neighborhoods |
| Conversion | Automatic widening is allowed; narrowing requires `cast(...)` |

---

## 15. Validation — LANGUAGE RULE

Validation is performed when the stencil definition is checked.

| Area | Required checks |
|---|---|
| Grid | At least one dimension; static integer extents must be positive; symbolic extents are allowed; supported dtype; `ndim == len(shape)` |
| Neighborhood | Non-negative reach, valid axes, offset rank equals `Grid.ndim`, no duplicate offsets, `n[offset]` belongs to the declared neighborhood |
| Signature | At least one grid input, supported scalar types, valid neighborhood binding, compatible grid rank/shape, exactly one scalar return value, compatible output type |
| Update / Ordering | `Jacobi` must not specify ordering; `GaussSeidel` and `SOR` require ordering; `SOR.omega` must be numeric; `RedBlack` requires every non-center dependency offset to change parity |
| Boundary | Valid axes, complete per-axis coverage, type-compatible `Dirichlet` value, deterministic corner resolution |

DSL validation errors represent invalid or incomplete semantics. Warnings may be emitted for valid but suspicious cases such as an empty neighborhood, radius larger than the useful domain extent, an unusually large neighborhood, a suspicious SOR relaxation factor, or precision-sensitive low-precision reductions.

---

## 16. Complete Examples

The following four examples show the intended DSL usage end to end. In every example, the stencil is first defined with `@stencil(...)` and then applied to one or more grids. `Neighborhood` objects are created automatically from the declared descriptors; callers pass only actual grids and scalar inputs.

### 16.1 Five-Point Star with Mixed Boundaries

This is a standard 2D five-point stencil. `star(radius=1, center=True)` exposes the center `(0, 0)` and the four axis-aligned neighbors. The body computes a discrete Laplacian and scales it with the scalar input `alpha`. `Jacobi` reads only the original input state. Axis 0 uses `Periodic()` wrapping, while axis 1 uses the fixed outside value `Dirichlet(0.0)`.

```python
from stencildsl import (
    Grid, Neighborhood, stencil, star,
    Jacobi, Periodic, Dirichlet, f32,
)

temperature = Grid(shape=(N, M), dtype=f32, name="temperature")

@stencil(
    neighborhood=star(radius=1, center=True),
    update=Jacobi,
    boundary={0: Periodic(), 1: Dirichlet(0.0)},
)
def heat_step(
    u: Grid[f32],
    n: Neighborhood["u"],
    alpha: f32,
) -> f32:
    center = n[0, 0]
    neighbor_sum = n[-1, 0] + n[1, 0] + n[0, -1] + n[0, 1]
    laplacian = neighbor_sum - 4 * center
    return center + alpha * laplacian

new_temperature = heat_step(temperature, alpha=0.1)
```

`heat_step(temperature, alpha=0.1)` applies the point computation over the whole grid and returns `new_temperature`. The caller passes the grid and scalar input; `n` is bound automatically.

---

### 16.2 Explicit Five-Point Offsets with Replicate Boundary

This example defines the five active points explicitly with `offsets([...])` instead of using `star`. The body forms a weighted local average: the center has weight `0.50` and each direct neighbor has weight `0.125`. `Replicate()` reuses the nearest valid edge value whenever an access would leave the grid.

```python
from stencildsl import (
    Grid, Neighborhood, stencil, offsets,
    Jacobi, Replicate, f32,
)

u = Grid(shape=(N, M), dtype=f32, name="u")

@stencil(
    neighborhood=offsets([
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]),
    update=Jacobi,
    boundary=Replicate(),
)
def weighted_five_point(
    u: Grid[f32],
    n: Neighborhood["u"],
) -> f32:
    return (
        0.50 * n[0, 0]
        + 0.125 * n[-1, 0]
        + 0.125 * n[1, 0]
        + 0.125 * n[0, -1]
        + 0.125 * n[0, 1]
    )

v = weighted_five_point(u)
```

`weighted_five_point(u)` evaluates the weighted expression at every point and returns `v`; the neighborhood comes entirely from the explicit offset list.

---

### 16.3 Per-Axis Asymmetric Neighborhood with Gauss-Seidel Ordering

This example uses `per_axis(...)` to define a different asymmetric reach on each dimension. Along axis 0, the stencil can read two positions backward and one forward; along axis 1, it can read one position backward and two forward. The center is included once through the axis-0 descriptor; the axis-1 descriptor contributes only its non-center offsets. `GaussSeidel` allows later points in the sweep to observe values already updated earlier, so `Lexicographic()` is required to define a deterministic order. `Mirror()` reflects out-of-domain accesses back into the valid range.

```python
from stencildsl import (
    Grid, Neighborhood, stencil, per_axis, asymmetric,
    GaussSeidel, Lexicographic, Mirror, f32,
)

u = Grid(shape=(N, M), dtype=f32, name="u")

@stencil(
    neighborhood=per_axis({
        0: asymmetric(backward=2, forward=1, center=True),
        1: asymmetric(backward=1, forward=2, center=False),
    }),
    update=GaussSeidel,
    ordering=Lexicographic(),
    boundary=Mirror(),
)
def biased_relaxation(
    u: Grid[f32],
    n: Neighborhood["u"],
) -> f32:
    return (
        0.10 * n[-2, 0]
        + 0.15 * n[-1, 0]
        + 0.30 * n[0, 0]
        + 0.10 * n[1, 0]
        + 0.10 * n[0, -1]
        + 0.10 * n[0, 1]
        + 0.15 * n[0, 2]
    )

v = biased_relaxation(u)
```

`biased_relaxation(u)` performs one ordered Gauss-Seidel sweep and returns `v`. The example shows that `per_axis(...)` combines independent axis-wise neighborhood rules into one static neighborhood, while update policy, ordering, and boundary behavior remain separate stencil-level choices.

---

### 16.4 Multiple Grid Inputs with Independent Neighborhoods

This example reads from two grids with independent neighborhoods. Grid `u` uses a five-point star and grid `v` uses a 3×3 box, producing the bound objects `nu` and `nv`. The body combines `mean(nu)` with `max(nv)`, scaled by the scalar input `scale`. `Periodic()` wraps out-of-domain accesses for both grids.

```python
from stencildsl import (
    Grid, Neighborhood, stencil, star, box,
    Jacobi, Periodic, mean, max, f32,
)

u = Grid(shape=(N, M), dtype=f32, name="u")
v = Grid(shape=(N, M), dtype=f32, name="v")

@stencil(
    neighborhood={
        "u": star(radius=1, center=True),
        "v": box(radius=1, center=True),
    },
    update=Jacobi,
    boundary=Periodic(),
)
def combine_fields(
    u: Grid[f32],
    nu: Neighborhood["u"],
    v: Grid[f32],
    nv: Neighborhood["v"],
    scale: f32,
) -> f32:
    local_u = mean(nu)
    local_v = max(nv)
    return local_u + scale * local_v

result = combine_fields(u, v, scale=0.25)
```

`combine_fields(u, v, scale=0.25)` passes only the two grids and scalar input. `nu` and `nv` are bound automatically, and the stencil produces one scalar per point, yielding the output grid `result`.

These examples cover the main ways the current DSL is intended to be used: predefined and explicit neighborhoods, symmetric, asymmetric, and per-axis access patterns, scalar arithmetic and reductions, Jacobi and ordered updates, different boundary policies, scalar inputs, stencil application, and multiple grid inputs. `SOR(omega)` uses the same ordering requirement as Gauss-Seidel but additionally applies the relaxation formula defined in Section 10; the remaining built-in operators and policies are specified in their respective sections and do not require separate examples.

---

## 17. Potential Extensions (Post-V1)

The following features are intentionally not part of V1, but the current design leaves room for them:

| Extension | Why it may be useful |
|---|---|
| **Multiple outputs** | Return multiple scalar values from one point computation and produce multiple output grids, useful for coupled PDE systems. |
| **Per-grid boundary policies** | Allow different input grids in the same stencil to use different boundary conditions. |
| **Different grid shapes or staggered grids** | Support fields that do not share identical extents or that live at different physical locations, such as pressure and velocity on staggered CFD grids. |
| **Vector/tensor-valued grid elements** | Allow one grid point to contain a small vector or tensor rather than only a scalar. |
| **Non-uniform grid spacing** | Associate spacing information with axes or points for discretizations whose coefficients depend on physical distance. |
| **Custom reductions / pure scalar functions** | Let users define reusable point-local computations beyond the built-in arithmetic, math, and reduction operators. |
| **Dynamic neighborhoods** | Allow data-dependent neighbor selection. V1 deliberately requires statically known offsets. |
| **Additional ordering policies** | Add other deterministic sweep or coloring schemes when required by specific algorithms. |
| **Explicit state access** | Provide an advanced `previous`/`current` escape hatch for algorithms that cannot be expressed cleanly using only the selected update policy. |

---

## 18. V1 Scope Summary

V1 supports 1D through N-D scalar grids; predefined and explicit static neighborhoods; symmetric, asymmetric, and per-axis patterns; arithmetic, math, comparison, select, reduction, and cast operators; Jacobi, Gauss-Seidel, and SOR update policies; deterministic ordering for order-dependent updates; per-axis boundary policies; multiple compatible grid inputs with independent neighborhoods; deterministic scalar type rules; and static validation.
