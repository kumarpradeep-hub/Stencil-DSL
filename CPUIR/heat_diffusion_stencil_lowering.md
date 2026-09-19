# Stencil IR Lowering Pipeline: `heat_diffusion` (1D Stencils)

This document traces the compilation of a **1D** three-point stencil kernel (`heat_diffusion`) through the `stencil-opt` pass pipeline, from high-level Stencil dialect IR down to vectorized `memref`/`scf` IR.

> **Scope:** This pipeline and trace apply to **one-dimensional stencils** — fields of shape `[N]xf64` with single-axis access offsets (e.g. `[-1]`, `[0]`, `[1]`). Multi-dimensional stencils (2D/3D) follow the same overall pass structure but involve additional axes for halo, tiling, and boundary-loop generation.

## Overview

The kernel computes, for each interior point `i`:

```
out[i] = u[i-1] + u[i] + u[i+1]
```

with **Dirichlet boundary conditions** (`boundary_value = 0.0`) applied at the domain edges, over a field of size `4096`.

## Pipeline Stages

| Stage | Pass flags added | Purpose |
|---|---|---|
| `0-input` | *(none — parse/verify)* | Canonicalizes generic syntax (e.g. drops explicit `!stencil.field`/`!stencil.temp` spelling redundancy) |
| `1-fusion` | `--stencil-cross-kernel-fusion --stencil-local-fusion` | Attempts to fuse this kernel with adjacent kernels / duplicate compute within the kernel. No-op here since there is only one `stencil.apply`. |
| `2-structural-analysis` | `--stencil-infer-halo --stencil-model-boundary-conditions --stencil-select-decomposition-strategy --stencil-decide-bounds-mode` | Infers halo width from access offsets, models the boundary condition, and decides on a decomposition/execution strategy. |
| `4-tile-fuse` | `--stencil-tile-fuse` | Tiling/fusion opportunities for locality. No-op for this simple kernel/size. |
| `5-to-memref` | `--stencil-to-memref` | Lowers the abstract `stencil.apply`/`stencil.access` operations into explicit `memref` loads/stores inside `scf.for` loops, splitting interior and boundary regions. |
| `6-vectorize` | `--stencil-vectorize --stencil-map-to-matrix-units` | Attempts SIMD/matrix-unit mapping. No-op here (loop-carried scalar loads/stores remain unchanged at this problem size/shape). |

## Stage-by-Stage IR

### Stage 0 — Input (parsed/canonicalized)

```mlir
module {
  func.func @heat_diffusion(%arg0: !stencil.field<[4096]x f64>, %arg1: !stencil.field<[4096]x f64>) {
    %0 = stencil.load %arg0 : <[4096]x f64> -> <[4096]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[4096]x f64>) -> !stencil.temp<[4096]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64} {
      %2 = stencil.access %arg2 [-1] : (!stencil.temp<[4096]x f64>) -> f64
      %3 = stencil.access %arg2 [0]  : (!stencil.temp<[4096]x f64>) -> f64
      %4 = arith.addf %2, %3 : f64
      %5 = stencil.access %arg2 [1]  : (!stencil.temp<[4096]x f64>) -> f64
      %6 = arith.addf %4, %5 : f64
      stencil.return %6 : f64
    }
    stencil.store %1 to %arg1 : <[4096]x f64> to <[4096]x f64>
    return
  }
}
```

### Stages 1 & 4 — Fusion / Tile-Fuse (no-ops)

Both the cross-kernel/local fusion pass and the tile-fuse pass leave the IR **unchanged**, since the module contains a single `stencil.apply` with no fusable sibling or nested tiling structure.

### Stage 2 — Structural Analysis

Halo inference and boundary-condition modeling annotate the `stencil.apply` and the enclosing module:

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: !stencil.field<[4096]x f64>, %arg1: !stencil.field<[4096]x f64>) {
    %0 = stencil.load %arg0 : <[4096]x f64> -> <[4096]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[4096]x f64>) -> !stencil.temp<[4096]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64,
                    stencil.halo = array<i64: 1>} {
      ...
    }
    stencil.store %1 to %arg1 : <[4096]x f64> to <[4096]x f64>
    return
  }
}
```

Key additions:
- **`stencil.halo = array<i64: 1>`** — a halo of 1 element is required on each side, derived from the `[-1]`/`[1]` access offsets.
- **`stencil.bounds_mode = "static"`** — domain bounds (4096) are known at compile time.
- **`stencil.decomposition_strategy = "single_rank"`** — no multi-rank/distributed decomposition needed; runs on a single rank.

### Stage 5 — Lowering to `memref` / `scf`

The abstract stencil apply is split into **three explicit loops**:

1. **Interior loop** (`i = 1 .. 4094`): the 3-point stencil computed from real neighbor loads.
2. **Left boundary loop** (`i = 0 .. 0`): writes the Dirichlet constant `0.0`.
3. **Right boundary loop** (`i = 4095 .. 4095`): writes the Dirichlet constant `0.0`.

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<4096xf64>, %arg1: memref<4096xf64>) {
    // Interior: out[i] = u[i-1] + u[i] + u[i+1]
    scf.for %arg2 = %c1 to %c4095 step %c1 {
      %0 = arith.addi %arg2, %c-1 : index
      %1 = memref.load %arg0[%0]    : memref<4096xf64>
      %2 = memref.load %arg0[%arg2] : memref<4096xf64>
      %3 = arith.addf %1, %2 : f64
      %4 = arith.addi %arg2, %c1 : index
      %5 = memref.load %arg0[%4] : memref<4096xf64>
      %6 = arith.addf %3, %5 : f64
      memref.store %6, %arg1[%arg2] : memref<4096xf64>
    }

    // Left Dirichlet boundary: out[0] = 0.0
    scf.for %arg2 = %c0 to %c1 step %c1 {
      memref.store %cst_0, %arg1[%arg2] : memref<4096xf64>
    }

    // Right Dirichlet boundary: out[4095] = 0.0
    scf.for %arg2 = %c4095 to %c4096 step %c1 {
      memref.store %cst_0, %arg1[%arg2] : memref<4096xf64>
    }
    return
  }
}
```

> Note: the single-trip boundary loops (`scf.for %arg2 = %c0 to %c1 step %c1`) are structurally loops but execute exactly once; a later canonicalization pass would typically simplify these into plain `memref.store` ops.

### Stage 6 — Vectorization

`--stencil-vectorize --stencil-map-to-matrix-units` runs but produces **no change** to the IR. The interior loop has loop-carried, offset (`i-1`, `i`, `i+1`) scalar memory accesses over a 1D array — at this size/shape the pass does not find a profitable SIMD/matrix-unit mapping, so the scalar `scf.for` loop nest from Stage 5 is preserved as-is.

## Summary

| Property | Value |
|---|---|
| Dimensionality | 1D |
| Field size | `4096` elements (`f64`) |
| Stencil pattern | 3-point (`[-1, 0, +1]`) |
| Halo width | 1 |
| Boundary condition | Dirichlet, value `0.0` |
| Decomposition strategy | `single_rank` |
| Bounds mode | `static` |
| Final form | 3 `scf.for` loops over `memref<4096xf64>` (interior + 2 boundary) |
| Vectorized? | No — pass ran but made no transformation |
