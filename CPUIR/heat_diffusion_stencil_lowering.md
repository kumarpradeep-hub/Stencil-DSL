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

Each stage below shows the **complete** IR emitted at that point, followed by an explicit **Changed** / **Unchanged** breakdown relative to the previous stage.

---

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

**Changed (vs. raw hand-written source):**
- SSA value names normalized to `%0, %1, %2, ...` instead of the original `%u_t`, `%result`, `%v0`, etc.
- Type spelling canonicalized: `!stencil.field<[4096]xf64>` → `<[4096]x f64>` at use sites (generic printer form).

**Unchanged:**
- Overall structure: one `stencil.load`, one `stencil.apply` (3-point stencil body), one `stencil.store`.
- All attributes (`boundary_condition = "dirichlet"`, `boundary_value = 0.0`).
- No module-level attributes yet.

---

### Stage 1 — Fusion (`--stencil-cross-kernel-fusion --stencil-local-fusion`)

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

**Changed:** nothing — IR is byte-for-byte identical to Stage 0.

**Unchanged:** everything. There is only a single `stencil.apply` in the module, so cross-kernel fusion has no second kernel to fuse with, and local fusion finds no duplicate/redundant sub-computations inside the one apply body to merge.

---

### Stage 2 — Structural Analysis (`--stencil-infer-halo --stencil-model-boundary-conditions --stencil-select-decomposition-strategy --stencil-decide-bounds-mode`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: !stencil.field<[4096]x f64>, %arg1: !stencil.field<[4096]x f64>) {
    %0 = stencil.load %arg0 : <[4096]x f64> -> <[4096]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[4096]x f64>) -> !stencil.temp<[4096]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64,
                    stencil.halo = array<i64: 1>} {
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

**Changed:**
- **Module attributes added:** `stencil.bounds_mode = "static"` and `stencil.decomposition_strategy = "single_rank"` attached to the top-level `module`.
- **`stencil.apply` attribute added:** `stencil.halo = array<i64: 1>`, computed by walking the `stencil.access` offsets (`[-1]`, `[0]`, `[1]`) inside the apply body and taking the max absolute offset (`1`) per axis.

**Unchanged:**
- The `stencil.apply` body itself (all `stencil.access`/`arith.addf`/`stencil.return` ops) — this pass only annotates, it does not rewrite computation.
- `stencil.load` / `stencil.store` ops.
- The pre-existing `boundary_condition` / `boundary_value` attributes (only read/modeled here, not altered).

---

### Stage 4 — Tile-Fuse (`--stencil-tile-fuse`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: !stencil.field<[4096]x f64>, %arg1: !stencil.field<[4096]x f64>) {
    %0 = stencil.load %arg0 : <[4096]x f64> -> <[4096]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[4096]x f64>) -> !stencil.temp<[4096]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64,
                    stencil.halo = array<i64: 1>} {
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

**Changed:** nothing — IR is byte-for-byte identical to Stage 2.

**Unchanged:** everything. Tile-fuse merges tiled loop nests that share iteration space after tiling; there is no tiling structure yet at this point (that only appears after lowering to loops), and only a single apply exists, so the pass is a no-op on this kernel.

> **Note:** `stage 3` is intentionally absent from the numbered trace — the pipeline's internal pass numbering skips from `2` to `4` here (no separate dumped stage exists between structural analysis and tile-fuse for this run).

---

### Stage 5 — Lowering to `memref` / `scf` (`--stencil-to-memref`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<4096xf64>, %arg1: memref<4096xf64>) {
    %c1 = arith.constant 1 : index
    %c4095 = arith.constant 4095 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c4095 step %c1_0 {
      %c-1 = arith.constant -1 : index
      %0 = arith.addi %arg2, %c-1 : index
      %1 = memref.load %arg0[%0] : memref<4096xf64>
      %2 = memref.load %arg0[%arg2] : memref<4096xf64>
      %3 = arith.addf %1, %2 : f64
      %c1_5 = arith.constant 1 : index
      %4 = arith.addi %arg2, %c1_5 : index
      %5 = memref.load %arg0[%4] : memref<4096xf64>
      %6 = arith.addf %3, %5 : f64
      memref.store %6, %arg1[%arg2] : memref<4096xf64>
    }
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1_1 step %c1_2 {
      %cst = arith.constant 0.000000e+00 : f64
      memref.store %cst, %arg1[%arg2] : memref<4096xf64>
    }
    %c4095_3 = arith.constant 4095 : index
    %c4096 = arith.constant 4096 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg2 = %c4095_3 to %c4096 step %c1_4 {
      %cst = arith.constant 0.000000e+00 : f64
      memref.store %cst, %arg1[%arg2] : memref<4096xf64>
    }
    return
  }
}
```

**Changed — this is the largest transformation in the pipeline:**
- **Types:** `!stencil.field<[4096]x f64>` → `memref<4096xf64>` for both arguments; `!stencil.temp<...>` is eliminated entirely.
- **Ops eliminated:** `stencil.load`, `stencil.store`, `stencil.apply`, `stencil.access`, `stencil.return` all disappear.
- **Ops introduced:** `scf.for` (×3), `arith.constant` (index and f64 constants for bounds/offsets/boundary value), `arith.addi` (index arithmetic for `i-1`/`i+1`), `memref.load`, `memref.store`.
- **Control flow introduced:** the single abstract `stencil.apply` becomes **three separate loops**:
  1. Interior loop, `i ∈ [1, 4095)`: real 3-point stencil (`memref.load` at `i-1`, `i`, `i+1`; `arith.addf` ×2; `memref.store` at `i`).
  2. Left boundary loop, `i ∈ [0, 1)`: stores the Dirichlet constant `0.0` — derived directly from `stencil.boundary_value` and the halo of `1`.
  3. Right boundary loop, `i ∈ [4095, 4096)`: stores the Dirichlet constant `0.0`, symmetric to the left boundary.

**Unchanged:**
- The module-level attributes (`bounds_mode`, `decomposition_strategy`) are carried through unmodified.
- The underlying arithmetic being performed (still `u[i-1] + u[i] + u[i+1]` for interior points, `0.0` at the two boundary points) — semantics preserved, only representation changes.

---

### Stage 6 — Vectorization (`--stencil-vectorize --stencil-map-to-matrix-units`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<4096xf64>, %arg1: memref<4096xf64>) {
    %c1 = arith.constant 1 : index
    %c4095 = arith.constant 4095 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c4095 step %c1_0 {
      %c-1 = arith.constant -1 : index
      %0 = arith.addi %arg2, %c-1 : index
      %1 = memref.load %arg0[%0] : memref<4096xf64>
      %2 = memref.load %arg0[%arg2] : memref<4096xf64>
      %3 = arith.addf %1, %2 : f64
      %c1_5 = arith.constant 1 : index
      %4 = arith.addi %arg2, %c1_5 : index
      %5 = memref.load %arg0[%4] : memref<4096xf64>
      %6 = arith.addf %3, %5 : f64
      memref.store %6, %arg1[%arg2] : memref<4096xf64>
    }
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1_1 step %c1_2 {
      %cst = arith.constant 0.000000e+00 : f64
      memref.store %cst, %arg1[%arg2] : memref<4096xf64>
    }
    %c4095_3 = arith.constant 4095 : index
    %c4096 = arith.constant 4096 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg2 = %c4095_3 to %c4096 step %c1_4 {
      %cst = arith.constant 0.000000e+00 : f64
      memref.store %cst, %arg1[%arg2] : memref<4096xf64>
    }
    return
  }
}
```

**Changed:** nothing — IR is byte-for-byte identical to Stage 5.

**Unchanged:** everything. `--stencil-vectorize` and `--stencil-map-to-matrix-units` run over the loop nest but do not rewrite it here: the interior loop's memory accesses are offset by a loop-carried induction variable (`i-1`, `i`, `i+1`) in a 1D scalar-store pattern, and the pass finds no profitable SIMD width / matrix-unit mapping to apply at this shape, so all three `scf.for` loops, and every op inside them, are passed through unchanged.

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
