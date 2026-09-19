# Stencil IR Lowering Pipeline: `heat_diffusion` (2D Stencils)

This document traces the compilation of a **2D** five-point stencil kernel (`heat_diffusion`) through the `stencil-opt` pass pipeline, from high-level Stencil dialect IR down to vectorized `memref`/`scf` IR, for a grid of shape `(64, 64)`.

> **Scope:** This trace applies to **two-dimensional stencils** — fields of shape `[N, M]xf64` with two-axis access offsets (e.g. `[-1, 0]`, `[1, 0]`, `[0, -1]`, `[0, 1]`, `[0, 0]`). It is the 2D counterpart of the 1D `heat_diffusion` trace, and additionally includes a final **kernel-launch-fusion** stage not present in the 1D run.

## Overview

The kernel computes, for each interior point `(i, j)`:

```
out[i, j] = u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] + u[i, j]
```

(a **5-point / von Neumann stencil**, i.e. the four axis-aligned neighbors plus the center point), with **Dirichlet boundary conditions** (`boundary_value = 0.0`) applied at all four edges of the `64 × 64` grid.

## Pipeline Stages

| Stage | Pass flags added | Purpose |
|---|---|---|
| `0-input` | *(none — parse/verify)* | Canonicalizes generic syntax. |
| `1-fusion` | `--stencil-cross-kernel-fusion --stencil-local-fusion` | Attempts kernel/local fusion. No-op here (single `stencil.apply`). |
| `2-structural-analysis` | `--stencil-infer-halo --stencil-model-boundary-conditions --stencil-select-decomposition-strategy --stencil-decide-bounds-mode` | Infers a **2-axis** halo from access offsets, models the boundary condition, decides decomposition/bounds strategy. |
| `4-tile-fuse` | `--stencil-tile-fuse` | Tiling/fusion opportunities. No-op for this kernel/size. |
| `5-to-memref` | `--stencil-to-memref` | Lowers to explicit `memref` loads/stores inside **nested** `scf.for` loops — one interior double loop plus four boundary double loops (top/bottom rows, left/right columns). |
| `6-vectorize` | `--stencil-vectorize --stencil-map-to-matrix-units` | Attempts SIMD/matrix-unit mapping. No-op here. |
| `7-kernel-launch-fusion` | `--stencil-kernel-launch-fusion` | Attempts to fuse separate loop nests into a single kernel launch. No-op here (only one kernel/function). |

## Stage-by-Stage IR

Each stage below shows the **complete** IR emitted at that point, followed by an explicit **Changed** / **Unchanged** breakdown relative to the previous stage.

---

### Stage 0 — Input (parsed/canonicalized)

```mlir
module {
  func.func @heat_diffusion(%arg0: !stencil.field<[64, 64]x f64>, %arg1: !stencil.field<[64, 64]x f64>) {
    %0 = stencil.load %arg0 : <[64, 64]x f64> -> <[64, 64]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[64, 64]x f64>) -> !stencil.temp<[64, 64]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64} {
      %2 = stencil.access %arg2 [-1, 0] : (!stencil.temp<[64, 64]x f64>) -> f64
      %3 = stencil.access %arg2 [1, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %4 = stencil.access %arg2 [0, -1] : (!stencil.temp<[64, 64]x f64>) -> f64
      %5 = stencil.access %arg2 [0, 1]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %6 = stencil.access %arg2 [0, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %7 = arith.addf %2, %3 : f64
      %8 = arith.addf %7, %4 : f64
      %9 = arith.addf %8, %5 : f64
      %10 = arith.addf %9, %6 : f64
      stencil.return %10 : f64
    }
    stencil.store %1 to %arg1 : <[64, 64]x f64> to <[64, 64]x f64>
    return
  }
}
```

**Changed (vs. raw hand-written source):**
- SSA value names normalized to `%0, %1, %2, ...`.
- Type spelling canonicalized to `<[64, 64]x f64>`.

**Unchanged:**
- Overall structure: one `stencil.load`, one `stencil.apply` with 5 `stencil.access` ops (4 neighbors + center) and 4 `arith.addf` ops, one `stencil.store`.
- `boundary_condition`/`boundary_value` attributes.

---

### Stage 1 — Fusion (`--stencil-cross-kernel-fusion --stencil-local-fusion`)

```mlir
module {
  func.func @heat_diffusion(%arg0: !stencil.field<[64, 64]x f64>, %arg1: !stencil.field<[64, 64]x f64>) {
    %0 = stencil.load %arg0 : <[64, 64]x f64> -> <[64, 64]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[64, 64]x f64>) -> !stencil.temp<[64, 64]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64} {
      %2 = stencil.access %arg2 [-1, 0] : (!stencil.temp<[64, 64]x f64>) -> f64
      %3 = stencil.access %arg2 [1, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %4 = stencil.access %arg2 [0, -1] : (!stencil.temp<[64, 64]x f64>) -> f64
      %5 = stencil.access %arg2 [0, 1]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %6 = stencil.access %arg2 [0, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %7 = arith.addf %2, %3 : f64
      %8 = arith.addf %7, %4 : f64
      %9 = arith.addf %8, %5 : f64
      %10 = arith.addf %9, %6 : f64
      stencil.return %10 : f64
    }
    stencil.store %1 to %arg1 : <[64, 64]x f64> to <[64, 64]x f64>
    return
  }
}
```

**Changed:** nothing — byte-for-byte identical to Stage 0.

**Unchanged:** everything. A single `stencil.apply` gives cross-kernel fusion nothing to fuse with, and local fusion finds no duplicate sub-computation within the apply body to merge.

---

### Stage 2 — Structural Analysis (`--stencil-infer-halo --stencil-model-boundary-conditions --stencil-select-decomposition-strategy --stencil-decide-bounds-mode`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: !stencil.field<[64, 64]x f64>, %arg1: !stencil.field<[64, 64]x f64>) {
    %0 = stencil.load %arg0 : <[64, 64]x f64> -> <[64, 64]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[64, 64]x f64>) -> !stencil.temp<[64, 64]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64,
                    stencil.halo = array<i64: 1, 1>} {
      %2 = stencil.access %arg2 [-1, 0] : (!stencil.temp<[64, 64]x f64>) -> f64
      %3 = stencil.access %arg2 [1, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %4 = stencil.access %arg2 [0, -1] : (!stencil.temp<[64, 64]x f64>) -> f64
      %5 = stencil.access %arg2 [0, 1]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %6 = stencil.access %arg2 [0, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %7 = arith.addf %2, %3 : f64
      %8 = arith.addf %7, %4 : f64
      %9 = arith.addf %8, %5 : f64
      %10 = arith.addf %9, %6 : f64
      stencil.return %10 : f64
    }
    stencil.store %1 to %arg1 : <[64, 64]x f64> to <[64, 64]x f64>
    return
  }
}
```

**Changed:**
- **Module attributes added:** `stencil.bounds_mode = "static"` and `stencil.decomposition_strategy = "single_rank"`.
- **`stencil.apply` attribute added:** `stencil.halo = array<i64: 1, 1>` — a halo of `1` on **each of the two axes**, derived from the maximum absolute offset per axis across all `stencil.access` ops (axis 0: `±1` from `[-1,0]`/`[1,0]`; axis 1: `±1` from `[0,-1]`/`[0,1]`).

**Unchanged:**
- The full `stencil.apply` body (all 5 `stencil.access` ops, 4 `arith.addf` ops, `stencil.return`) — unchanged from Stage 1.
- `stencil.load` / `stencil.store` ops and the pre-existing boundary attributes.

---

### Stage 4 — Tile-Fuse (`--stencil-tile-fuse`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: !stencil.field<[64, 64]x f64>, %arg1: !stencil.field<[64, 64]x f64>) {
    %0 = stencil.load %arg0 : <[64, 64]x f64> -> <[64, 64]x f64>
    %1 = stencil.apply(%arg2 = %0 : !stencil.temp<[64, 64]x f64>) -> !stencil.temp<[64, 64]x f64>
        attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64,
                    stencil.halo = array<i64: 1, 1>} {
      %2 = stencil.access %arg2 [-1, 0] : (!stencil.temp<[64, 64]x f64>) -> f64
      %3 = stencil.access %arg2 [1, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %4 = stencil.access %arg2 [0, -1] : (!stencil.temp<[64, 64]x f64>) -> f64
      %5 = stencil.access %arg2 [0, 1]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %6 = stencil.access %arg2 [0, 0]  : (!stencil.temp<[64, 64]x f64>) -> f64
      %7 = arith.addf %2, %3 : f64
      %8 = arith.addf %7, %4 : f64
      %9 = arith.addf %8, %5 : f64
      %10 = arith.addf %9, %6 : f64
      stencil.return %10 : f64
    }
    stencil.store %1 to %arg1 : <[64, 64]x f64> to <[64, 64]x f64>
    return
  }
}
```

**Changed:** nothing — byte-for-byte identical to Stage 2.

**Unchanged:** everything. There's still no tiling structure at the abstract `stencil.apply` level (that only appears after lowering to loops), and only one apply exists, so tile-fuse is a no-op.

> **Note:** as in the 1D trace, `stage 3` is skipped in the numbered dump — the pipeline runs continuously from structural analysis (`2`) directly into tile-fuse (`4`).

---

### Stage 5 — Lowering to `memref` / `scf` (`--stencil-to-memref`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<64x64xf64>, %arg1: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c63 step %c1_0 {
      %c1_11 = arith.constant 1 : index
      %c63_12 = arith.constant 63 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c1_11 to %c63_12 step %c1_13 {
        %c-1 = arith.constant -1 : index
        %0 = arith.addi %arg2, %c-1 : index
        %1 = memref.load %arg0[%0, %arg3] : memref<64x64xf64>
        %c1_14 = arith.constant 1 : index
        %2 = arith.addi %arg2, %c1_14 : index
        %3 = memref.load %arg0[%2, %arg3] : memref<64x64xf64>
        %c-1_15 = arith.constant -1 : index
        %4 = arith.addi %arg3, %c-1_15 : index
        %5 = memref.load %arg0[%arg2, %4] : memref<64x64xf64>
        %c1_16 = arith.constant 1 : index
        %6 = arith.addi %arg3, %c1_16 : index
        %7 = memref.load %arg0[%arg2, %6] : memref<64x64xf64>
        %8 = memref.load %arg0[%arg2, %arg3] : memref<64x64xf64>
        %9 = arith.addf %1, %3 : f64
        %10 = arith.addf %9, %5 : f64
        %11 = arith.addf %10, %7 : f64
        %12 = arith.addf %11, %8 : f64
        memref.store %12, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1_1 step %c1_2 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c63_3 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg2 = %c63_3 to %c64 step %c1_4 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_5 = arith.constant 0 : index
    %c64_6 = arith.constant 64 : index
    %c1_7 = arith.constant 1 : index
    scf.for %arg2 = %c0_5 to %c64_6 step %c1_7 {
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_8 = arith.constant 0 : index
    %c64_9 = arith.constant 64 : index
    %c1_10 = arith.constant 1 : index
    scf.for %arg2 = %c0_8 to %c64_9 step %c1_10 {
      %c63_11 = arith.constant 63 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c63_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    return
  }
}
```

**Changed — this is the largest transformation in the pipeline:**
- **Types:** `!stencil.field<[64, 64]x f64>` → `memref<64x64xf64>`; `!stencil.temp<...>` eliminated.
- **Ops eliminated:** `stencil.load`, `stencil.store`, `stencil.apply`, `stencil.access`, `stencil.return`.
- **Ops introduced:** nested `scf.for` (double loops, ×5 total loop nests), `arith.constant` (index bounds + f64 boundary value), `arith.addi` (2D index arithmetic for `i±1`, `j±1`), `memref.load`, `memref.store`.
- **Control flow introduced:** the single 2D `stencil.apply` becomes **five separate nested loop nests** (vs. three flat loops in the 1D case, since each region now needs 2 nested `scf.for` loops):
  1. **Interior double loop**, `i ∈ [1, 63)`, `j ∈ [1, 63)`: the 5-point stencil (4 neighbor `memref.load`s at `i-1`, `i+1` on axis 0 and `j-1`, `j+1` on axis 1, plus the center load; 4 `arith.addf`s; store to `out[i, j]`).
  2. **Top row boundary**, `i ∈ [0, 1)`, `j ∈ [0, 64)`: stores Dirichlet `0.0` across the entire top row.
  3. **Bottom row boundary**, `i ∈ [63, 64)`, `j ∈ [0, 64)`: stores Dirichlet `0.0` across the entire bottom row.
  4. **Left column boundary**, `i ∈ [0, 64)`, `j ∈ [0, 1)`: stores Dirichlet `0.0` down the entire left column.
  5. **Right column boundary**, `i ∈ [0, 64)`, `j ∈ [63, 64)`: stores Dirichlet `0.0` down the entire right column.

**Unchanged:**
- Module-level attributes (`bounds_mode`, `decomposition_strategy`) carried through unmodified.
- The underlying arithmetic semantics (`u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] + u[i,j]` for interior, `0.0` on all four edges) — only the representation changes.

> **Note on corners:** the four boundary loop nests overlap at the grid's four corners (e.g. `(0,0)` is written by both the top-row loop and the left-column loop). This is harmless here since every boundary write stores the same Dirichlet constant `0.0`, so the redundant store is idempotent.

---

### Stage 6 — Vectorization (`--stencil-vectorize --stencil-map-to-matrix-units`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<64x64xf64>, %arg1: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c63 step %c1_0 {
      %c1_11 = arith.constant 1 : index
      %c63_12 = arith.constant 63 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c1_11 to %c63_12 step %c1_13 {
        %c-1 = arith.constant -1 : index
        %0 = arith.addi %arg2, %c-1 : index
        %1 = memref.load %arg0[%0, %arg3] : memref<64x64xf64>
        %c1_14 = arith.constant 1 : index
        %2 = arith.addi %arg2, %c1_14 : index
        %3 = memref.load %arg0[%2, %arg3] : memref<64x64xf64>
        %c-1_15 = arith.constant -1 : index
        %4 = arith.addi %arg3, %c-1_15 : index
        %5 = memref.load %arg0[%arg2, %4] : memref<64x64xf64>
        %c1_16 = arith.constant 1 : index
        %6 = arith.addi %arg3, %c1_16 : index
        %7 = memref.load %arg0[%arg2, %6] : memref<64x64xf64>
        %8 = memref.load %arg0[%arg2, %arg3] : memref<64x64xf64>
        %9 = arith.addf %1, %3 : f64
        %10 = arith.addf %9, %5 : f64
        %11 = arith.addf %10, %7 : f64
        %12 = arith.addf %11, %8 : f64
        memref.store %12, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1_1 step %c1_2 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c63_3 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg2 = %c63_3 to %c64 step %c1_4 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_5 = arith.constant 0 : index
    %c64_6 = arith.constant 64 : index
    %c1_7 = arith.constant 1 : index
    scf.for %arg2 = %c0_5 to %c64_6 step %c1_7 {
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_8 = arith.constant 0 : index
    %c64_9 = arith.constant 64 : index
    %c1_10 = arith.constant 1 : index
    scf.for %arg2 = %c0_8 to %c64_9 step %c1_10 {
      %c63_11 = arith.constant 63 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c63_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    return
  }
}
```

**Changed:** nothing — byte-for-byte identical to Stage 5.

**Unchanged:** everything. `--stencil-vectorize` and `--stencil-map-to-matrix-units` run over the 2D loop nest, but the interior loop's row/column memory accesses are offset by loop-carried induction variables (`i±1` on the outer axis, `j±1` on the inner axis) in a pattern that isn't unit-stride/contiguous enough for the pass to find a profitable SIMD width or matrix-unit tiling at this shape — so all five loop nests pass through unchanged.

---

### Stage 7 — Kernel-Launch Fusion (`--stencil-kernel-launch-fusion`)

```mlir
module attributes {stencil.bounds_mode = "static", stencil.decomposition_strategy = "single_rank"} {
  func.func @heat_diffusion(%arg0: memref<64x64xf64>, %arg1: memref<64x64xf64>) {
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c1_0 = arith.constant 1 : index
    scf.for %arg2 = %c1 to %c63 step %c1_0 {
      %c1_11 = arith.constant 1 : index
      %c63_12 = arith.constant 63 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c1_11 to %c63_12 step %c1_13 {
        %c-1 = arith.constant -1 : index
        %0 = arith.addi %arg2, %c-1 : index
        %1 = memref.load %arg0[%0, %arg3] : memref<64x64xf64>
        %c1_14 = arith.constant 1 : index
        %2 = arith.addi %arg2, %c1_14 : index
        %3 = memref.load %arg0[%2, %arg3] : memref<64x64xf64>
        %c-1_15 = arith.constant -1 : index
        %4 = arith.addi %arg3, %c-1_15 : index
        %5 = memref.load %arg0[%arg2, %4] : memref<64x64xf64>
        %c1_16 = arith.constant 1 : index
        %6 = arith.addi %arg3, %c1_16 : index
        %7 = memref.load %arg0[%arg2, %6] : memref<64x64xf64>
        %8 = memref.load %arg0[%arg2, %arg3] : memref<64x64xf64>
        %9 = arith.addf %1, %3 : f64
        %10 = arith.addf %9, %5 : f64
        %11 = arith.addf %10, %7 : f64
        %12 = arith.addf %11, %8 : f64
        memref.store %12, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c1_1 step %c1_2 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c63_3 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c1_4 = arith.constant 1 : index
    scf.for %arg2 = %c63_3 to %c64 step %c1_4 {
      %c0_11 = arith.constant 0 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_5 = arith.constant 0 : index
    %c64_6 = arith.constant 64 : index
    %c1_7 = arith.constant 1 : index
    scf.for %arg2 = %c0_5 to %c64_6 step %c1_7 {
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c0_11 to %c1_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    %c0_8 = arith.constant 0 : index
    %c64_9 = arith.constant 64 : index
    %c1_10 = arith.constant 1 : index
    scf.for %arg2 = %c0_8 to %c64_9 step %c1_10 {
      %c63_11 = arith.constant 63 : index
      %c64_12 = arith.constant 64 : index
      %c1_13 = arith.constant 1 : index
      scf.for %arg3 = %c63_11 to %c64_12 step %c1_13 {
        %cst = arith.constant 0.000000e+00 : f64
        memref.store %cst, %arg1[%arg2, %arg3] : memref<64x64xf64>
      }
    }
    return
  }
}
```

**Changed:** nothing — byte-for-byte identical to Stage 6.

**Unchanged:** everything. Kernel-launch fusion looks for multiple separate kernel launches (e.g. from distinct `func.func`s or distinct top-level compute regions) that can be merged into one launch to reduce launch overhead. This module has only ever contained a single function/kernel, so there is nothing else to fuse it with, and all five loop nests remain exactly as emitted in Stage 5.

## Summary

| Property | Value |
|---|---|
| Dimensionality | 2D |
| Grid size | `64 × 64` elements (`f64`) |
| Stencil pattern | 5-point / von Neumann (`[-1,0], [1,0], [0,-1], [0,1], [0,0]`) |
| Halo width | `(1, 1)` — 1 element on each axis |
| Boundary condition | Dirichlet, value `0.0`, applied on all 4 edges |
| Decomposition strategy | `single_rank` |
| Bounds mode | `static` |
| Final form | 5 nested `scf.for` loop pairs over `memref<64x64xf64>` (1 interior + 4 boundary edges) |
| Vectorized? | No — pass ran but made no transformation |
| Kernel-launch fused? | No — only one kernel/function exists, so the pass is a no-op |
