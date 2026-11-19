# Multi-Level MLIR Lowering Pipeline for Dynamic 2D Heat Diffusion Stencil

This document outlines the end-to-end MLIR lowering pipeline for a dynamic $M \times N$ **2D 5-Point Heat Diffusion Stencil** targeting modern CPU architectures with L1/L2 cache locality and SIMD vectorization.

The pipeline lowers `stencil` directly to `memref` + `affine` — with interior and boundary regions separated. So that affine's precise dependence analysis is available for producer-consumer loop fusion across multiple `stencil.apply` ops, loop interchange for cache locality, and halo-aware tiling, all before dropping to `scf`/`vector` for the final vectorized lowering.

---

## 1. Why Lower Directly to Affine
- **Cross-`stencil.apply` fusion**: if a `func.func` contains two `stencil.apply` ops (e.g. one stencil feeding another), whether they can be fused as a producer-consumer pair depends on the exact dependence *distance and direction* between their accesses to the same memref. That's computed by `mlir::MemRefDependenceCheck` / `getDependenceComponents`, which only operates on affine access patterns.
- **Loop interchange for cache locality**: deciding whether `(i, j)` or `(j, i)` traversal order gives better spatial/temporal reuse is a dependence-vector question — a permutation is only legal if it doesn't reverse a dependence direction. This is native to `affine.for` via `affine::permuteLoops`; `linalg.generic` only tags dimensions as parallel/reduction and has no native notion of loop order.
- **Halo-aware tiling**: for a 5-point stencil, tiles need overlap (a "halo") equal to the stencil radius, and the profitability of a given tile shape is again a question of reuse distance between accesses — the same analysis, the same dialect.


---

## 2. Pipeline Overview

```
stencil.apply (possibly multiple, same func.func)
  — operates on !stencil.field / !stencil.temp 
      │
      ▼
Domain-Specific Structural Analysis
  — Halo/ghost-region derivation
  — Boundary-condition modeling
      │
      ▼
memref<?x?xf64>  +  affine.for / affine.load / affine.store
  — interior loop and boundary loops separated immediately,
    boundary already branch-free (constant-zero substitution, no affine.if)
      │
      ▼  [cleanup: canonicalize, cse, constant-fold]
      │
Affine dependence analysis (MemRefDependenceCheck)
  — is producer/consumer fusion legal across stencil.apply boundaries?
  — -affine-loop-fusion where legal
      │
      ▼  [cleanup: cse, constant-fold]
      │
Loop interchange (affine::permuteLoops)
  — reorder (i, j) vs (j, i) for spatial + temporal cache reuse,
    only where the dependence vector permits it
      │
      ▼
Cache-aware tiling (affine tiling with halo-aware tile shape)
  — tile sizes chosen so each tile's working set (interior + halo) fits L1/L2
      │
      ▼  [cleanup: licm, cse, canonicalize]
      │
scf.parallel (tiles, multi-core) + scf.for (inner)
      │
      ▼
vector.transfer_read/write (in_bounds, shuffle-reused loads)
      │
      ▼  [cleanup: cse, constant-fold]
      │
LLVM IR
```

---

## 3. MLIR Dialect Lowering Stages

### Stage 1: Input High-Level Stencil IR (`stencil` Dialect) 
represenatiaon of the 2 stencil.apply ops to see the fusion opportunity during the IR transformation.
```mlir
module {
  func.func @heat_diffusion_chain(%arg0: !stencil.field<[?, ?]x f64>,
                                   %arg1: !stencil.field<[?, ?]x f64>,
                                   %arg2: !stencil.field<[?, ?]x f64>) {
    %0 = stencil.load %arg0 : <[?, ?]x f64> -> <[?, ?]x f64>

    // First stencil: 5-point diffusion sum, arg0 -> temp
    %1 = stencil.apply(%a = %0 : !stencil.temp<[?, ?]x f64>) -> !stencil.temp<[?, ?]x f64>
         attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64} {
      %t = stencil.access %a [-1, 0] : (!stencil.temp<[?, ?]x f64>) -> f64
      %b = stencil.access %a [1, 0]  : (!stencil.temp<[?, ?]x f64>) -> f64
      %l = stencil.access %a [0, -1] : (!stencil.temp<[?, ?]x f64>) -> f64
      %r = stencil.access %a [0, 1]  : (!stencil.temp<[?, ?]x f64>) -> f64
      %c = stencil.access %a [0, 0]  : (!stencil.temp<[?, ?]x f64>) -> f64
      %s1 = arith.addf %t, %b : f64
      %s2 = arith.addf %s1, %l : f64
      %s3 = arith.addf %s2, %r : f64
      %s4 = arith.addf %s3, %c : f64
      stencil.return %s4 : f64
    }
    stencil.store %1 to %arg1 : <[?, ?]x f64> to <[?, ?]x f64>

    // Second stencil: consumes arg1's output directly (producer-consumer chain)
    %2 = stencil.load %arg1 : <[?, ?]x f64> -> <[?, ?]x f64>
    %3 = stencil.apply(%a = %2 : !stencil.temp<[?, ?]x f64>) -> !stencil.temp<[?, ?]x f64>
         attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.000000e+00 : f64} {
      %c = stencil.access %a [0, 0] : (!stencil.temp<[?, ?]x f64>) -> f64
      %half = arith.constant 5.000000e-01 : f64
      %s = arith.mulf %c, %half : f64
      stencil.return %s : f64
    }
    stencil.store %3 to %arg2 : <[?, ?]x f64> to <[?, ?]x f64>

    return
  }
}
```

---
### Stage 2: Domain-Specific Structural Analysis
- **Halo/ghost-region derivation**: scan stencil.access offsets across all stencil.apply ops to compute the minimal halo shape and size. The halo defines the interior/boundary split of each apply's iteration domain and later sizes tile halos.
- **Boundary-condition modeling**: a **first-class representation**, not manually-encoded
  conditionals over memory accesses. (The reference stencil-dialect work explicitly flagged
  boundary conditions as unsupported at a high level — this is our opportunity to solve it
  properly rather than inherit the limitation.) Must be rich enough that interior and boundary
  regions can be fused independently (see the fused-boundary limitation under Status).
- **Bounds-mode decision**: compile-time-known bounds vs. dynamic bounds, decided explicitly and
  recorded as an attribute, since it materially affects which downstream optimizations are legal
  (constant-folding of address computations, register pressure reduction) — a known tradeoff
  flagged in the reference stencil-dialect work.
- **Algebraic Simplification**: Try to optimize if anything can be done for algebraic simplification is possible without changing the calculation.

---
### Stage 3: Direct Stencil → `memref` + Affine Lowering (Interior/Boundary Split)

This stage does two things at once, and both matter. First, `!stencil.field<[?, ?]x f64>` / `!stencil.temp<[?, ?]x f64>` — value-like, addressless SSA types — are lowered to `memref<?x?xf64>`, a concrete, addressable buffer with a real (default row-major) layout. Second, that buffer is what the affine loop nest is written against.

Each `stencil.apply` lowers to a pair of loop groups in `affine` over these `memref` operands: one interior nest, and boundary loops for all four edges plus four corners. Every out-of-bounds neighbor is a compile-time `arith.constant 0.0`, never a load, never an `affine.if`. `noalias` is attached to the `memref` arguments at the function boundary since this is a Jacobi-style update (distinct read/write buffers) — a `memref`-level attribute with no `stencil`-level or `tensor`-level equivalent.

**Interior nest for the first stencil (`%arg0 -> %arg1`):**

```mlir
#map_sub1 = affine_map<()[s0] -> (s0 - 1)>
#map_id   = affine_map<()[s0] -> (s0)>

func.func @stencil1_interior(%arg0: memref<?x?xf64> {llvm.noalias},
                              %arg1: memref<?x?xf64> {llvm.noalias}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %arg0, %c0 : memref<?x?xf64>
  %N = memref.dim %arg0, %c1 : memref<?x?xf64>
  %M_minus_1 = affine.apply #map_sub1()[%M]
  %N_minus_1 = affine.apply #map_sub1()[%N]

  affine.for %i = 1 to #map_id()[%M_minus_1] {
    affine.for %j = 1 to #map_id()[%N_minus_1] {
      %top    = affine.load %arg0[%i - 1, %j] : memref<?x?xf64>
      %bottom = affine.load %arg0[%i + 1, %j] : memref<?x?xf64>
      %left   = affine.load %arg0[%i, %j - 1] : memref<?x?xf64>
      %right  = affine.load %arg0[%i, %j + 1] : memref<?x?xf64>
      %center = affine.load %arg0[%i, %j]     : memref<?x?xf64>
      %s1 = arith.addf %top, %bottom : f64
      %s2 = arith.addf %s1, %left : f64
      %s3 = arith.addf %s2, %right : f64
      %s4 = arith.addf %s3, %center : f64
      affine.store %s4, %arg1[%i, %j] : memref<?x?xf64>
    }
  }
  return
}
```

**Boundary loops for the first stencil — all 4 edges + all 4 corners, each substituting `arith.constant 0.0` for the neighbor direction that points off-grid. This boundary computation varies depending on the type of the Stencil OPS specified in the decorator of the stencil DSL. A Structural analysis pass will run where these computation patterns will be elaborated, and neccessary atribute is set for subsequent passes** Edge loops iterate only their strict interior range (e.g. left/right edges run `i = 1 .. M-2`, not `i = 0 .. M-1`), so every corner is written by exactly one dedicated corner block, never by an edge loop:

```mlir
module {
  func.func @stencil1_boundary(%arg0: memref<?x?xf64> {llvm.noalias},
                                %arg1: memref<?x?xf64> {llvm.noalias}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_zero = arith.constant 0.000000e+00 : f64

    %M = memref.dim %arg0, %c0 : memref<?x?xf64>
    %N = memref.dim %arg0, %c1 : memref<?x?xf64>
    %M_minus_1 = affine.apply #map_sub1()[%M]
    %M_minus_2 = affine_map<()[s0] -> (s0 - 2)>()[%M]
    %N_minus_1 = affine.apply #map_sub1()[%N]
    %N_minus_2 = affine_map<()[s0] -> (s0 - 2)>()[%N]

    // -------------------------------------------------------------------
    // TOP EDGE (row 0), j = 1 .. N-2: top neighbor -> 0.0
    // -------------------------------------------------------------------
    affine.for %j = 1 to #map_id()[%N_minus_1] {
      %v_top    = arith.constant 0.000000e+00 : f64                // (-1, j) OOB -> 0.0
      %v_bottom = affine.load %arg0[1, %j]     : memref<?x?xf64>   // real
      %v_left   = affine.load %arg0[0, %j - 1] : memref<?x?xf64>   // real
      %v_right  = affine.load %arg0[0, %j + 1] : memref<?x?xf64>   // real
      %v_center = affine.load %arg0[0, %j]     : memref<?x?xf64>   // real

      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      affine.store %res, %arg1[0, %j] : memref<?x?xf64>
    }

    // -------------------------------------------------------------------
    // BOTTOM EDGE (row M-1), j = 1 .. N-2: bottom neighbor -> 0.0
    // -------------------------------------------------------------------
    affine.for %j = 1 to #map_id()[%N_minus_1] {
      %v_top    = affine.load %arg0[%M_minus_1 - 1, %j] : memref<?x?xf64>  // real
      %v_bottom = arith.constant 0.000000e+00 : f64                        // (M, j) OOB -> 0.0
      %v_left   = affine.load %arg0[%M_minus_1, %j - 1] : memref<?x?xf64>  // real
      %v_right  = affine.load %arg0[%M_minus_1, %j + 1] : memref<?x?xf64>  // real
      %v_center = affine.load %arg0[%M_minus_1, %j]     : memref<?x?xf64>  // real

      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      affine.store %res, %arg1[%M_minus_1, %j] : memref<?x?xf64>
    }

    // -------------------------------------------------------------------
    // LEFT EDGE (col 0), i = 1 .. M-2: left neighbor -> 0.0
    // -------------------------------------------------------------------
    affine.for %i = 1 to #map_id()[%M_minus_1] {
      %v_top    = affine.load %arg0[%i - 1, 0] : memref<?x?xf64>   // real
      %v_bottom = affine.load %arg0[%i + 1, 0] : memref<?x?xf64>   // real
      %v_left   = arith.constant 0.000000e+00 : f64                // (i, -1) OOB -> 0.0
      %v_right  = affine.load %arg0[%i, 1]     : memref<?x?xf64>   // real
      %v_center = affine.load %arg0[%i, 0]     : memref<?x?xf64>   // real

      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      affine.store %res, %arg1[%i, 0] : memref<?x?xf64>
    }

    // -------------------------------------------------------------------
    // RIGHT EDGE (col N-1), i = 1 .. M-2: right neighbor -> 0.0
    // -------------------------------------------------------------------
    affine.for %i = 1 to #map_id()[%M_minus_1] {
      %v_top    = affine.load %arg0[%i - 1, %N_minus_1] : memref<?x?xf64>  // real
      %v_bottom = affine.load %arg0[%i + 1, %N_minus_1] : memref<?x?xf64>  // real
      %v_left   = affine.load %arg0[%i, %N_minus_1 - 1] : memref<?x?xf64>  // real
      %v_right  = arith.constant 0.000000e+00 : f64                        // (i, N) OOB -> 0.0
      %v_center = affine.load %arg0[%i, %N_minus_1]     : memref<?x?xf64>  // real

      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      affine.store %res, %arg1[%i, %N_minus_1] : memref<?x?xf64>
    }

    // -------------------------------------------------------------------
    // 4 CORNERS: exactly 2 real neighbor loads + 2 constant-zero fallbacks
    // each, written once, never touched by any edge loop above.
    // -------------------------------------------------------------------

    // Corner (0, 0): top/left OOB
    {
      %v_top    = arith.constant 0.000000e+00 : f64
      %v_left   = arith.constant 0.000000e+00 : f64
      %v_bottom = memref.load %arg0[%c1, %c0] : memref<?x?xf64>
      %v_right  = memref.load %arg0[%c0, %c1] : memref<?x?xf64>
      %v_center = memref.load %arg0[%c0, %c0] : memref<?x?xf64>
      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      memref.store %res, %arg1[%c0, %c0] : memref<?x?xf64>
    }

    // Corner (0, N-1): top/right OOB
    {
      %v_top    = arith.constant 0.000000e+00 : f64
      %v_right  = arith.constant 0.000000e+00 : f64
      %v_bottom = memref.load %arg0[%c1, %N_minus_1] : memref<?x?xf64>
      %v_left   = memref.load %arg0[%c0, %N_minus_2] : memref<?x?xf64>
      %v_center = memref.load %arg0[%c0, %N_minus_1] : memref<?x?xf64>
      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      memref.store %res, %arg1[%c0, %N_minus_1] : memref<?x?xf64>
    }

    // Corner (M-1, 0): bottom/left OOB
    {
      %v_bottom = arith.constant 0.000000e+00 : f64
      %v_left   = arith.constant 0.000000e+00 : f64
      %v_top    = memref.load %arg0[%M_minus_2, %c0] : memref<?x?xf64>
      %v_right  = memref.load %arg0[%M_minus_1, %c1] : memref<?x?xf64>
      %v_center = memref.load %arg0[%M_minus_1, %c0] : memref<?x?xf64>
      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      memref.store %res, %arg1[%M_minus_1, %c0] : memref<?x?xf64>
    }

    // Corner (M-1, N-1): bottom/right OOB
    {
      %v_bottom = arith.constant 0.000000e+00 : f64
      %v_right  = arith.constant 0.000000e+00 : f64
      %v_top    = memref.load %arg0[%M_minus_2, %N_minus_1] : memref<?x?xf64>
      %v_left   = memref.load %arg0[%M_minus_1, %N_minus_2] : memref<?x?xf64>
      %v_center = memref.load %arg0[%M_minus_1, %N_minus_1] : memref<?x?xf64>
      %t1 = arith.addf %v_top, %v_bottom : f64
      %t2 = arith.addf %t1, %v_left : f64
      %t3 = arith.addf %t2, %v_right : f64
      %res = arith.addf %t3, %v_center : f64
      memref.store %res, %arg1[%M_minus_1, %N_minus_1] : memref<?x?xf64>
    }

    return
  }
}
```

**The second stencil (`%arg1 -> %arg2`, the scale-by-0.5 pass)** lowers the same way, but its body is a single-input pointwise map with no neighbor accesses, so its "boundary" is trivial — the loop already runs the full `0 .. M`/`0 .. N` range with no neighbor access at all:

```mlir
func.func @stencil2_interior(%arg1: memref<?x?xf64> {llvm.noalias},
                              %arg2: memref<?x?xf64> {llvm.noalias}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %half = arith.constant 5.000000e-01 : f64
  %M = memref.dim %arg1, %c0 : memref<?x?xf64>
  %N = memref.dim %arg1, %c1 : memref<?x?xf64>

  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      %v = affine.load %arg1[%i, %j] : memref<?x?xf64>
      %r = arith.mulf %v, %half : f64
      affine.store %r, %arg2[%i, %j] : memref<?x?xf64>
    }
  }
  return
}
```

**Cleanup pass here:** run `canonicalize` + `cse` immediately after this lowering. The direct `stencil` → `affine` conversion tends to emit repeated `affine.apply` computations for the same bound (e.g. `M_minus_1` recomputed per loop) — CSE collapses these back to one value before any analysis pass has to reason about them.

---

### Stage 3: Affine Dependence Analysis & Producer-Consumer Fusion

`stencil1_interior` writes `%arg1`; `stencil2_interior` reads `%arg1` at the *same* index `(i, j)` with no offset (a pointwise consumer, not a neighbor-shifted one). Running `MemRefDependenceCheck` between the store in the first loop nest and the load in the second yields a dependence vector of `(0, 0)` — same iteration, no lag, no reversal — which is exactly the profile `-affine-loop-fusion` accepts for a legal producer-consumer fusion at fusion depth 2 (both loop levels fused).

```mlir
// After -affine-loop-fusion (producer-consumer, depth=2):
affine.for %i = 1 to #map_id()[%M_minus_1] {
  affine.for %j = 1 to #map_id()[%N_minus_1] {
    // fused body: stencil1's 5-point sum feeds directly into stencil2's scale,
    // with the intermediate %arg1 store/load pair eliminated by
    // -affine-scalrep / -affine-loop-fusion's private-memref promotion.
    %top    = affine.load %arg0[%i - 1, %j] : memref<?x?xf64>
    %bottom = affine.load %arg0[%i + 1, %j] : memref<?x?xf64>
    %left   = affine.load %arg0[%i, %j - 1] : memref<?x?xf64>
    %right  = affine.load %arg0[%i, %j + 1] : memref<?x?xf64>
    %center = affine.load %arg0[%i, %j]     : memref<?x?xf64>
    %s1 = arith.addf %top, %bottom : f64
    %s2 = arith.addf %s1, %left : f64
    %s3 = arith.addf %s2, %right : f64
    %s4 = arith.addf %s3, %center : f64
    %half = arith.constant 5.000000e-01 : f64
    %r = arith.mulf %s4, %half : f64
    affine.store %r, %arg2[%i, %j] : memref<?x?xf64>
  }
}
```

This eliminates one full read/write pass over `%arg1` — the intermediate buffer's write and immediate re-read are removed entirely. This is a real memory-bandwidth win.

**NOTE:** Sometime this fusion of fusion may create problem is combined Halo width of both stencils width is not considered while doing the fusion. Here only pointwise operation is there in the 2nd stencils operation so it is not required.

**Cleanup pass here:** `cse` + `constant-fold` again — fusion tends to duplicate constants (`%half` appears once now instead of per-callsite, but only after CSE runs) and can expose redundant loads that a second CSE pass can clean up post-fusion.

---

### Stage 4: Loop Interchange for Spatial/Temporal Cache Reuse

With the loop nest now in a single `affine.for %i { affine.for %j { ... } }` form, interchange is evaluated: does `(i, j)` or `(j, i)` order give better reuse, given that `%arg0` is stored row-major (matching `memref<?x?xf64>`'s default layout)?

- **Current order (`i` outer, `j` inner):** for a fixed `i`, the inner `j` loop walks contiguous memory in `%arg0`'s row `i` (good spatial locality for `center`, `left`, `right`), and simultaneously re-touches rows `i-1` and `i+1` at the same `j` across the inner loop (good temporal locality for `top`/`bottom`, since row `i` becomes row `i-1`/`i+1` of the *next* outer iteration and is likely still cache-resident).
- **Swapped order (`j` outer, `i` inner)** would instead stride through memory by row-length increments in the inner loop — poor spatial locality on this layout.

A cost model will be implemented for the loop interchange for the legal possible permutation if the loop order. And based on that loop order will be selected that is more cost-effective.


**Cleanup pass here:** re-run `canonicalize` if interchange rewrites any affine maps, so tiling in the next stage sees maps in canonical form.

---

### Stage 5: Cache-Aware Tiling (Halo-Aware Tile Sizing)

Tiling is applied to the fused, ordered affine loop nest. Two things distinguish this from a naive tile-size pick:

1. **Tile bounds are clamped** via `affine.min` against `M_minus_1`/`N_minus_1`, so a dynamic $M \times N$ that doesn't divide evenly by the tile size is handled without dropping or overrunning the remainder.
2. **Tile size accounts for the stencil's halo.** A tile of logical size $T_i \times T_j$ actually touches $(T_i + 2) \times (T_j + 2)$ input elements (one extra row/column of neighbors on each side). The chosen $32 \times 64$ tile at `f64` (8 bytes) means each tile's *working set*, input + halo + output, needs to fit comfortably in L2:

   $\text{working set} \approx \big((32+2)\times(64+2) + 32\times64\big) \times 8\ \text{bytes} \approx 34.6\ \text{KB}$

   which fits well inside a typical 256 KB–1 MB L2, leaving headroom for the halo of *adjacent* tiles that may still be resident from a neighboring tile's recent access — the actual mechanism giving the cache hit opportunity the fusion/interchange stages set up.

```mlir
#map_clamp_i = affine_map<(d0)[s0] -> (d0 + 32, s0)>   // min(d0+32, M-1)
#map_clamp_j = affine_map<(d0)[s0] -> (d0 + 64, s0)>   // min(d0+64, N-1)

affine.for %i_outer = 1 to #map_id()[%M_minus_1] step 32 {
  affine.for %j_outer = 1 to #map_id()[%N_minus_1] step 64 {
    %i_hi = affine.min #map_clamp_i(%i_outer)[%M_minus_1]
    %j_hi = affine.min #map_clamp_j(%j_outer)[%N_minus_1]

    affine.for %i = %i_outer to %i_hi {
      affine.for %j = %j_outer to %j_hi {
        // fused stencil1+stencil2 body from Stage 3
        %top    = affine.load %arg0[%i - 1, %j] : memref<?x?xf64>
        %bottom = affine.load %arg0[%i + 1, %j] : memref<?x?xf64>
        %left   = affine.load %arg0[%i, %j - 1] : memref<?x?xf64>
        %right  = affine.load %arg0[%i, %j + 1] : memref<?x?xf64>
        %center = affine.load %arg0[%i, %j]     : memref<?x?xf64>
        %s1 = arith.addf %top, %bottom : f64
        %s2 = arith.addf %s1, %left : f64
        %s3 = arith.addf %s2, %right : f64
        %s4 = arith.addf %s3, %center : f64
        %half = arith.constant 5.000000e-01 : f64
        %r = arith.mulf %s4, %half : f64
        affine.store %r, %arg2[%i, %j] : memref<?x?xf64>
      }
    }
  }
}
```

The outer tile loop then converts to `scf.parallel` for multi-core execution — safe here because tiles only read the untouched `%arg0` input and write disjoint regions of `%arg2`.

**Cleanup pass here:** `licm` matters most at this stage — tiling can leave loop-invariant subexpressions sitting inside the innermost loop; hoist them out before vectorization sees the body. Follow with `cse` and `canonicalize`.

---

### Stage 6: Vectorized Dialect (SIMD Execution with Load Reuse)

The innermost `scf.for %j` loop issues one wide row load per row and derives `left`/`center`/`right` via `vector.extract_strided_slice` instead of three overlapping transfers — since `left`, `center`, and `right` are all reads of the same row, offset by one element, one wide load covers all three instead of three separate re-reads of overlapping memory. Every transfer carries `in_bounds = [true]` since the tiling/clamping in Stage 5 already guarantees the access range is valid for full-width iterations.

```mlir
func.func @heat_diffusion_vectorized(%arg0: memref<?x?xf64> {llvm.noalias},
                                      %arg2: memref<?x?xf64> {llvm.noalias},
                                      %i: index, %j_start: index, %j_end: index) {
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_f64 = arith.constant 0.000000e+00 : f64
  %half = arith.constant 5.000000e-01 : f64

  %i_sub = arith.subi %i, %c1 : index
  %i_add = arith.addi %i, %c1 : index

  scf.for %j = %j_start to %j_end step %c8 {
    %v_top    = vector.transfer_read %arg0[%i_sub, %j], %c0_f64
                  {in_bounds = [true]} : memref<?x?xf64>, vector<8xf64>
    %v_bottom = vector.transfer_read %arg0[%i_add, %j], %c0_f64
                  {in_bounds = [true]} : memref<?x?xf64>, vector<8xf64>

    %j_sub = arith.subi %j, %c1 : index
    %v_row_wide = vector.transfer_read %arg0[%i, %j_sub], %c0_f64
                    {in_bounds = [true]} : memref<?x?xf64>, vector<10xf64>

    %v_left   = vector.extract_strided_slice %v_row_wide
                  {offsets = [0], sizes = [8], strides = [1]} : vector<10xf64> to vector<8xf64>
    %v_center = vector.extract_strided_slice %v_row_wide
                  {offsets = [1], sizes = [8], strides = [1]} : vector<10xf64> to vector<8xf64>
    %v_right  = vector.extract_strided_slice %v_row_wide
                  {offsets = [2], sizes = [8], strides = [1]} : vector<10xf64> to vector<8xf64>

    %t1 = arith.addf %v_top, %v_bottom : vector<8xf64>
    %t2 = arith.addf %t1, %v_left      : vector<8xf64>
    %t3 = arith.addf %t2, %v_right     : vector<8xf64>
    %s4 = arith.addf %t3, %v_center    : vector<8xf64>

    %half_vec = vector.broadcast %half : f64 to vector<8xf64>
    %v_res = arith.mulf %s4, %half_vec : vector<8xf64>

    vector.transfer_write %v_res, %arg2[%i, %j] {in_bounds = [true]} : vector<8xf64>, memref<?x?xf64>
  }
  return
}
```


**Cleanup pass here:** `cse` + `constant-fold` — vectorization tends to reintroduce duplicate `vector.broadcast`s of the same scalar constant if the fused body had multiple pointwise constants.

---

### Stage 7: Convert MLIR IR to LLVMIR

Now the MLIR IR is complemented now it will be lower to the LLVM IR to lower to the target code Generation





