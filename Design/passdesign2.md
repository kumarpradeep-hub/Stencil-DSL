# Stencil DSL — CPU Lowering Pipeline Design

---

## The idea in 7 points

- Turn every stencil into **one loop nest** over the whole grid.
- **Fuse on loops (affine IR)**, only when dependence analysis says it is safe.
- Accept **repeated work** from fusion or temporal blocking only up to a limit **τ** (default 10%).
- Fuse **siblings** (nests reading the same input), so the input is read once.
- Fusion is **consumer-driven**: tile the consumer, compute only the producer part that tile needs; tile size and fusion are decided together.
- **Temporal blocking:** run several time steps on a tile while it is in cache.
- Handle grid edges **at the end**: only tiles that touch an edge copy their input into a padded buffer (**pack boundary tiles only**).

Scope: [Scope](#scope). Examples: [Appendix A](#appendix-a--worked-example-2d-5-point-heat-diffusion) (2D heat, IR after every stage), [Appendix B](#appendix-b--mathematical-rewrites-optional-start-of-s2) (optional math rewrites), [Appendix C](#appendix-c--ir-patterns-by-stage) (IR patterns by stage).

> **Assumption:** row-major arrays (the last axis is contiguous).

---

## Scope

- Any shape given by constant offsets: star, box, asymmetric (upwind).
- Any radius, per side and per axis (`r_lo`, `r_hi`).
- 1D, 2D, 3D.
- `f32`, `f64`; several inputs; variable coefficients (as extra inputs).
- Linear and non-linear bodies (Appendix B's rewrites need linear stencils with constant weights).
- Time loops with any time distance d.
- Boundary rules `constant`, `replicate`, `periodic`, `reflect`, per side and axis.
- Declared in-place updates (e.g. Gauss-Seidel): tiles depend on each other within a step, so S7 runs them serially.
- Dynamic shapes: the tile size comes from the cache sizes; the tile count, τ and interior checks run at run time.
- Convergence loops (`while error > ε`): convergence is checked every b_t steps, or the loop is not blocked.


---

## Pipeline at a glance

```
stencil IR
  │  S0  pick target, load CPU hardware numbers
  │  S1  analyse: halo, boundary rule, shapes, aliasing, layout, time distance
  │  S2  optional math rewrites (fp-mode=relaxed), then
  │      lower to ONE affine loop nest per stencil               (affine)
  │  S3  fuse: pointwise, rolling buffer, siblings (no tiling),
  │      then consumer-driven tile-and-fuse                      (affine)
  │  S4  temporal blocking: overlapped, diamond,
  │      or folded (fp-mode=relaxed)                             (affine)
  │  S5  tile the grid + pack only the boundary tiles            (affine)
  │  S7  run tiles in parallel                                   (omp)
  │  S6  align, unroll-and-jam, vectorize                        (vector)
  ▼  S8  lower to LLVM IR, compile for the target CPU
```

- Cleanup passes (`canonicalize`, `cse`, `licm`) run between stages.
- From S2 to S5 every input is read through a **halo view**. S5 removes the views.

---

## Words used in this document

- **Halo (r):** how far a stencil reads from the centre point (5-point stencil: r = 1).
- **Boundary rule:** the value used when a stencil reads outside the grid: `constant(v)`, `replicate`, `periodic`, `reflect`.
- **Tile:** a rectangular block of the grid processed together.
- **Interior tile:** all its reads, halo included, are inside the grid. **Boundary tile:** some are not.
- **Producer / consumer:** nest P writes X, nest C reads X.
- **Siblings:** two nests that read the same input and do not depend on each other.
- **Footprint:** bytes one tile needs in cache (input window + buffers + output).
- **τ (tau):** the most repeated work accepted (default 10%).
- **b_t:** time steps run per tile in one temporal block.
- **Linear stencil with constant weights:** `Σ w_k · a[offset_k]`, every `w_k` a compile-time constant.
- **fp-mode:** `exact` (default) keeps floating-point operations as written; `relaxed` allows reordering them and fusing multiply + add into FMA.

---

## Stage 0 — Select target

- Record the CPU's numbers on the module: L1/L2/L3 size, vector width, masking support, vector registers, cores, NUMA nodes.
- Later stages read only these numbers.

---

## Stage 1 — Structural analysis (stencil dialect)

- For each input: the **halo** per side and axis, and the **boundary rule** per side and axis.
- Whether shapes are **static or dynamic**.
- The **aliasing contract**.
- The **layout**: which axis is contiguous.
- For fields carried across time steps: how many steps back they are read (**time distance d**).
- Only exact cleanup (constant folding). No reordering of floating-point operations.
- No fusion at this level, fusion is done on loops (S3).

---

## Stage 2 — Lower each stencil to one affine loop nest

- Each `stencil.apply` becomes **one** nest over the **whole** grid: no edge loops, no `if` inside. An apply with several outputs becomes one nest with one store per output.
- **Halo view** (`stencil.halo_view`): each input is read through a memref that is `lo + hi` larger per axis; the extra cells hold the boundary rule's values.
  - Every load stays inside the memref's shape. Affine analysis needs this: with out-of-shape loads, upstream fusion produced wrong results (Appendix A, S3).
  - The view is not a copy. All reads and writes of a buffer go through its view.
- Inputs and outputs become `memref`s; intermediate results become local `memref.alloc`s that fusion can remove.
- **Loop order follows the layout:** the innermost loop walks the contiguous axis.
- **Time-loop buffers:** a field with time distance d gets **d + 1** buffers, indexed `t mod (d + 1)`; **1** buffer if d = 0 or if it reads only its own point. The count is fixed by the stencil.
- Optional math rewrites (`fp-mode=relaxed`) run on the stencil IR just before this lowering: [Appendix B](#appendix-b--mathematical-rewrites-optional-start-of-s2).

---

## Aliasing (applies to all stages)

- Affine analysis assumes different memrefs never share memory.
- **Rule:** two arrays are the same memory **only if they are the same memref value**.
- In-place updates (e.g. Gauss-Seidel) must be declared in the DSL (`inplace=True`); the compiler then uses one memref for reading and writing.
- The entry function checks at run time whether the argument arrays overlap. Option **`on-overlap`**: `copy` (default, copy the input first) or `error` (strict).

---

## Stage 3 — Fusion

**Step 1 — Find candidates**
- Pair: P writes X, C reads X. Siblings: two nests read the same input.
- Fusing only if dependence analysis says it is safe.

**Step 2 — Fuse without tiling** (no repeated work)
- Pointwise: C reads X where P wrote it. Merge the loops; X is removed.
- Rolling buffer: C reads a few nearby rows of X. Keep only those rows.
- Siblings: merge the loops; the merged nest reads the union of their halos.
- Siblings reading the same producer's output are merged before Step 3. Others are merged after it, only if the tile still fits.

**Step 3 — Tile and fuse** (C reads X with a halo)
- Tile C. Per tile, compute P on the tile plus C's halo into a buffer `Xt`, then run C on `Xt`.
- Halo cells are computed by two neighbouring tiles; this repeated work must stay ≤ τ.

**Step 4 — Tile size**
- Try sizes largest first; the inner width is a multiple of the vector width.
- Take the first size with: footprint ≤ 50% of L2, repeated work ≤ τ, tiles ≥ cores. If none fits, do not fuse.
- Record it as `stencil.tile`.

**Step 5 — Grid edges**
- Cells of `Xt` outside the grid take X's boundary rule (what the unfused code reads). S5 fills them.

---

## Stage 4 — Temporal blocking

- A tile runs **b_t time steps** while it is in cache. The time loop becomes T / b_t blocks run in order; it is never unrolled.
- **Overlapped tiles** (default): each tile recomputes a halo that grows by r per step. Tiles are independent. Repeated work ≤ τ.
- **Diamond tiles:** tile along the dependence cone. No repeated work; rows of diamonds run in order. Used when overlapped tiles exceed τ.
- **Folding** (`fp-mode=relaxed`, linear stencil with constant weights): a third option ([Appendix B](#appendix-b--mathematical-rewrites-optional-start-of-s2), Rewrite 3).
- Skip this stage if there is no time loop or d = 0.
- **b_t and tile size are chosen together:** start from `stencil.tile` (or the S3 Step 4 rule), try b_t = 1, 2, 3, … and keep the last that passes τ and the footprint. b_t = 1 means no blocking.
- Boundary tiles apply the boundary rule after every step.
- Fields updated together in one step are blocked together; each keeps its own time buffers.

---

## Stage 5 — Tiling + pack only the boundary tiles

- Use `stencil.tile` where set. Other nests get a size from the S3 Step 4 rule without the τ check (no repeated work without fusion).
- Nests already tiled by S3/S4 get only the interior/boundary split.
- Each tile is classified **once** (using the union halo for merged siblings):
  - **Interior tile:** reads the input directly.
  - **Boundary tile:** `stencil.pack` copies its input window (tile + halo) into a small buffer. Halo cells inside the grid get real values; cells outside get the boundary rule. The same loop body then runs on the buffer.
- Fused buffers (`Xt`, `W`) get their off-grid cells from the boundary rule here, in boundary tiles only.
- `stencil.pack` is lowered to plain loops; no `stencil.halo_view` remains.

---

## Stage 7 — Parallelize

- Tile loops with no dependence between tiles run on all cores (OpenMP). The time/group loop stays sequential. Diamond rows run in order; the tiles in a row run in parallel.
- **One parallel region**, outside the sequential loop; only the work-sharing loop is inside it.
- **Scratch buffers once per thread** (`Xt`, `W`, `S`), at the start of the region; on the stack if small, else on the heap. Never inside the tile loop.
- **Interior and boundary tiles are two static loops**, the first with `nowait`.
- **One tile-to-thread mapping** (static, by tile-rows) for copy-in, compute and copy-out; threads pinned to cores. No dynamic schedule with more than one NUMA node.
- **Copy-in is kept** with more than one NUMA node and a long time loop.

---

## Stage 6 — Vectorize

1. **Align compiler-owned buffers:** 64-byte start; low pad and row pitch rounded up to the vector width. User arrays keep their layout.
2. **Unroll-and-jam one outer axis** (the reuse axis a) by U, and remove repeated loads.
   - 2D: `i`. 3D: picked by the cost model, usually `k` (planes are streamed and reused). 1D: none, so this step is skipped.
   - Loads per output vector: (U + 2·r_a)/U + (loads for the other axes' neighbours).
   - U: the largest of 1, 2, 4, 8 whose live vectors fit in the registers.
   - Gain is measured, not assumed (it saves L1 load instructions, not memory traffic).
3. **Vectorize `j`:** aligned full vectors; masked head and tail.
4. **Neighbour shuffles** (`valignq`) instead of unaligned loads: considered when r_j ≥ 2, used only if measured faster.
5. **Non-temporal stores** for a step's final output when the grid is larger than L3 and the output is not read again soon.
6. **FMA:** `contract` flag only with `fp-mode=relaxed`.
- **Skewed (diamond) tiles** lose alignment: every row gets its own masked head and tail. S4 counts this cost.

---

## Stage 8 — Lower to LLVM

- Reuse `stencil-to-llvm`; `vector` and `omp` are lowered too, and the alias-check entry function is emitted.
- The native compile targets the CPU from S0 (`-mcpu`, `-O3`). Without `-mcpu`, 512-bit vectors are split into SSE ([Appendix A, S8](#after-s8--llvm-everything--llvm--llvm-ir)).

---

## Cost model inputs

- Cache sizes → tile size, buffer sizes, b_t (footprint ≤ 50% of L2).
- L3 size → non-temporal stores.
- Vector width, masking, vector registers → vector width, head/tail handling, unroll factor U.
- Cores → minimum number of tiles. NUMA nodes → static mapping, copy-in kept or not.
- Memory bandwidth, peak FLOP/s → whether temporal blocking pays.
- Sibling outputs, footprint, registers → whether a sibling merge pays.
- τ (default 10%) → limit on repeated work in S3 and S4.
- fp-mode → math rewrites, folding, FMA.

---

## Passes, stage by stage

**S0** · `builtin` attributes
- `stencil-select-target` — records the CPU numbers on the module.

**S1** · `stencil` (attributes only)
- `stencil-inline-kernels` — inlines calls to stencil kernels so fusion sees one function.
- `stencil-infer-halo` — halo per input, side and axis; time distance d for fields carried across steps.
- `stencil-model-boundary-conditions` — boundary rule per input, side and axis.
- `stencil-decide-bounds-mode` — static or dynamic shapes.
- `stencil-infer-layout` — contiguous axis of each array.
- `stencil-alias-contract{on-overlap=copy|error}` — which arguments must not overlap, and the run-time action.

**S2** · `stencil` → `affine` + `memref` + `arith`
- `stencil-group-weights{fp-mode}` — relaxed only: add points that share a weight, multiply once.
- `stencil-split-separable{fp-mode}` — relaxed only: split an exactly separable kernel into 1D passes.
- `stencil-to-affine` — one nest per stencil, inputs through `stencil.halo_view`, loop order from layout, d + 1 time buffers.

**S3** · `affine` + `memref`
- `stencil-affine-fusion` — pointwise, rolling-buffer and sibling fusion.
- `stencil-tile-and-fuse{tau}` — consumer-driven tile-and-fuse; picks the tile size; records `stencil.tile`.

**S4** · `affine` + `memref`
- `stencil-temporal-tile{tau, strategy}` — blocks the time loop (overlapped, diamond, folded); picks b_t with the tile size.

**S5** · `affine` + `memref`
- `stencil-tile-and-pack` — tiles untiled nests; adds `stencil.pack` to boundary tiles.
- `stencil-lower-pack` — lowers `stencil.pack` to copy loops with the boundary rule.
- `stencil-verify-no-halo-view` — stops the pipeline if a halo view is left.

**S7** · `scf` + `omp`
- `affine-parallelize` — marks tile loops with no dependence between iterations.
- `convert-scf-to-openmp` — turns them into OpenMP loops.
- `stencil-omp-schedule` — one parallel region, buffers once per thread, two static loops, one mapping for copy-in/compute/copy-out.

**S6** · `vector` + `scf`
- `stencil-align-buffers` — aligned start, pad and pitch to the vector width.
- `stencil-unroll-jam{factor}` — unroll-and-jam one outer axis (none in 1D) and remove repeated loads (upstream `affine-loop-unroll-jam` and `affine-scalrep` are not enough; Appendix A, S6).
- `stencil-vectorize{shuffle-neighbours, nontemporal}` — vectorize `j`, masked head/tail, optional shuffles, non-temporal final stores, `contract` flag when relaxed.

**S8** · everything → `llvm` → LLVM IR
- `stencil-to-llvm` — lowers all remaining dialects; emits the alias-check entry function.

---

## Decisions needed from the team

1. **τ = 10%** for fusion and temporal blocking.
2. **Aliasing:** arrays are distinct by default; in-place must be declared; overlap is copied at run time.
3. **A time-loop construct is added to the DSL** (it must expose the time distance d).
4. **Fusion is consumer-driven**; tile size and fusion are chosen together.
5. **Fused buffers follow the boundary rule outside the grid**; fusion never changes the result.
6. **fp-mode=exact by default**; math rewrites, folding and FMA only with `relaxed`.

---

## Open questions

The cost model picks among the options below; we need to agree on which it considers.

1. **How far back along a producer chain to fuse:** (a) one producer + one consumer, or (b) walk back while repeated work ≤ τ and the footprint fits (the halo adds up: C tile T, B needs T + 2·r_C, A needs T + 2·r_C + 2·r_B).
2. **One producer, several consumers:** (a) one producer tile covering the union of the consumers' halos (= merge the consumers as siblings first), (b) recompute the producer per consumer, or (c) do not fuse.
3. **Temporal blocking as fusion:** (a) treat step t as producer and t + 1 as consumer, so S3 and S4 share one "grow the tile by the halo" path with b_t as the chain depth, or (b) keep S4 separate.

---

## How we will check it

- **Correctness** against PyTorch: every boundary rule; 1D, 2D, 3D; radius 1–3; fused chains; siblings with different halos; several time steps; overlapping arrays.
- `fp-mode=exact` must match the unoptimized pipeline bit for bit; `relaxed` within a small relative tolerance.
- **Speed** against the current `llvm` pipeline and the memory-bandwidth limit.

---

## Appendix A — Worked example: 2D 5-point heat diffusion

This walks one small program through every stage and shows the IR after each one.

**The program**
- Grid: 4096 × 4096, `f64`, row-major.
- Time steps: T = 100.
- One step: `lap = 5-point Laplacian of u`, then `u_new = u + 0.1 * lap`.
- Boundary rule: `constant(0.0)` on all four sides (zero Dirichlet).
- Example CPU: L1 48 KB, L2 2 MB per core, L3 32 MB, 16 cores on 2 NUMA nodes, AVX-512 (8 × f64). τ = 10%.

**Notes on the IR below**
- `...` means "same as above" and is used to keep the listings short.
- `!F` and `!T` are type aliases for `!stencil.field<[4096,4096]xf64>` and `!stencil.temp<[4096,4096]xf64>`.
- From S4 on, loop bounds like `max(%ti - 1, 0)` stand for `affine.max` / `affine.min` maps.

---

### Input — Stencil IR (from the Python DSL)

```mlir
!F = !stencil.field<[4096,4096]xf64>
!T = !stencil.temp<[4096,4096]xf64>

func.func @heat(%in: !F, %out: !F) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cT = arith.constant 100 : index
  %u0 = stencil.load %in : !F -> !T
  %uT = scf.for %t = %c0 to %cT step %c1 iter_args(%u = %u0) -> (!T) {      // time loop
    // lap = n + s + w + e - 4c
    %lap = stencil.apply(%a = %u : !T) -> !T
             attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.0 : f64} {
      %n = stencil.access %a [-1, 0] : (!T) -> f64
      %s = stencil.access %a [1, 0]  : (!T) -> f64
      %w = stencil.access %a [0, -1] : (!T) -> f64
      %e = stencil.access %a [0, 1]  : (!T) -> f64
      %c = stencil.access %a [0, 0]  : (!T) -> f64
      %m4 = arith.constant -4.0 : f64
      %s1 = arith.addf %n, %s : f64
      %s2 = arith.addf %s1, %w : f64
      %s3 = arith.addf %s2, %e : f64
      %c4 = arith.mulf %c, %m4 : f64
      %l  = arith.addf %s3, %c4 : f64
      stencil.return %l : f64
    }
    // u_new = u + 0.1 * lap   (pointwise)
    %un = stencil.apply(%a = %u : !T, %b = %lap : !T) -> !T {
      %c = stencil.access %a [0, 0] : (!T) -> f64
      %l = stencil.access %b [0, 0] : (!T) -> f64
      %alpha = arith.constant 0.1 : f64
      %d = arith.mulf %l, %alpha : f64
      %r = arith.addf %c, %d : f64
      stencil.return %r : f64
    }
    scf.yield %un : !T
  }
  stencil.store %uT to %out : !T to !F
  return
}
```

---

### After S0 — Select target (`builtin` attributes)

- The module gets the CPU numbers. Nothing else changes.

```mlir
module attributes {stencildsl.target_kind = "llvm",
                   stencildsl.hw_spec = {l1_bytes = 49152, l2_bytes = 2097152, l3_bytes = 33554432,
                                         cores = 16, numa_nodes = 2, simd_f64 = 8, masked_simd = true}} {
  func.func @heat(...) { ...unchanged... }
}
```

---

### After S1 — Structural analysis (`stencil` → `stencil`)

- `lap` reads `%a` one cell in every direction, so its halo is 1 on every side.
- `u_new` reads only the centre, so it has no halo and needs no boundary rule.
- The function records static shapes, row-major layout, and that `%in` and `%out` must not overlap.

```mlir
func.func @heat(%in: !F, %out: !F)
    attributes {stencil.bounds_mode = "static",
                stencil.layout      = {in = 1, out = 1},        // contiguous axis = 1 (row-major)
                stencil.alias       = {in_out = "distinct"},
                stencil.on_overlap  = "copy"} {
  ...
  %lap = stencil.apply(%a = %u : !T) -> !T
           attributes {stencil.halo = {a = {lo = [1, 1], hi = [1, 1]}},
                       stencil.bc   = {a = {kind = "constant", value = 0.0 : f64}}} { ...same body... }
  %un  = stencil.apply(%a = %u : !T, %b = %lap : !T) -> !T
           attributes {stencil.halo = {a = {lo = [0, 0], hi = [0, 0]},
                                       b = {lo = [0, 0], hi = [0, 0]}}} { ...same body... }
  ...
}
```

---

### After S2 — Lower to loops (`stencil` → `affine` + `memref` + `arith`)

- The fields become `memref`s.
- The time loop becomes `affine.for %t` over a **ping-pong buffer** `%U` with 2 slots: step t reads slot `t mod 2` and writes slot `(t+1) mod 2`.
- `%U` is read and written only through its halo view `%Uv`. `%Uv` is 1 cell bigger on each side of `i` and `j`, so grid `(i, j)` is `Uv[.., i+1, j+1]`.
- `lap` becomes its own nest writing a temporary `%LAP`. `u_new` becomes a second nest.
- The loop order is `i` then `j`, because `j` is contiguous (row-major).

```mlir
func.func @heat(%in: memref<4096x4096xf64>, %out: memref<4096x4096xf64>) attributes {...} {
  %m4    = arith.constant -4.0 : f64
  %alpha = arith.constant 0.1 : f64
  %U   = memref.alloc() : memref<2x4096x4096xf64>                    // ping-pong time buffer
  %Uv  = stencil.halo_view %U {lo = [0, 1, 1], hi = [0, 1, 1], bc = constant(0.0)}
           : memref<2x4096x4096xf64> -> memref<2x4098x4098xf64>
  %LAP = memref.alloc() : memref<4096x4096xf64>                      // temp from `lap`

  affine.for %i = 0 to 4096 { affine.for %j = 0 to 4096 {            // copy in: slot 0 = in
    %v = affine.load %in[%i, %j] : memref<4096x4096xf64>
    affine.store %v, %Uv[0, %i + 1, %j + 1] : memref<2x4098x4098xf64> } }

  affine.for %t = 0 to 100 {
    affine.for %i = 0 to 4096 {                                      // nest 1: lap
      affine.for %j = 0 to 4096 {
        %n = affine.load %Uv[%t mod 2, %i,     %j + 1] : memref<2x4098x4098xf64>   // u(i-1, j)
        %s = affine.load %Uv[%t mod 2, %i + 2, %j + 1] : memref<2x4098x4098xf64>   // u(i+1, j)
        %w = affine.load %Uv[%t mod 2, %i + 1, %j    ] : memref<2x4098x4098xf64>   // u(i, j-1)
        %e = affine.load %Uv[%t mod 2, %i + 1, %j + 2] : memref<2x4098x4098xf64>   // u(i, j+1)
        %c = affine.load %Uv[%t mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>   // u(i, j)
        %s1 = arith.addf %n, %s : f64
        %s2 = arith.addf %s1, %w : f64
        %s3 = arith.addf %s2, %e : f64
        %c4 = arith.mulf %c, %m4 : f64
        %l  = arith.addf %s3, %c4 : f64
        affine.store %l, %LAP[%i, %j] : memref<4096x4096xf64>
      }
    }
    affine.for %i = 0 to 4096 {                                      // nest 2: u_new
      affine.for %j = 0 to 4096 {
        %c = affine.load %Uv[%t mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>
        %l = affine.load %LAP[%i, %j] : memref<4096x4096xf64>
        %d = arith.mulf %l, %alpha : f64
        %r = arith.addf %c, %d : f64
        affine.store %r, %Uv[(%t + 1) mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>
      }
    }
  }

  affine.for %i = 0 to 4096 { affine.for %j = 0 to 4096 {            // copy out: 100 is even → slot 0
    %v = affine.load %Uv[0, %i + 1, %j + 1] : memref<2x4098x4098xf64>
    affine.store %v, %out[%i, %j] : memref<4096x4096xf64> } }
  memref.dealloc %LAP : memref<4096x4096xf64>
  memref.dealloc %U : memref<2x4096x4096xf64>
  return
}
```

---

### After S3 — Fusion (`affine` → `affine`)

- Dependence analysis: nest 2 reads `%LAP[i, j]` exactly where nest 1 wrote it, so the distance is **(0, 0)**. This is **pointwise** fusion (S3 Step 2): no extra work and no tile needed.
- So S3 Step 3 (tile-and-fuse) is not used here, and no `stencil.tile` is set. The tile size is picked in S4, together with b_t.
- The two nests merge into one. `%LAP` becomes a scalar and its `alloc` is removed.
- The two loads of the centre `u(i, j)` become one (CSE).
- **No siblings here:** there is one field `u`, and the two nests are a producer→consumer pair, not two readers of the same input.
- The copy-in and copy-out nests are **not** fused with the time loop: the time loop reads neighbours from the previous step, so that fusion is illegal.

```mlir
  affine.for %t = 0 to 100 {
    affine.for %i = 0 to 4096 {
      affine.for %j = 0 to 4096 {
        %n = affine.load %Uv[%t mod 2, %i,     %j + 1] : memref<2x4098x4098xf64>
        %s = affine.load %Uv[%t mod 2, %i + 2, %j + 1] : memref<2x4098x4098xf64>
        %w = affine.load %Uv[%t mod 2, %i + 1, %j    ] : memref<2x4098x4098xf64>
        %e = affine.load %Uv[%t mod 2, %i + 1, %j + 2] : memref<2x4098x4098xf64>
        %c = affine.load %Uv[%t mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>
        %l = ...n + s + w + e - 4c...                                // %LAP is now just this value
        %d = arith.mulf %l, %alpha : f64
        %r = arith.addf %c, %d : f64
        affine.store %r, %Uv[(%t + 1) mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>
      }
    }
  }
```

**What we checked with upstream `mlir-opt -affine-loop-fusion` (and why S2 uses halo views)**
- On the S2 IR written **with** halo views (all indices inside the memref's shape), upstream fusion gave exactly the result above. Running it gave the same numbers as the unfused program.
- On the same IR written **without** halo views (loads like `U[t mod 2, i - 1, j]` that go outside the memref at `i = 0`), upstream fusion moved the time loop **inside** the `i, j` loops. That is illegal. Run at 2048 × 2048 with 4 steps, the checksum was **7.14e12 instead of 2.05e13**.
- This is why S2 puts the halo in the memref type, and why our S3 pass does its own legality check instead of trusting upstream alone.

---

### After S4 — Temporal blocking (`affine` → `affine`)

- **Choice:** overlapped tiles, **b_t = 2** steps per tile, tile **128 × 256** (i × j).
  - Extra work: step 1 computes 130 × 258 = 33,540 points instead of 32,768. That is 772 extra of 65,536, i.e. **1.2% ≤ τ (10%)**.
  - Cache: input window 132 × 260, W buffer 130 × 258, and output 128 × 256, all `f64`. Together that is **about 805 KB ≤ 1 MB** (half of L2).
  - b_t = 3 would need a second intermediate buffer: 134·262 + 132·260 + 130·258 + 128·256 f64 ≈ 1.09 MB > 1 MB, so b_t = 2.
  - The inner width 256 is a multiple of the cache line (8 × `f64`), so every row starts aligned.
  - Tiles: 32 × 16 = **512**, more than the 16 cores.
- 100 steps become **50 groups** of 2 steps.
- Each tile:
  - step 1 computes the tile **plus a 1-cell ring** into a small buffer `%W`
  - step 2 computes the tile itself from `%W`
- **Slots change:** group g reads slot `g mod 2` and writes slot `(g+1) mod 2`, so no tile ever writes a slot that another tile is still reading. After 50 groups the result is in slot 0.
- Step 1 only computes points **inside the grid**. `%W`'s cells outside the grid must hold the boundary value; S5 fills them.

```mlir
  affine.for %g = 0 to 50 {                                          // 50 groups × 2 steps
    affine.for %ti = 0 to 4096 step 128 {
      affine.for %tj = 0 to 4096 step 256 {                          // tiles: independent
        %W = memref.alloca() : memref<130x258xf64>                   // step-1 result, tile + 1-cell ring
        // step 1: rows ti-1 .. ti+128, cols tj-1 .. tj+256, clamped to the grid
        affine.for %i = max(%ti - 1, 0) to min(%ti + 129, 4096) {
          affine.for %j = max(%tj - 1, 0) to min(%tj + 257, 4096) {
            %n = affine.load %Uv[%g mod 2, %i,     %j + 1] ...  %e, %w, %s, %c likewise
            %r = ...c + 0.1 * (n + s + w + e - 4c)...
            affine.store %r, %W[%i - %ti + 1, %j - %tj + 1] : memref<130x258xf64>
          }
        }
        // step 2: the tile only, reads W, writes the other slot
        affine.for %i = %ti to %ti + 128 {
          affine.for %j = %tj to %tj + 256 {
            %n = affine.load %W[%i - %ti,     %j - %tj + 1] : memref<130x258xf64>
            %s = affine.load %W[%i - %ti + 2, %j - %tj + 1] : memref<130x258xf64>
            %w = affine.load %W[%i - %ti + 1, %j - %tj    ] : memref<130x258xf64>
            %e = affine.load %W[%i - %ti + 1, %j - %tj + 2] : memref<130x258xf64>
            %c = affine.load %W[%i - %ti + 1, %j - %tj + 1] : memref<130x258xf64>
            %r = ...c + 0.1 * (n + s + w + e - 4c)...
            affine.store %r, %Uv[(%g + 1) mod 2, %i + 1, %j + 1] : memref<2x4098x4098xf64>
          }
        }
      } {stencil.halo_bufs = {W = constant(0.0)}}                    // W's off-grid cells = 0.0 (filled in S5)
    }
  }
```

---

### After S5 — Tiling + pack only boundary tiles (`affine` + `stencil.pack` → `affine`)

- The tile size is already fixed by S4 (128 × 256).
- **Interior tile:** its input window (the tile plus 2 cells on each side) is fully inside the grid, i.e. `ti ≥ 2`, `ti + 130 ≤ 4096`, `tj ≥ 2`, `tj + 258 ≤ 4096`.
  - That gives 30 of 32 row-tiles and 14 of 16 column-tiles, so **420 interior tiles**.
  - They read `%U` directly. The halo view is gone: `Uv[s, i, j + 1]` becomes `U[s, i - 1, j]`.
- **Boundary tile:** the other **92 tiles (18%)**.
  - `stencil.pack` copies the 132 × 260 window into `%S`, writing `0.0` for cells outside the grid.
  - `%W` is set to `0.0` first, so its off-grid ring holds the boundary value.
  - Step 1 then reads `%S`.
- Step 2 is the same for both kinds of tile.

```mlir
  %zero = arith.constant 0.0 : f64
  affine.for %g = 0 to 50 {
    affine.for %ti = 0 to 4096 step 128 {
      affine.for %tj = 0 to 4096 step 256 {
        %S = memref.alloca() : memref<132x260xf64>                   // packed window (boundary tiles)
        %W = memref.alloca() : memref<130x258xf64>
        affine.if #interior(%ti, %tj) {
          // step 1, interior: read U directly (clamp is a no-op here)
          affine.for %i = %ti - 1 to %ti + 129 { affine.for %j = %tj - 1 to %tj + 257 {
            %n = affine.load %U[%g mod 2, %i - 1, %j] ...            // real array, no halo view
            ...
            affine.store %r, %W[%i - %ti + 1, %j - %tj + 1] } }
        } else {
          // step 1, boundary: pack, then read S
          stencil.pack %U[%g mod 2, %ti - 2, %tj - 2] into %S {bc = constant(0.0)}
          linalg.fill ins(%zero : f64) outs(%W : memref<130x258xf64>)            // off-grid ring of W = 0.0
          affine.for %i = max(%ti - 1, 0) to min(%ti + 129, 4096) {
            affine.for %j = max(%tj - 1, 0) to min(%tj + 257, 4096) {
              %n = affine.load %S[%i - %ti + 1, %j - %tj + 2] ...    // grid (i-1, j)
              %s = affine.load %S[%i - %ti + 3, %j - %tj + 2] ...    // grid (i+1, j)
              %w = affine.load %S[%i - %ti + 2, %j - %tj + 1] ...    // grid (i, j-1)
              %e = affine.load %S[%i - %ti + 2, %j - %tj + 3] ...    // grid (i, j+1)
              %c = affine.load %S[%i - %ti + 2, %j - %tj + 2] ...    // grid (i, j)
              ...
              affine.store %r, %W[%i - %ti + 1, %j - %tj + 1] } }
        }
        // step 2 (both kinds): unchanged from S4, but writes %U directly
        ... affine.store %r, %U[(%g + 1) mod 2, %i, %j] ...
      }
    }
  }
```

`stencil-lower-pack` then turns the pack into plain loops. The per-element check only runs inside boundary tiles:

```mlir
  affine.for %x = 0 to 132 { affine.for %y = 0 to 260 {
    %v = affine.if #in_grid(%ti + %x - 2, %tj + %y - 2) -> f64 {
      %u = affine.load %U[%g mod 2, %ti + %x - 2, %tj + %y - 2] : memref<2x4096x4096xf64>
      affine.yield %u : f64
    } else {
      affine.yield %zero : f64                                      // constant(0.0)
    }
    affine.store %v, %S[%x, %y] : memref<132x260xf64> } }
```

---

### After S7 — Parallelize (`affine` → `scf.parallel` → `omp`)

Machine: 16 cores on 2 NUMA nodes (threads 0–7 on node 0, 8–15 on node 1). 512 tiles = 32 tile-rows × 16 tile-columns; 420 interior, 92 boundary (tile-rows 0 and 31, tile-columns 0 and 15).

**1. One parallel region**
- The tile loops `(ti, tj)` are parallel; the group loop `g` is not (group g+1 reads what g wrote).
- `omp.parallel` is placed outside `g`, so the thread team starts **once instead of 50 times**. Every thread runs the `g` loop; the barrier at the end of each work-sharing loop keeps them in step. The slot is `g mod 2`, computed by each thread, so no shared state needs updating.
- Advantage: small here (a few µs per start against about a ms per group), but large for small grids with many steps.

**2. Buffers once per thread**
- `%S` (132 × 260) and `%W` (130 × 258) take about 530 KB.
- Inside the tile loop, each tile would take new stack that is not freed until the region ends (tested: the lowered `alloca` sits in the loop with no stack restore). 32 tiles per thread → **about 17 MB**, above the usual 4 MB OpenMP thread stack: a crash.
- Allocated once at the top of `omp.parallel`: **530 KB per thread**, reused by every tile.

**3. Two static loops**
- One static split of all 512 tiles gives each thread 2 tile-rows. Threads 0 and 15 then get 18 boundary tiles, the others 4. A boundary tile costs an extra (1 + x) with x ≈ 0.3–1 for the pack, so the others wait at the barrier:

| x (pack cost) | slowest thread | even split | time lost |
|---|---|---|---|
| 0.5 | 41 | 34.9 | ≈ 15% |
| 1.0 | 50 | 37.8 | ≈ 24% |

- Round-robin (`chunk = 1`) is worse here: a row has 16 tiles, so thread k gets tile-column k, and threads 0 and 15 get only boundary tiles.
- Two loops: 420 interior tiles (26–27 per thread) with `nowait`, then 92 boundary tiles (5–6 per thread). Every thread gets an even share of both. The `nowait` is safe: the loops write different tiles and read only the old slot.

**4. Same mapping for copy-in and compute (NUMA)**
- A page is placed on the node of the thread that **first writes** it. The copy-in first writes slot 0; group 0's step 2 first writes slot 1.
- The copy-in uses the same two loops over the same tiles. With the static split, threads 0–7 (node 0) get interior tile-rows 1–15 and threads 8–15 (node 1) get 16–30, both for copy-in and compute, so almost every read is local. Only some boundary tiles cross nodes (under 92 of 512).
- Threads are pinned (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`), so the mapping holds.
- Without this, a one-thread copy-in puts slot 0 (128 MB) on node 0, and node 1's cores read it remotely: up to about 2× slower for a memory-bound stencil.
- The copy-in is kept: the input array was written by one Python thread and sits on one node; the copy spreads it over both.

```mlir
  omp.parallel {
    %S = memref.alloca() : memref<132x260xf64>                   // once per thread
    %W = memref.alloca() : memref<130x258xf64>
    scf.for %g = %c0 to %c50 step %c1 {                          // every thread, same g
      omp.wsloop nowait {                                        // 420 interior tiles, static
        omp.loop_nest (%a, %b) : index = (%c1, %c1) to (%c31, %c15) step (%c1, %c1) collapse(2) {
          // ti = a * 128, tj = b * 256: read U directly, step 1 → W, step 2 → U
          omp.yield
        }
      }
      omp.wsloop {                                               // 92 boundary tiles, static
        omp.loop_nest (%k) : index = (%c0) to (%c92) step (%c1) {
          // (ti, tj) = k-th edge tile: pack into S, step 1 → W, step 2 → U
          omp.yield
        }
      }                                                          // barrier: group g done
    }
    omp.terminator
  }
  // copy-in and copy-out: the same two loops over the same tiles
```

---

### After S6 — Vectorize (`scf` + `memref` → `vector`)

**Step 1 — Align**
- `%W` becomes `130 × 272` (was 130 × 258): grid column j sits at W column `j - tj + 8`, so column `tj` starts on a cache line and the pitch is a multiple of 8. `%S` becomes `132 × 272` the same way. Both get `alignment = 64`; `%U` (pitch 4096) only gets `alignment = 64`.
- Why: a 512-bit access that is not aligned always touches 2 cache lines. Stores suffer most.
- Footprint again: 132·272 + 130·272 + 128·256 f64 ≈ **832 KB ≤ 1 MB**, so the tile size still holds.

**Step 2 — Unroll-and-jam, U = 4** (2D, r_i = r_j = 1)
- Loads per output vector: (4 + 2)/4 + 2 = **3.5 instead of 5** (14 loads per 4 rows).
- Live vectors: 6 rows + 8 west/east + 4 results + 2 constants = **20 ≤ 32**. U = 8 would need about 36.
- Step 2 has 128 rows (32 × 4), no leftover. Step 1 has 130 rows: 32 jammed groups + 2 normal rows.
- Measure it: `%W` (283 KB) sits in L2, not L1 (48 KB). The 3 rows each output row reuses (about 6 KB) already fit in L1 without the jam, so the jam saves load instructions, not L2 traffic. If L2 bandwidth is the limit, the gain is small.
- Radius 4 instead would give (4 + 8)/4 + 8 = 11 loads per output and about 50 live vectors at U = 4, so U drops to 1–2 there.
- Why a custom pass (tested): upstream `affine-loop-unroll-jam` only transforms the first op in a function, and `affine-scalrep` removed the repeated loads (20 → 14) only when the read buffer was local; with a function argument MLIR assumed it could overlap the output and kept all 20. `stencil-unroll-jam` does both, using the Aliasing rule.

**Step 3 — Vectorize, 8 × `f64`**
- Step 2: 256 columns from `tj` = **32 aligned vectors**, no head or tail.
- Step 1: 258 columns = **1-column masked head** + 32 aligned vectors + **1-column masked tail**.
- The constants are broadcast once, outside the loops.

**Step 4 — Shuffles:** off (r_j = 1: only 2 unaligned loads per row to replace).

**Step 5 — Non-temporal stores**
- Each slot of `%U` is 128 MB, larger than L3, and the slot step 2 writes is read again only in the next group, after the whole grid has been swept: by then it has left the cache. So step 2 stores to `%U` are non-temporal.
- Per point: 8 B read + 8 B written + 8 B read-before-write = 24 B → **16 B, about 33% less memory traffic**.
- Step 1 writes `%W`, which step 2 reads right away, so those stay normal stores.

**Step 6 — FMA:** off, because this example uses `fp-mode=exact`. With `relaxed`, the heat body goes from 7 to 5 vector instructions (tested with `llc -mcpu=skylake-avx512`).

**Skewed tiles:** not used here (S4 chose overlapped tiles), so every row keeps the same aligned start.

```mlir
  %m4v    = vector.broadcast %m4    : f64 to vector<8xf64>
  %alphav = vector.broadcast %alpha : f64 to vector<8xf64>
  %W = memref.alloca() {alignment = 64} : memref<130x272xf64>       // grid col j → W col j - tj + 8
  ...
  scf.for %i = %ti to %ti_end step %c4 {                             // step 2: 4 rows per iteration
    scf.for %j = %tj to %tj_end step %c8 {                           // 32 aligned vectors
      // rows wi-1 .. wi+4, each loaded ONCE at the aligned column wj
      %r0 = vector.load %W[%wi_m1, %wj] : memref<130x272xf64>, vector<8xf64>   // n of row i
      %r1 = vector.load %W[%wi,    %wj] : memref<130x272xf64>, vector<8xf64>   // c of i,   n of i+1
      %r2 = vector.load %W[%wi_p1, %wj] : memref<130x272xf64>, vector<8xf64>   // s of i,   c of i+1, n of i+2
      %r3 = vector.load %W[%wi_p2, %wj] : memref<130x272xf64>, vector<8xf64>   // s of i+1, c of i+2, n of i+3
      %r4 = vector.load %W[%wi_p3, %wj] : memref<130x272xf64>, vector<8xf64>   // s of i+2, c of i+3
      %r5 = vector.load %W[%wi_p4, %wj] : memref<130x272xf64>, vector<8xf64>   // s of i+3
      // west / east: 2 unaligned loads per output row
      %w0 = vector.load %W[%wi, %wj_m1] : memref<130x272xf64>, vector<8xf64>
      %e0 = vector.load %W[%wi, %wj_p1] : memref<130x272xf64>, vector<8xf64>
      ...                                                            // %w1/%e1, %w2/%e2, %w3/%e3
      // row i  (n = r0, s = r2, c = r1)
      %s1 = arith.addf %r0, %r2 : vector<8xf64>
      %s2 = arith.addf %s1, %w0 : vector<8xf64>
      %s3 = arith.addf %s2, %e0 : vector<8xf64>
      %c4 = arith.mulf %r1, %m4v : vector<8xf64>
      %l  = arith.addf %s3, %c4 : vector<8xf64>
      %d  = arith.mulf %l, %alphav : vector<8xf64>
      %o0 = arith.addf %r1, %d : vector<8xf64>
      vector.store %o0, %U[%slot_out, %i, %j] {nontemporal = true} : memref<2x4096x4096xf64>, vector<8xf64>   // aligned, non-temporal
      ...                                                            // rows i+1, i+2, i+3 the same way
    }
  }
  // %wi = i - ti + 1, %wj = j - tj + 8 (a multiple of 8), %wi_m1 = wi - 1, ... (computed with affine.apply)

  // step 1, head column j = tj - 1 (the tail j = tj + 256 is the same): one masked vector, 1 lane on
  %m1 = vector.create_mask %c1 : vector<8xi1>
  %hc = vector.maskedload %U[%slot_in, %i, %j_head], %m1, %zerov : memref<2x4096x4096xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
  ...
  vector.maskedstore %W[%wi, %wj_head], %m1, %hr : memref<130x272xf64>, vector<8xi1>, vector<8xf64>
```

---

### After S8 — LLVM (`everything` → `llvm` → LLVM IR)

- Loops become branches, and memrefs become plain pointers (bare-pointer ABI).
- Vector ops become LLVM vector instructions.
- `omp` ops are turned into OpenMP runtime calls when translating to LLVM IR.
- The entry wrapper with the overlap check is added.
- `llc` gets `-mcpu` from S0 and `-O3`. Tested without `-mcpu`: a row kernel written with `vector<8xf64>` became 27 128-bit SSE instructions instead of 5 512-bit ones. The current test harness (`test/python/correctness/jit_compile.py`) passes no `-mcpu` yet.

```mlir
// llvm dialect (inner loop body)
%p  = llvm.getelementptr %W_base[%off_n] : (!llvm.ptr, i64) -> !llvm.ptr, f64
%n  = llvm.load %p {alignment = 8 : i64} : !llvm.ptr -> vector<8xf64>
...
%s1 = llvm.fadd %n, %s : vector<8xf64>
...
llvm.store %r, %q {alignment = 64 : i64, nontemporal} : vector<8xf64>, !llvm.ptr   // aligned, non-temporal (S6 steps 1, 5)
```

```llvm
; LLVM IR after mlir-translate
define i32 @heat_entry(ptr %in, ptr %out) {
  ; overlap check: in < out + 128 MiB  &&  out < in + 128 MiB
  br i1 %overlap, label %copy_path, label %fast_path
fast_path:
  call void @heat(ptr noalias %in, ptr noalias %out)
  ret i32 0
  ...
}
define internal void @heat..omp_par(...) {           ; body of one OpenMP thread
  %n = load <8 x double>, ptr %p, align 8
  %s1 = fadd <8 x double> %n, %s
  ...
  store <8 x double> %r, ptr %q, align 64, !nontemporal !0
}
; in @heat:  call void @__kmpc_fork_call(ptr @loc, i32 N, ptr @heat..omp_par, ...)
```

---

## Appendix B — Mathematical rewrites (optional, start of S2)

**When they apply**
- Only on **linear stencils with constant weights**: the result is a weighted sum `Σ w_k · a[offset_k]` where every weight is a compile-time constant.
- Only when the option **`fp-mode=relaxed`** is set. The default is **`fp-mode=exact`**, which turns all rewrites here off.
  - Every rewrite changes the order of the additions, so the last bits of the result can change (floating-point addition is not associative).
  - `fp-mode` is separate from `on-overlap` (strict aliasing, see Aliasing).
- They run on the `stencil` dialect, **before** `stencil-to-affine`, so the rest of the pipeline sees an ordinary stencil.

**Rewrite 1 — Group equal weights**
- First write the body as a weighted sum, folding constants: `c + 0.1·(n+s+w+e−4c)` becomes `0.6·c + 0.1·n + 0.1·s + 0.1·w + 0.1·e`.
- Then add the points that share a weight, and multiply **once per distinct weight**.
- Symmetric stencils have few distinct weights, so the saving grows with the radius:

| stencil | points | distinct weights | ops before | ops after (with FMA) |
|---|---|---|---|---|
| 2D heat, r = 1 | 5 | 2 | 7 | 6 (5) |
| 2D star, r = 4 | 17 | 5 | 33 | 21 (17) |

```mlir
%v = stencil.apply (%a = %u) {         // n, s, w, e, c = accesses at (-1,0), (1,0), (0,-1), (0,1), (0,0)
  %v = c + 0.1*(n + s + w + e - 4.0*c)
}
// after
%v = stencil.apply (%a = %u) {
  %v = 0.6*c + 0.1*(n + s + w + e)     // 1 multiply per distinct weight
}
```

**Rewrite 2 — Split a separable kernel into two 1D passes**
- Write the 2D weights as a matrix W. If **W = u · vᵀ** (rank 1), the stencil is a pass along `j` with weights v, then a pass along `i` with weights u.
- Cost per point drops from (2r+1)² to 2·(2r+1) multiply-adds: a 7×7 box goes from **49 to 14**.
- Rank k > 1 (a sum of k such pairs) is used only if k·2·(2r+1) is clearly below (2r+1)².
- **Only exact factorizations are used.** The check is done in exact rational arithmetic, and u, v must be exactly representable in the element type. No approximate (SVD) factorizations.
- The split creates an intermediate `h`. S3 then fuses the two passes like any producer→consumer pair (rolling buffer or tile-and-fuse), so `h` never becomes a full grid.
- **Boundary rule of `h`:** it has the same kind as the input's rule, so no new rule is needed (Decision 5).
  - `replicate`, `periodic`, `reflect`: unchanged. Applying a 1D pass along `j` does not change how rows repeat along `i`.
  - `constant(c)`: becomes `constant(c · Σ v)`.
  - This holds because S1 records the boundary rule per axis and per side.

```mlir
%r = stencil.apply (%a = %in) {        // 3×3 Gaussian, W = [[1,2,1],[2,4,2],[1,2,1]] / 16
  ... 9 accesses, 9 multiplies, 8 adds ...
}
// after:  W = u · vᵀ with u = v = [1, 2, 1] / 4  (exact in binary)
%h = stencil.apply (%a = %in) {        // pass along j
  %x = 0.25*(a[0,-1] + a[0,1]) + 0.5*a[0,0]
}
%r = stencil.apply (%b = %h) {         // pass along i; h keeps the input's boundary rule
  %y = 0.25*(b[-1,0] + b[1,0]) + 0.5*b[0,0]
}
```

**Rewrite 3 — Temporal folding (an alternative to S4, chosen by the cost model)**
- For a linear stencil with constant weights, **k time steps equal one stencil** whose weights are the kernel convolved with itself k times. Its radius is k·r.
- Its many points have few distinct weights, so Rewrite 1 applies to it. For 2D heat:

| steps folded | points | distinct weights | rank |
|---|---|---|---|
| 1 | 5 | 2 | 2 |
| 2 | 13 | 4 | 3 |
| 3 | 25 | 6 | 4 |

- **What it saves:** one pass over the grid covers k steps, with **no intermediate buffer** (no `%W`) and no repeated halo work. For heat, the arithmetic stays about the same (≈ 13 ops per 2 steps against ≈ 12); the gain is memory traffic.
- **Where it is exact:** only in the interior. The boundary rule is applied after every step, and a folded kernel skips those steps. So:
  - interior tiles run the folded kernel
  - boundary tiles run the k steps one by one (the interior/boundary split of S5)
  - a tile counts as interior only if its whole k·r halo is inside the grid
- **Conditions:** the same weights at every step, and no other stencil between the steps.
- It is a third choice in S4 next to overlapped and diamond tiles. The cost model compares them on memory traffic and footprint. Folding has no `%W`, so it is not limited by the buffer that stops overlapped tiles at b_t = 2 in Appendix A.

**Later (not in the first version)**
- **Partial-sum reuse between neighbouring points:** point j+1 reuses sums already computed for point j (for example a column sum), together with unroll-and-jam in S6.
- **FFT:** with `periodic` boundaries and constant weights, T steps become one pointwise multiply in Fourier space, O(N log N) whatever T is. Not useful with other boundary rules.

**When it pays off**
- Rewrites 1 and 2 cut arithmetic, so they help stencils where compute is the limit: **large radius or box** stencils, or any stencil after S4 has made it cache-resident.
- A small stencil such as 2D heat does about 0.4 flops per byte, far below what the CPU needs to be compute-bound. It is limited by memory, so only Rewrite 3 (or S4) helps it much.
- ConvStencil's approach (turning the stencil into a matrix multiply) is not used on the CPU: it adds zero-padded work and only pays off with a matrix unit, and there is no f64 matrix unit on our target (AMD Zen 5 has none; Intel AMX has no f64).

---

## Appendix C — IR patterns by stage

Generic before/after IR for each stage. Appendix A shows the same stages on a full example.

**S0 — Select target**
```mlir
module { ... }
// after
module attributes {stencildsl.target_kind = "llvm",
                   stencildsl.hw_spec = {l1 = 48KB, l2 = 2MB, l3 = 32MB, simd_f64 = 8,
                                         vregs = 32, cores = 64, numa_nodes = 2, ...}} { ... }
```

**S1 — Structural analysis** (only attributes are added)
```mlir
%r = stencil.apply (%a = %x) -> !stencil.temp<..> { ...body... }
// after
%r = stencil.apply (%a = %x) -> !stencil.temp<..>
       attributes {stencil.halo = {a: lo = [1, 1], hi = [1, 1]},
                   stencil.bc   = {a: constant(0.0)}} { ...body... }
```

**S2 — One nest per stencil, halo view**
```mlir
%t = stencil.apply (%a = %in) {
  %n = stencil.access %a[-1, 0]
  %c = stencil.access %a[0, 0]
  ...
}
// after
%Av = stencil.halo_view %A {lo = [1, 1], hi = [1, 1], bc = constant(0.0)}
        : memref<N0xN1xf64> -> memref<(N0+2)x(N1+2)xf64>   // grid (i, j) = Av[i+1, j+1]
affine.for %i = 0 to %N0 {
  affine.for %j = 0 to %N1 {
    %n = affine.load %Av[%i, %j + 1]       // = A[i-1, j]; at i = 0 a halo cell (boundary rule)
    %c = affine.load %Av[%i + 1, %j + 1]   // = A[i, j]
    ...
    affine.store %v, %OUT[%i, %j]
  }
}
```

**S2 — Time-loop buffers**

| step t+1 reads | example | buffers |
|---|---|---|
| no earlier step (d = 0) | each step starts from a fixed input | 1 |
| step t, same point only | `u = 0.99 * u` | 1 (in place) |
| step t with neighbours (d = 1) | 2D heat | 2 (`t mod 2`) |
| steps t and t-1 (d = 2) | wave equation | 3 (`t mod 3`) |

```mlir
affine.for %t = 0 to %T {                                 // d = 1: two slots
  affine.for %i ... affine.for %j ... {
    %x = affine.load  %Uv[%t mod 2, %i, %j + 1]           // U[old step, i-1, j]
    affine.store %v, %Uv[(%t + 1) mod 2, %i + 1, %j + 1]  // U[new step, i, j]
  }
}
```

**Aliasing — entry function**
```mlir
func.func @k(%in: memref<?x?xf64>, %out: memref<?x?xf64>)
// after
func.func @k(%in: memref<?x?xf64> {llvm.noalias}, %out: memref<?x?xf64> {llvm.noalias})
func.func @k_entry(%in, %out) -> i32 {
  if overlap(%in, %out) {
    // on-overlap = copy (default): copy %in, then run
    %tmp = copy(%in); call @k(%tmp, %out); return 0
    // on-overlap = error: return 1  → Python raises StencilAliasError
  }
  call @k(%in, %out); return 0
}
```

**S3 — Pointwise and sibling fusion**
```mlir
affine.for %i, %j { ... affine.store %a, %X[%i, %j] }             // P
affine.for %i, %j { %x = affine.load %X[%i, %j] ... }             // C (pointwise)
// after: one loop, X is gone
affine.for %i, %j { %a = ... ; ... use %a directly ... }

affine.for %i, %j { ... %Uv[%i + 1, %j] ... store %DX }           // sibling 1, reads U[i, j-1]
affine.for %i, %j { ... %Uv[%i, %j + 1] ... store %DY }           // sibling 2, reads U[i-1, j]
// after: one loop, U read once
affine.for %i, %j { ... %Uv[%i + 1, %j] ... store %DX ;  ... %Uv[%i, %j + 1] ... store %DY }
```

**S3 — Tile and fuse** (C reads X with r = 1)
```mlir
affine.for %i, %j { ... affine.store %a, %Xv[...] }                      // P
affine.for %i, %j { %x = affine.load %Xv[%i, %j + 1] ... }               // C, reads X[i-1, j]
// after
affine.for %ti, %tj step (Ti, Tj) {                                      // tiles of C
  %Xt = memref.alloca() : memref<(Ti+2)x(Tj+2)xf64>                      // C's tile + halo
  affine.for %i, %j in (C tile + halo) ∩ grid { ... affine.store %a, %Xt[...] }  // P
  // off-grid cells of %Xt: boundary rule of X (filled in S5)
  affine.for %i, %j in C tile { %x = affine.load %Xt[...] ... }          // C
} {stencil.tile = [Ti, Tj]}
```

**S4 — Overlapped temporal blocking**
```mlir
affine.for %t = 0 to %T {
  affine.for %i, %j over the whole grid { ... }
}
// after
affine.for %tt = 0 to %T step b_t {                  // T / b_t blocks, in order
  affine.for %ti, %tj step (Ti, Tj) {                // tiles, independent
    affine.for %t = %tt to %tt + b_t {               // steps inside one tile
      affine.for %i, %j in tile + shrinking halo { ... }
    }
  }
}
```

**S4 — Diamond tiles** (1D space shown)
```mlir
// tile along the two cone directions instead of along t and i:
(t, i) -> (r·t + i, r·t - i)          // tiles are squares here, diamonds in (t, i)
// diamonds in the same row are independent; rows run in order
```

**S5 — Interior / boundary tiles**
```mlir
%Av = stencil.halo_view %A {lo = [1, 1], hi = [1, 1], bc = constant(0.0)}
affine.for %i, %j over the whole grid { %n = affine.load %Av[%i, %j + 1] ... }   // = A[i-1, j]
// after
%S = memref.alloca() : memref<(T+2)x(T+2)xf64>                    // padded buffer
affine.for %ti, %tj step T {
  affine.if #is_interior(%ti, %tj) {
    affine.for %i, %j in tile { %n = affine.load %A[%i - 1, %j] ... }          // direct
  } else {
    stencil.pack %A[%ti - 1, %tj - 1] into %S {bc = constant(0.0)}            // copy + boundary rule
    affine.for %i, %j in tile { %n = affine.load %S[%i - %ti, %j - %tj + 1] ... }
  }
}
```

**S7 — Parallel region**
```mlir
scf.for %g { affine.for %ti, %tj { ... } }
// after
omp.parallel {                                   // once
  %W = memref.alloca() ...                       // once per thread
  scf.for %g {
    omp.wsloop nowait { interior tiles }
    omp.wsloop        { boundary tiles }         // barrier at the end
  }
}
```

**S6 — Unroll-and-jam + vectorize** (U = 4)
```mlir
affine.for %i { affine.for %j { %x = affine.load %A[%i - 1, %j] ... } }
// after
scf.for %i step 4 {                                   // 4 rows per iteration
  scf.for %j step 8 {                                 // aligned main part
    %r0 = vector.load %A[%i - 1, %j] : vector<8xf64>  // each row loaded once,
    ...                                               // shared by the 4 output rows
    vector.store %o0, %B[%i, %j] {nontemporal = true} // if grid > L3
  }
  // head / tail: vector.maskedload / vector.maskedstore
}
scf.for %i { ...rows left over, not jammed... }
```
