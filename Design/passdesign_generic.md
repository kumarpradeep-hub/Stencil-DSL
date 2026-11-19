# Stencil DSL — One Pipeline for CPU, NVIDIA, AMD and TileIR

Extends the CPU design in [`passdesign2.md`](passdesign2.md). Only what is new or different is described here.

---

## The idea in 6 points

- **One pipeline, no duplicated passes.** A pass does the same work on every target; the numbers and capabilities it needs come from a **target description**.
- **Rule:** a pass may branch on a capability (`hasAsyncCopy`, `matrixShapes(type)`), never on a vendor name.
- **S1–S5a are shared by all targets.** After S5a each kernel takes one **route**: `cpu`, `gpu-cuda` (NVIDIA), `gpu-rocm` (AMD), `gpu-tensor` (tensor-core methods; ConvStencil is the one implemented today) or `tileir`.
- **Shared boundary IR:** parallel loops are `scf.forall` with mapping attributes, not `gpu.launch` or `omp` directly. Each target lowers from there.
- **All data types:** `i8`–`i32`, `fp8`, `f16`, `bf16`, `f32`, `f64`. The route depends on the type (matrix units exist only for some types on some chips).
- **Default GPU route is the direct kernel** (tile in shared memory/LDS, per-thread register blocking). Matrix units are used when the type, `fp-mode` and cost model allow.

---

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Shared boundary = `scf.forall` + target-specific mapping attributes (`#gpu.block`, `#stencil.cpu_thread`, `#chip.pe`) | CPU and GPUs share S6; also fits non-SIMT hardware. The `gpu` dialect is used one level lower, shared by NVIDIA and AMD. |
| 2 | Target description = **DLTI** for numbers + **`StencilTargetInterface`** for rules | DLTI is standard and scoped (CPU spec on the module, GPU spec on `gpu.module`). Rules like occupancy are functions, not numbers. A verifier catches missing keys. |
| 3 | Numbers come from a **chip table** (`sm_90`, `gfx942`, `znver4`, …) + `probe.py` results, overridable by options | Works when cross-compiling; reproducible. |
| 4 | Float matrix units (ConvStencil, TileIR `mmaf`) only with `fp-mode=relaxed` | They reorder sums, and zero weights turn `Inf` into `NaN`. `exact` stays bit for bit. |
| 5 | Integer matrix units allowed under `exact` if overflow is `wrap` | Wrapping arithmetic gives the same bits in any order. |
| 6 | TileIR forks **after S5a**; no pack, no S7 inner mapping, no S6 | `tileiras` owns everything inside a tile. Packing into shared memory ourselves measured ~6 GPt/s vs 46 GPt/s for direct loads. |
| 7 | TileIR loads an **overlapping power-of-2 window** (128 × 128, step 126) | Tile shapes must be powers of 2. Doubling (32 → 64) wastes 75% of loads; overlapping wastes ~3%. |
| 8 | TileIR matmul form: **window + banded matrices** or **gather + stencil2row**; `tileir-form=auto` takes the window form when eligible, `gather` is chosen explicitly (no cost model yet) | Their costs cross over with radius and folding. |
| 9 | f64 tensor route: `nvgpu` DMMA (NVIDIA), MFMA (AMD), only under `relaxed`; otherwise `gpu-cuda` (NVIDIA) / `gpu-rocm` (AMD) | `tileiras` lowers f64 `mmaf` to scalar code (measured). |
| 10 | S5 is split: **S5a `stencil-tile`** (shared) and **S5b `stencil-pack`** (not on TileIR) | TileIR needs tiles but not packing. |

---

## Pipeline at a glance

```
stencil IR
  │  S0   target description, fp-mode, overflow mode
  │  S1   analysis + classify each kernel
  │  S2   lower to one affine nest per stencil
  │  S3   fusion: pointwise, inline (producer into consumer), tile-and-fuse decision
  │  S4   fold the ring, temporal blocking (diamond / overlapped), roll the steps into a loop
  │  S5a  tile every nest (sweep slabs on CPU), split interior / boundary tiles
  ├─ route per kernel ──────────────────────────────────────────────────────────────┐
  │  cpu / gpu-cuda / gpu-rocm / gpu-tensor                         tileir        │
  │  S5b  pack or version boundary tiles, skew the long axis,         stencil-to-tileir
  │       drop the copy-in / copy-out                                 cuda-tile-translate
  │  S6   align, prefetch, unroll-and-jam, vectorize                  tileiras → cubin
  │  S7   map scf.forall → omp | gpu.launch
  ▼  S8   lower to LLVM IR → CPU object (opt -O3, llc) | PTX/cubin | HSACO
```

- S6 runs **before** S7 lowers the loops (S6 needs affine loops).
- Passes marked **new** were added after the comparison with Pluto / YASK / Devito / xDSL; IR examples are in [Appendix B](#appendix-b--ir-examples-for-the-new-passes).

| Stage | CPU | NVIDIA | AMD | TileIR |
|---|---|---|---|---|
| S0–S5a | shared | shared | shared | shared |
| S5b pack | boundary tiles only; diamond and sweep tiles are versioned, not packed | all tiles → shared mem (`cp.async`) | all tiles → LDS | — |
| S7 | forall → `omp` | forall → `gpu.launch` | forall → `gpu.launch` | tile → one TileIR block |
| S6 | SIMD, unroll-jam | per-thread blocking | per-thread blocking | — (`tileiras`) |
| S8 | `llvm` | `nvvm` → PTX | `rocdl` → HSACO | `cuda_tile` bytecode |

### All passes, in pipeline order

| Stage | Pass | What it does | Applies to |
|---|---|---|---|
| S0 | `stencil-select-target` | Attaches chip numbers (DLTI), fp-mode, overflow, strategy and overrides | all |
| S0 | `stencil-verify-target` | Fails if a number the target rules need is missing | all |
| S1 | `stencil-infer-halo` | Reach of each stencil per axis | all |
| S1 | `stencil-model-boundary-conditions` | Records the boundary rule per axis | all |
| S1 | `stencil-decide-bounds-mode` | Records static or dynamic shapes | all |
| S1 | `stencil-infer-layout` | Records the contiguous axis of each field | all |
| S1 | `stencil-alias-contract` | Which arguments must not overlap, and the run-time action | all |
| S1 | `stencil-classify` | Type, linear or not, star or box, radius, time loop | all |
| S2 | `stencil-group-weights` | `fp-mode=relaxed`: adds points sharing a weight, multiplies once | all |
| S2 | `stencil-to-affine` | One affine nest per stencil over halo views; time loop over a ring buffer; marks `stencil.nest`, `stencil.time_loop` | all |
| S3 | `stencil-affine-fusion` | Fuses nests that read the same point | all |
| S3 | `stencil-inline-fusion` **new** | Substitutes a producer into the consumer that reads it with a halo; the temporary disappears | CPU (on); GPU `gpu-cuda` / `gpu-rocm` / `gpu-tensor` (opt-in, off by default); never TileIR |
| S3 | `stencil-tile-and-fuse` | Decides the remaining fused chains and their tile (S5a builds them) | all |
| S4 | `stencil-fold-ring` **new** | 3 time levels → 2, written in place (wave equation) | CPU, diamond |
| S4 | `stencil-temporal-tile` | Chooses strategy, tile and depth; emits diamond or overlapped tiles | all |
| S4 | `stencil-roll-steps` **new** | One step loop per tile instead of one code copy per step | CPU |
| S5a | `stencil-tile` | Tiles nests that are not time-blocked (CPU: sweep slabs); interior / boundary split | all |
| S5a | `stencil-choose-route` | `cpu`, `gpu-cuda`, `gpu-rocm`, `gpu-tensor`, `tileir` per kernel | all |
| S5a | `stencil-to-tileir` | Builds the TileIR module for kernels on that route | TileIR |
| S5b | `stencil-pack` | Packs tile inputs, or versions boundary tiles without packing; padded ring | not TileIR |
| S5b | `stencil-lower-pack` | Pack ops → loops or async copies | not TileIR |
| S5b | `stencil-verify-no-halo-view` | Fails if a halo view is left | not TileIR |
| S5b | `stencil-skew-inner` **new** | Cuts the long (contiguous) axis into time-skewed blocks | CPU, memory-bound |
| S5b | `stencil-ring-endpoints` **new** | First step reads the input, last step writes the output; copies removed | CPU |
| S6 | `stencil-align-buffers` | Buffer pitch to the vector width / copy alignment | all |
| S6 | `stencil-prefetch` | Prefetch the next tile's window (`prefetch=on`, default off) | CPU |
| S6 | `stencil-unroll-jam` | Several outputs per inner iteration | all |
| S6 | `stencil-vectorize` | SIMD on the contiguous axis with a scalar remainder; splits max/min bounds so boundary blocks vectorize too | CPU |
| S7 | `stencil-map-parallel` | Loops → parallel levels (`scf.forall` + mapping) | all |
| S7 | `stencil-forall-to-omp` | `scf.forall` → OpenMP, gap affinity | CPU |
| S7 | `stencil-forall-to-gpu` | `scf.forall` → `gpu.launch` | GPU |
| S8 | `stencil-lower-target-ops` | Lowers `copy_async`, `barrier`, `matrix_contract` | all |
| S8 | `stencil-static-buffers` | Compiler-owned buffers kept across calls (`buffers=static`) | CPU |
| S8 | `stencil-emit-alias-check` | Entry wrapper that checks argument overlap at run time (copy or error) | CPU |
| S8 | `stencil-emit-host-schedule` | Records buffers, loops and launches of the host function as JSON | GPU |
| S8 | upstream MLIR lowering | affine → scf → cf → LLVM dialect; `gpu-module-to-binary` on GPU | all |

- **Not every pass fires on every stencil.** Each one checks its own condition and otherwise leaves the IR unchanged (e.g. fold-ring only on a 3-level ring, skew-inner only when the grid does not fit in cache).
- **Design-only, not built:** `stencil-map-to-matrix-units`, `stencil-async-pipeline` (the `gpu-tensor` route).

---

## Target description

**Numbers (DLTI keys, per device):** memory levels (size, hardware- or software-managed), parallel levels (units, warp/wave size, max threads), SIMD bits, vector registers, NUMA nodes, bandwidth, peak FLOP/s per type, async-copy kind (`none`, `tma`, `cp_async`, `lds_direct`), matrix types per route (from `probe.py`).

**Rules (`StencilTargetInterface`):**

| Method | Used by | CPU | SIMT GPU (NVIDIA + AMD) |
|---|---|---|---|
| `fastMemory()` | S3, S4, S5b | L2, hardware-managed | shared mem / LDS, software-managed |
| `occupancy(bytes, regs)` | S3, S4 | always 1 | blocks per SM / CU |
| `innerWidth(type)` | S3, S5a | vector width | 128-byte coalescing |
| `padPitch(pitch, type)` | S6 | vector width | async-copy alignment + bank rule |
| `parallelLevels()` | S7 | thread (+ SIMD) | block, thread |
| `mappingAttr(level)` | S7 | `#stencil.cpu_thread` | `#gpu.block`, `#gpu.thread` |
| `matrixShapes(type, route)` | route choice, S6 | none | DMMA / MFMA / `mmaf` shapes |

- NVIDIA and AMD share **one** implementation (`#stencil.simt_gpu`); they differ only in numbers.
- The LLVM backend is picked by the `gpu.module` target attribute (`#nvvm.target`, `#rocdl.target`), not by the interface.

---

## Stage 0 — Select target

**Purpose:** attach everything later stages need.
**Passes:** `stencil-select-target{chip, fp-mode, overflow, backend}`, `stencil-verify-target`.

- Attaches the interface attribute, the DLTI numbers, and `#nvvm.target` / `#rocdl.target` on each `gpu.module`.
- **Decision:** which numbers. **Why:** from the chip table, so a build is reproducible and can target another machine.
- `stencil-verify-target` fails if a key the interface needs is missing (e.g. `vregs`).

---

## Stage 1 — Analysis + classify

**Purpose:** facts every later decision needs. As in the CPU design, plus `stencil-classify`.
**Passes:** `stencil-infer-halo`, `stencil-model-boundary-conditions`, `stencil-decide-bounds-mode`, `stencil-infer-layout`, `stencil-alias-contract`, `stencil-classify`.

- `stencil-classify` records per kernel: element type, linear with constant weights or not, star or box, radius, time loop, in-place, overflow mode.
- **Decision:** none yet; the route is chosen after S5a, when fusion is known.

---

## Stage 2 — Lower to affine

**Purpose:** one affine nest per stencil, inputs through halo views. Same passes as the CPU design.

- **Data types:** arithmetic stays in the source type under `exact`.
  - Under `relaxed`, `f16`/`bf16`/`fp8` accumulate in `f32`.
  - Integers with `wrap` may accumulate in a wider type (`i8` → `i32`), then truncate. The result is identical, so this is allowed under `exact`.
  - `saturate` is not associative: no widening, no weight grouping, no folding.
- A `constant(c)` boundary value must be representable in the element type.

---

## Stage 3 — Fusion

**Purpose:** fewer passes over memory.
**Passes:** `stencil-affine-fusion`, `stencil-inline-fusion` (new), `stencil-tile-and-fuse`.

- **`stencil-affine-fusion`:** fuses nests that read each other at the same point (as in the CPU design).
- **`stencil-inline-fusion`** (CPU: on, off with `inline_fusion=0`; GPU: off by default, `inline_fusion_gpu=1` turns it on; never on the TileIR route):
  - A producer that writes a temporary read with a halo is substituted into its consumer, once per read of the temporary.
  - Conditions: the consumer is the only reader of the temporary; size limit (at most 9 reads of the temporary, 16 reads in the producer, product ≤ 64).
  - Cost: the producer is recomputed once per read.
  - The temporary disappears; the step becomes **one** nest, so S4 can use diamond tiles.
  - The halo view of the input grows by the shift (consumer's reach + producer's reach in total).
  - Boundary rule of the temporary:
    - `constant` / none: the read becomes `select(off-grid, c, producer)`; the `off-grid` flag is a placeholder that S5b resolves in boundary strips and folds to `false` elsewhere.
    - `replicate` / `reflect`: the copied reads are tagged with the shift and the rule; S5b remaps them in the strips.
    - `periodic`: not inlined (`stencil-tile-and-fuse` handles it).
  - **GPU (2026-11-17):**
    - The rewrite is the same affine transformation; only the gate changed. It runs on `gpu-cuda`, `gpu-rocm` and `gpu-tensor` when `inline_fusion_gpu=1` (a chip-table override, default 0). It stays off by default until a cost model chooses between inlining and tile-and-fuse; that model does not exist yet, so nothing selects it automatically.
    - **TileIR never inlines.** The TileIR lowering does not resolve the `off-grid` placeholders and tagged reads, so boundary cells came out wrong when it was forced on (chain, 100×72: 1000 cells wrong for replicate / constant, 9 for reflect; 600×520: 8400). The pass returns early when the route is TileIR.
    - Checked on sm_90 with `fp-mode=exact`, bit for bit against the PyTorch reference: chain with replicate, constant and reflect boundaries (f64, f32) and a 600×520 case, on `auto`, `gpu-cuda`, `gpu-tensor` and `tileir`: 20/20 pass (`test_inline_fusion_gpu`). On `tileir` the override changes nothing, because the pass skips that route.
    - **Not measured:** whether inlining is faster than tile-and-fuse on a GPU. An inlined step can only use diamond tiles or no time blocking (`b_t = 1`), and GPU diamond tiles carry per-step barriers, so the benefit is unproven. The cost model should be built from that measurement.
- **`stencil-tile-and-fuse`:** only decides. It marks a chain and its tile; S5a builds it with the same generator as S4.
- **Decision:** tile size. The rule is the same for all targets, with numbers from the interface:
  - footprint ≤ `fastMemory()` share (CPU: 50% of L2; GPU: shared mem / LDS ÷ target blocks per unit),
  - inner width a multiple of `innerWidth(type)`,
  - repeated work ≤ τ, enough tiles for all units.
- **The rolling buffer is the same transformation as GPU 2.5D streaming** (walk the outer axis, keep a few planes).

---

## Stage 4 — Temporal blocking

**Purpose:** run b_t steps per tile to cut memory traffic.
**Passes:** `stencil-fold-ring` (new), `stencil-temporal-tile`, `stencil-roll-steps` (new).

- **`stencil-fold-ring`** (CPU; off with `fold_ring=0`):
  - Applies when a step keeps 3 time levels and the oldest, u(t−1), is read only at the point being written (wave equation).
  - The new value overwrites u(t−1) in place: 3 slots → 2.
  - The step is then diamond-tileable. The overlapped form and the GPU routes keep 3 slots.
- **`stencil-temporal-tile`:**
  - **Decision:** strategy, tile and b_t. Overlapped: the tile and b_t with the lowest modelled memory traffic per point per step that pass τ and the footprint. Diamond: the ranking below. `auto` takes diamond if its traffic × `diamond_penalty_pct` is below overlapped's traffic × (1 + redundancy).
  - `temporal=off | overlapped | diamond | auto`. `off` leaves the step loop alone (S5a then cuts sweep slabs).
  - Diamond: one tile loop per phase (tile / gap bit per split axis); a gap axis has one more region than tiles.
  - In-place (folded) or inlined steps: diamond or no blocking only.
  - A forced diamond shape that cannot be built gives a warning.
- **CPU diamond chooser** (numbers from the chip table):
  - Every tile must fit L2 + this core's share of L3 (both ring slots of the tile's window) and be wider than 2 · reach · (b_t − 1).
  - **Fits in cache** (both ring slots ≤ all L2 + all L3 of the machine). Ranking: keep the contiguous axis whole, split as few axes as possible, largest depth (up to 8), about 4 tiles per core, lowest traffic.
  - **Memory-bound** (anything larger). Ranking: contiguous axis not cut below 512 points, largest depth (up to 64), power-of-two widths before in-between ones, tile footprint closest to L2, longest contiguous axis. In-between widths (12 … 124, step 4) are also tried on the outer axes.
  - **1-D:** both slots of the tile in L1, then the largest depth (up to 1024), then the widest tile.
  - **Skewed axis:** on a memory-bound grid, if the chosen shape cuts the contiguous axis and at least one other axis, the contiguous axis is not diamond-split; it is recorded as a skew block width for `stencil-skew-inner` (`diamond_skew=0` uses a diamond split instead).
  - Chosen shapes (64 cores): heat2d 4096² 16 × 4096, depth 8; heat2d 16384² 128 rows × the whole row, skewed in blocks of 512, depth 64; heat3d 512³ 20 × 20 × 512, depth 10; heat1d 2048 wide, full depth.
- **`stencil-roll-steps`** (CPU; off with `diamond_roll=0`):
  - S4 emits **one** step per tile with a step placeholder; this pass wraps it in a step loop.
  - Tile bounds shrink by the reach per step, gap bounds grow: tile `[tW + s·r, tW + W − s·r)`, gap `[tW − s·r, tW + s·r)`, clamped to the grid.
  - Removes the old depth cap of 8 (each step used to be a separate copy of the code).
  - **GPU** (`diamond_roll_gpu=1`, default off): the step loop runs inside the block, a `stencil.barrier` ends each step, the depth cap is 32 (the tile must stay wider than 2 × reach × (b_t − 1)). S7 sizes the thread block from the mean tile width over the steps. Bit-exact on `gpu-cuda`/`auto`/`tileir` for heat 1D/2D/3D. **No speed gain measured** (H200, heat2d f64 4096² × 100: default 41.8 ms; diamond unrolled depth 8 46.2 ms; rolled depth 8 46.5, depth 16 44.6, depth 32 44.2 ms), so it stays off and the cost model still prefers overlapped tiles. Shared-memory staging (`diamond_smem=1`) is not ported to rolled steps; it does not fire on them.
- **Allowed strategies per target:**

| Target | Allowed | Why |
|---|---|---|
| CPU | overlapped, diamond | `auto` takes diamond (measured faster). |
| GPU CUDA | overlapped, diamond | `auto` takes overlapped: diamond costs one launch per phase and a barrier per thin gap step. |
| TileIR | b_t = 1 | Step-to-step shifted reads inside a tile go through memory (measured slow). |

---

## Stage 5a — Tile (shared)

**Purpose:** every nest tiled; each tile classified once as interior or boundary.
**Passes:** `stencil-tile`, `stencil-choose-route`, `stencil-to-tileir`.

- **`stencil-tile`** tiles the nests S4 did not block (b_t = 1) and builds the chains S3 marked.
- **CPU sweep slabs** (single-stage nest; off with `sweep_slabs=0`):
  - outer axis: one slab per core;
  - contiguous axis: kept whole;
  - middle axes (3D): cut only until the moving front fits L2 + this core's share of L3;
  - slabs are not packed (boundary slabs are versioned in S5b).
- On TileIR the tile is `window − 2R` (window a power of 2), e.g. 126 × 126 from a 128 × 128 window.

### Route choice (per kernel)

**Pass:** `stencil-choose-route` (used by `stencil-pipeline-auto`).

1. `backend=` given → that route.
2. CPU → `cpu`.
3. GPU → a **tensor route** is possible if all hold:
   - the fused body is linear with constant weights;
   - `matrixShapes(type, route)` is not empty;
   - floats: `fp-mode=relaxed`; integers: `overflow=wrap`; `f32` via TF32: only with `fp-mode=lowp`.
4. Take the tensor route unless the cost model predicts the direct kernel more than 10% faster; else the direct route (`gpu-cuda` on NVIDIA, `gpu-rocm` on AMD).

| Type | NVIDIA tensor route | AMD tensor route (gfx942) | Otherwise |
|---|---|---|---|
| `f64` | `nvgpu` DMMA | MFMA | `gpu-cuda` / `gpu-rocm` |
| `f32` | TF32 (`lowp` only) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `f16`, `bf16`, `fp8` | `tileir` | MFMA | `gpu-cuda` / `gpu-rocm` |
| `i8` | `tileir` (native s8 matrix unit; also `gpu-tensor` m16n8k32) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `i16`, `i32` | none | none | `gpu-cuda` / `gpu-rocm` |

- Kernels of one program may take different routes; the host code launches each.

---

## Stage 5b — Pack (not on TileIR)

**Purpose:** give each tile its input with the boundary rule applied, at the lowest cost.
**Passes:** `stencil-pack{scope, memspace}`, `stencil-lower-pack`, `stencil-verify-no-halo-view`, `stencil-skew-inner` (new), `stencil-ring-endpoints` (new).

- **`stencil-pack`**, three ways to handle the boundary:
  - **Pack** (overlapped tiles; all GPU tiles): copy the window into a buffer with the rule applied. CPU: boundary tiles only, thread-local. GPU: every tile, into shared memory / LDS.
  - **Version without packing** (CPU diamond tiles and sweep slabs): per tile index,
    - first and last tiles (and any tile that reaches the edge when the grid is not a multiple of the tile) get constant bounds and are cut into edge strips + a clean centre;
    - the rule is applied only in the strips;
    - middle tiles drop the clamps, so their loops have single-expression bounds.
  - **Padded ring** (CPU, `constant` rule, only when no tile is packed; `pad_ring_pct`, default 4): if the strips would cover ≥ 4% of the grid (sum over axes of 2 · reach / extent), the ring gets a real halo filled once with the constant; no rule code is left at all. Cost: a ring slot no longer has the shape of the arguments, so the copy-in and copy-out stay.
- Also resolves the `off-grid` placeholders and tagged reads left by `stencil-inline-fusion`.
- **`stencil-lower-pack`** emits `stencil.copy_async` when `asyncCopy != none`, else plain loops.
- **`stencil-skew-inner`** (CPU, rolled diamond tiles for which S4 recorded a skewed axis):
  - cuts the contiguous axis into blocks of width B around the step loop;
  - block k at step s covers `[kB − s·r, kB + B − s·r)`, so a block's data stays in L2 for all steps;
  - middle blocks get single-expression bounds; first and last intersect with the grid.
- **`stencil-ring-endpoints`** (CPU; not with a padded ring or the overlapped strategy; off with `ring_endpoints=0`):
  - the 2-slot ring becomes two buffers A and B;
  - the array of each read / write is selected by the time, outside the point loops: time 0 reads the input argument, time T writes the output argument;
  - the copy-in and copy-out nests are deleted (about 1.5 ms at 4096², 25 ms at 16384²).
- GPU boundary rule: `constant(0)` needs nothing more when the async copy fills off-grid cells with zero; other rules add a small fix-up loop in boundary tiles only.

---

## Stage 7 — Parallelize

**Purpose:** map tiles and work inside a tile onto the hardware.
**Passes:** `stencil-map-parallel` (shared), then `stencil-forall-to-omp` (CPU) or `stencil-forall-to-gpu` (NVIDIA, AMD; uses upstream forall-to-block/thread mapping).

- **Thread count (CPU) - design choice:** the thread count is ONE number, given by the user at compile time (`compile(..., threads=N)` or `threads=N` in the pipeline options; the same as `override=units=N`). The chip table's `units` (zen5: 64) is only the default when the user gives none. GPU chips ignore `threads`.
  - *How it works:* S0 stores N as `units` in the target description. The tile passes (S4, S5a, cost model) read `units` to cut the grid (heat2d, N = 64: 1024 / 64 = 16-row bands, 64 x 1 tiles; N = 32: 32 tiles of 32 rows). `stencil-forall-to-omp` reads the same `units` and writes `num_threads(N)` on `omp.parallel`. The generated code therefore contains N.
  - *Why:* before, tiling used the chip table and the thread count came from `OMP_NUM_THREADS` at run time, so the two could disagree (64 tiles on 48 threads: 16 threads get two tiles, 32 get one, and the run waits for the slowest). Using one compile-time number removes that.
  - *Consequences:* `OMP_NUM_THREADS` is not read by the generated code and the CPU runtime no longer sets it. Changing N changes the generated code, so it is compiled again (the compiled-code cache is keyed by the code, so the old binary is never reused). The tile count is about N, not always exactly N (some nests keep more tiles).
  - *Trade-off:* a thread-count change costs a recompile. A run-time choice would avoid it but brings the uneven load back; it was rejected for that reason.
  - *Checked in:* `test/lit/S7/parallelize.mlir` (THREADS and DEFAULT lines) and `test_threads` in `test/python/test_correctness.py` (threads 1, 3, 16, 64, 100, bit-exact).
- **Decision (shared):** which loop goes to which parallel level, and the per-thread work (rows per thread = the S6 unroll factor).
- **Time loop:**
  - CPU keeps it inside one parallel region.
  - GPU makes one kernel launch per time block. **Why:** no sync across blocks inside a kernel.
- **Scratch buffers:**
  - CPU: once per thread.
  - GPU: in the block's shared memory / LDS.
- **Interior and boundary tiles:**
  - CPU: one worksharing loop over a linear tile index.
  - GPU: one launch with a branch per block (no divergence, the whole block takes one side).
- **Diamond phases:** each phase is one parallel loop (CPU) or one launch (GPU).
  - CPU gap affinity: a gap phase reuses the tile phase's static blocks, so a gap runs on a thread that computed one of its neighbours.

---

## Stage 6 — Inner code

**Purpose:** fast code inside a tile. Runs before S7 lowers the loops.
**Passes:** `stencil-align-buffers`, `stencil-prefetch`, `stencil-unroll-jam`, `stencil-vectorize`.

| Pass | CPU | GPU |
|---|---|---|
| `align-buffers` | pitch to vector width | pitch to async-copy alignment; bank padding only if a warp reads down a column |
| `prefetch` | next tile's window into L2 (`prefetch=on`; default off, no gain measured) | — |
| `unroll-jam` | U outputs along the reuse axis | U rows per thread (register blocking) |
| `vectorize` | SIMD, main loop + scalar remainder | up to 128-bit loads per thread; thread-guard instead of masks |

- **Bound versioning (new, in `stencil-vectorize`):** a loop whose bounds are still `max(a, b)` / `min(c, d)` is copied under `affine.if` into single-bound versions, so boundary blocks and skewed blocks vectorize too.
- The remainder loop starts at `max(split, lower bound)`, so it does not run on an empty range.
- **Streaming stores** (`nontemporal=auto`): for final outputs larger than L3, also through a `select`.
- **Design-only:** `stencil-map-to-matrix-units`, `stencil-async-pipeline` (tensor route, not built).

---

## Stage 8 — Lower

**Purpose:** machine code.
**Passes:** `stencil-lower-target-ops`, then per target.

- `stencil-lower-target-ops` lowers the 3 target-neutral ops: `stencil.copy_async`/`wait`, `stencil.barrier`, `stencil.matrix_contract`.
- **CPU only:** `stencil-static-buffers` (`buffers=static`), `stencil-emit-alias-check`, then the upstream lowering to the LLVM dialect.
- **GPU only:** `gpu-kernel-outlining`, `convert-gpu-to-nvvm` / `-rocdl`, `gpu-module-to-binary`, `stencil-emit-host-schedule`.

| | copy_async | barrier | matrix_contract | then |
|---|---|---|---|---|
| CPU | — | — | — | `omp` → `llvm`, native compile with `-mcpu` |
| NVIDIA | `cp.async` (TMA planned) | `gpu.barrier` | `nvgpu.mma.sync` (DMMA) | `convert-gpu-to-nvvm`, `gpu-module-to-binary` |
| AMD | `amdgpu.gather_to_lds` or loads | `gpu.barrier` | `amdgpu.mfma` | `convert-gpu-to-rocdl`, `gpu-module-to-binary` |

- Under `exact`, fp-contract is forced off in all three backends (NVPTX fuses multiply-add by default).
- **CPU runtime (`runtime_cpu.py`):**
  - runs LLVM `opt -O3` before `llc` (`STENCIL_LLVM_OPT=none` skips it); without it index arithmetic stayed inside the inner loops;
  - medium code model, so static buffers above 2 GB link;
  - pins one thread per physical core (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`) unless the user set it.
- **TileIR route:** `stencil-to-tileir{form=window|gather}` → `cuda-tile-translate` → `tileiras`. Print with `--mlir-print-elementsattrs-with-hex-if-larger=-1`.

---

## TileIR route in detail

- **Window load:** a `strided_view` with tile 128 × 128 and step 126 per axis. The outer ring of each result is invalid; a masked store writes the inner 126 × 126.
- **Boundary rules:**
  - `constant(0)`: out-of-grid view loads return zero.
  - Other rules: boundary windows use a gather with a remapped index (clamp, mirror, mod); interior windows keep the view load.
- **Matmul, window form:** `out = Σ_a S_a · X · T_a` over row offsets a.
  - `S_a` shifts rows; `T_a` is banded with row a's weights.
  - Star stencils need only two products: `out = L·X + X·R`.
  - Cost: ~2(2R+1)·N multiply-adds per output (2N for a star). The constant matrices count in the footprint.
- **Matmul, gather form** (`tileir-form=gather`, `lib/TileIR/GatherForm.cpp`, implemented): stencil2row, a data matrix `A[M,K]` of gathered neighbours × a weight matrix. Fewer multiply-adds (~K), but pointer gathers instead of view loads; no chain parameter and no border strips. Rank 1 to 3, any linear kernel, f16, bf16, i8 on matrix units, f32 / f64 as scalar code. Measured once (H200, heat2d 4096² x100, f16): gather is 2.16× the window form without a chain and about equal to the best chained window (7.75 vs 7.06 ms); f32 about equal, f64 0.59× (slide 107). Option: `tileir-form=auto|direct|window|gather`; bench: `bench_tileir_chain.py DTYPE N STEPS --form window|gather|both`.
- **Limits:** only linear bodies with constant weights (the body is rebuilt from the S5a loops). Anything else uses the direct route (`gpu-cuda` / `gpu-rocm`).

---

## Adding a new target

1. A chip-table entry (DLTI numbers).
2. An interface implementation, or reuse `#stencil.simt_gpu`.
3. Mapping attributes for its parallel levels.
4. Lowerings for the 3 target-neutral ops.
5. A serializer (`gpu::TargetAttrInterface`).

No shared pass changes.

---

## To verify before implementing

- `probe.py`: add `i8` → `i32` (`mmai`), `fp8`, TF32, and `padding_value` on out-of-bounds integer view loads.
- TileIR: cost of the masked 126 × 126 store; size limit for constant matrices.
- AMD: this LLVM build lacks the AMDGPU target; rebuild. Check `gather_to_lds` on gfx942 for 8-byte elements.

---

## Implementation status (`stencil-dsl/`)

**Implemented and tested** (re-run 2026-10-08: 23/23 lit tests; 182/182 correctness cases, bit for bit under `exact`; `check.py --chip host` 26/26 for each of `auto`, `diamond`, `off`):
- S0–S8 as described, one pass list for all chips (`stencil-pipeline-auto`).
- CPU thread count is user-given (`threads=N`, default = chip-table `units`); the tile count and the OpenMP `num_threads` clause use the same number (checked in `test/lit/S7/parallelize.mlir` and `test_threads`).
- `gpu-tensor` runs f64 (m8n8k4), f32 (tf32 m16n8k4) and, natively with no data conversion, f16 / bf16 (`mma.sync` m16n8k16, f32 accumulate) and i8 (m16n8k32, s32 accumulate; the weights are taken modulo 256 into [-128, 127]). The matrix is 16 groups of `P` outputs per MMA for the native kinds (`ConvStencilPlan::mmaM / mmaK / mmaPack`); the tile fill is a predicated load and store because `cp.async` cannot copy 1 or 2-byte elements. Every boundary rule works: the tile fill applies it with `emitBoundaryLoad` (a plain load and store instead of the zero-filling `cp.async`) and, between fused steps of an overlapped chain, ghost cells are pinned to the constant, left as the real wrapped cells (periodic) or copied from the interior after the stage (replicate, reflect); periodic is not taken in a diamond tile. The route stays off in `fp-mode=exact`, even when forced (the matrix units regroup the arithmetic, and f32 uses tf32). The native kinds are taken only with `backend=gpu-tensor` (automatic route choice keeps f64 / f32 until they are measured: heat2d f16 4096² x100 takes 28.7 ms against 5.3 ms for the direct kernel). The fused-chain tile is re-planned until its real shared memory (widened, two buffers) fits the part. The f32 tolerance on this route grows with the step count (tf32 truncation repeats per step: 1.8e-3 after 1 step, 1.2e-2 after 16, identical for every boundary rule); `check.py` allows 5e-3 + 1e-3 per extra step. Tests: `test_gpu_tensor_boundary_rules` (117), `test_gpu_tensor_native_dtypes` (27), `test_gpu_tensor_native_instructions`, `test_gpu_tensor_i8_large_weights_wrap_exactly`, `test_gpu_tensor_fused_chain_fits_shared_memory`; lit: `S5/gpu_tensor_lower.mlir` (F16, BF16).
- **gpu-tensor forms.** The route has two forms, chosen by `tensor-form=auto|gather|window` (`auto` = gather; each launch carries `stencil.tensor_form`). *Gather* is the ConvStencil kernel described above. *Window* (`lib/GpuTensor/WindowForm.cpp`, shared helpers in `TensorBuilder.h`) stages the input window in shared memory with plain loads and computes `out = Σ_i X_i · T_i`: `X_i` is the window shifted by the row tap `i` (a plain M × K block, so an A fragment is consecutive cells), `T_i[k][n] = w[i][k − n]` is a banded compile-time table (one B fragment per lane). Per type: f64 m8n8k4, f32 tf32 m16n8k4, f16 / bf16 m16n8k16, i8 m16n8k32, N = 8 outputs per MMA, ceil((8 + nLast − 1) / K) steps per tap tuple; M = 16 rows (f64 8) along the second-to-last axis (rank 1 treats the line as rows of S cells that overlap by the halo). A block is 32 rows × 128 columns (smaller if the shared tile does not fit); the shared-memory row stride is the one with the fewest simulated bank wavefronts. Rank 1 to 3, every boundary rule, the five types. **One step per launch:** it needs `temporal=off`; a fused chain (`b_t` > 1) or a diamond tile is an error that names `temporal=off` / `tensor-form=gather`, not a silent fallback. Tests: `test_gpu_tensor_window_form` (125 + 20), `test_gpu_tensor_window_form_is_used`, `test_gpu_tensor_window_form_rejects_fused_chains`; lit `S5/gpu_tensor_window.mlir`; bench `test/python/bench_gpu_tensor.py DTYPE N STEPS --form gather,window`. Measured once (H200 shared, heat2d 4096² x100, `temporal=off`, window / gather speed): f16 1.00×, bf16 0.95×, i8 1.34×, f32 1.11×, f64 1.42×; heat3d f16 1024³ x50 1.67×. Not tuned, not repeated: the tile is a first choice. Open: window form for fused chains and diamond tiles; a cost model to choose the form.

  - **Open (gpu-tensor), speed.** Nothing in the native f16 / bf16 / i8 kernels or in the new boundary-rule paths is tuned or benchmarked. Measured once, H200, heat2d 4096² x 100, best of the timing windows: f16 28.7 ms, bf16 28.7 ms, i8 20.1 ms, f32 (tf32) 20.6 ms on `gpu-tensor`, against 5.3 ms (f16) and 4.9 ms (i8) for the direct `gpu-cuda` kernel and 15.3 ms for `tileir` f16. Likely causes, not profiled: A fragments are assembled from scalar shared-memory loads (no `ldmatrix`), the tile fill is a predicated load and store with a per-element select, the fused chain asks for a large shared-memory tile (up to 204 KB), and the per-MMA cost used by the planner (`mmaCostScale = 0.55`) is an estimate, not a calibration. The replicate / reflect repair pass after each fused step and the pinned constant ghost cells add work that was not measured separately.
  - **Open (gpu-tensor), automatic route choice.** `stencil-choose-route` picks `gpu-tensor` by itself only for f64 and f32 (`fp-mode=relaxed`, linear one-stage stencil). f16, bf16 and i8 run on it only with an explicit `backend=gpu-tensor` (`consider()` in `S5a_Tile.cpp`: "native <type> matrix kernels are used only with an explicit backend=gpu-tensor"); the f16 case therefore still goes to the direct kernel or `tileir`. To lift this: benchmark the native kernels against the direct kernel on the shapes in `test/python/bench.py`, calibrate `mmaCostScale` and the bank-conflict model for 2-byte and 1-byte elements, add the comparison to the route choice, and update `S5/choose_route.mlir` (AUTO16).
  - **Other limits kept:** one stage and one ring slot only (no chain, no wave); linear weights only; i16 / i32 / i64 have no tensor-core instruction; the route is off in `fp-mode=exact` even when forced.
- Routes `cpu`, `gpu-cuda` (NVIDIA, runs here), `gpu-rocm` (AMD: compiles to ROCDL and AMDGPU LLVM IR), `tileir`.
- All data types i8–i64, f16, bf16, f32, f64; all boundary rules, per axis.

**Changes from the design**
- **S6 runs before S7 lowers the loops.** S6 needs affine loops; after S7 they are `scf`/`omp`/`gpu` (review item 9). S7's `scf.forall` is still the shared boundary.
- **Unroll-and-jam and vectorization need single-expression loop bounds.** Interior tiles have them; CPU boundary tiles get them from S5b's versioning and S6's bound versioning. Packed boundary tiles stay scalar.
- **S3 tile-and-fuse only decides.** It marks the chain and its tile; S5a builds it with the same generator as S4 ("grow the tile by the halo").
- **The CPU tile loop is one linear index** per worksharing loop, not two static loops for interior and boundary tiles.
- **TileIR window form:** 2D, one input, `constant(0)` or no rule, float, `relaxed`. The first rows and columns (before the window can start) use the direct form. The direct form is used for everything else, and with `exact`.
- **Route choice uses measured efficiencies** (chip table: `eff_direct_pct`, `eff_tileir_pct`). On the H200 the direct kernel is predicted faster than the TileIR window form (measured 316 vs 106 GPoint-steps/s, f16 heat 4096²), so auto takes `gpu-cuda`; `backend=tileir` forces TileIR.
- **`overflow=saturate` only keeps integer operations in their written order.** The DSL has no saturating arithmetic.

**CPU memory-traffic options** (measured, 64 cores, interleaved runs; `test/python/bench_cpu.py`)

| Option | What it does | Gain | Default |
|---|---|---|---|
| `nontemporal=auto` | Streaming (cache-bypassing) stores for final outputs larger than L3; 64-byte aligned full vectors only, `sfence` after the nest, alignment checked at the entry | 1.04–1.08× on a single-step memory-bound stencil (4096²); none when blocked; 0.83× if forced on data that fits in L3 | `auto` |
| `prefetch=on` | At the start of an interior tile, prefetch the next tile's input window into L2 | none measurable (0.99–1.04×): Zen 5's hardware prefetchers already cover it | `off` |
| `buffers=static` | Compiler-owned buffers (rings, temporaries) kept across calls instead of allocated per call | per-call cost 22.8 → 1.7 ms; heat 100 steps 1.47× | `heap` (static is not reentrant) |

Also measured and rejected: a round-robin tile schedule (0.75×, loses NUMA and cache locality). Also measured with no gain: unroll-and-jam of the scratch-buffer stages (it was missing, now applied). The time-blocked kernel is compute-bound: about 6 cycles per point.

**Diamond tiles** (`temporal=diamond`; `auto` compares with overlapped)
- **How:** along each split axis, tiles shrink by the reach each step and gaps grow; 2^(split axes) phases run as rank + 1 waves. No repeated work. The tiles read the arrays directly (no pack).
- **CPU boundary tiles are versioned per axis** on the tile index (S5b):
  - first and last tiles get constant bounds and are cut into edge strips (the rule is applied there) and a clean centre;
  - middle tiles drop the clamps;
  - so every loop keeps single-expression bounds and S6 vectorizes it.
  - Before this, boundary tiles applied the rule on every read, in scalar code; that was the cause of the earlier 0.8× (it happened even on 1 thread, so it was not cross-core traffic).
- **CPU chooser (S4), first version** (now the "fits in cache" branch; the current rules are in [Stage 4](#stage-4--temporal-blocking)), ranking measured on Zen 5:
  - keep the contiguous axis whole;
  - split as few axes as possible;
  - then the largest b_t;
  - then about 4 tiles per core;
  - then the lowest traffic;
  - budget: L2 plus this core's share of L3.
  - Picks 16×4096 for 4096², 16×16×256 for 256³.
- **CPU gap affinity (S7):** a gap phase shares out the tile phase's index (same static blocks). Each tile runs the gaps just below it, so every gap runs on a thread that computed one of its two neighbours.
- **GPU:** one launch per phase. Gap blocks are sized from their mean region (about 4 points per thread, at least one warp), not the tile. Shared-memory staging (two buffers per tile; fill only what the tile did not compute; write through to global) is implemented but **opt-in** (`diamond_smem=1`): L1 already caches the direct reads.
- **Tuning overrides:** `diamond_tile<d>`, `diamond_bt`, `diamond_smem`, `diamond_threads`, `diamond_affinity`, `diamond_penalty_pct`.
- **Auto rule:** diamond if its modelled traffic × `diamond_penalty_pct` is below overlapped's traffic × (1 + redundancy). Penalty: 1.0 on CPUs (measured faster), 6.9 on GPUs (measured f64 time ratio 1.52 / traffic ratio 0.22). So auto takes diamond on CPUs and overlapped on GPUs.

Measured speed-up over `temporal=off` (CPU: `bench_cpu.py`, interleaved, `buffers=static`; GPU: `bench.py`, H200). These CPU rows were measured **before** the passes added after the comparison (the `off` baseline itself was 5× slower then):

| Case | Overlapped | Diamond, before | Diamond, now |
|---|---|---|---|
| Zen 5, heat2d f64 4096² × 100, 64 threads | 1.71× | 0.86× | **6.14×** |
| Zen 5, same, 1 thread | 1.84× | 0.92× | **2.44×** (16 × 4096 forced; chosen shape 1.37× over overlapped) |
| Zen 5, heat3d f64 256³ × 40, 64 threads | 0.99× | — | **2.74×** |
| Zen 5, gap affinity (on vs off) | — | — | 1.04–1.09× (2D), 1.07–1.16× (3D) |
| H200, heat2d f64 4096² × 100 | 1.34× | 0.83× | 0.88× (shared memory: 0.90×) |
| H200, heat2d f32 | 1.52× | 0.96× | 0.93× (shared memory: 0.89×) |
| H200, heat2d f16 | 1.70× | 1.35× | **1.79×** (shared memory: 1.69×) |
| H200, heat3d f64 256³ × 20 | 1.00× | 0.37× | 0.46× (shared memory: 0.34×) |

- **CPU tile shapes** (64 threads, vs overlapped): 16 × 4096: 3.1×; 32 × 1024: 3.0×; 128 × 512: 2.3×; 256 × 256: 1.9×. The old chooser's square tiles were the worst. Whole rows give long unit-stride streams and full-width vector gaps.
- **GPU, what limits diamond** (Nsight Compute, f64, 8 steps, 64 × 32 tiles):
  - the tile phase takes about 250 µs, 2.4× faster per step than `off`;
  - the three gap phases do about 12% of the work but take about as long again;
  - barrier stalls are 52–60%.
  - Right-sizing gap blocks recovered part of this. The rest is the per-step barrier on thin regions.
  - Tradeoff: bigger tiles make gaps relatively thinner but give fewer blocks. 128 × 128 measured 0.49×.
- **Compile time:** boundary versioning copies each stage up to 3^(split axes) times. The pytest suite went from 83 s to 150 s. 3D with three split axes took 73 s per case; the current chooser splits two axes (10 s).

**Comparison with other stencil compilers** (scripts in `baselines/bench/`)
- **Baselines:** Pluto 0.13 (diamond), YASK, Devito 4.8, xDSL 0.71 stencil lowering, plain C + OpenMP, PyTorch CPU. Each was tuned by a sweep.
- **Setup:** same input, zero halo, 64 threads pinned (`OMP_PLACES=cores OMP_PROC_BIND=close`), kernel time, tools run in interleaved rounds.
- **Machine:** 2 × 32 cores, 320 MB of cache in total. 4096² f64 (268 MB) fits in cache, so 16384² is used for time-blocking claims. The machine is shared (load 20–95).
- **Start point** (heat2d f64 × 100, before the passes below; `final_4096.log`, `final_16384.log`): 4096² ours 12.7 ms (auto), 79.3 (off), Pluto 10.7; 16384² ours 587 ms (auto), Pluto 178, YASK 177.

**Passes added after the comparison** (2026-10-02/03; details in the stage sections and [Appendix B](#appendix-b--ir-examples-for-the-new-passes))

| Pass / change | Stage | Fixes | Measured |
|---|---|---|---|
| Sweep slabs | S5a | b_t = 1 sweeps were square tiles, packed | `off` 79.3 → 16.4 ms (4096²) |
| `stencil-roll-steps` + two-regime chooser | S4 | diamond depth capped at 8 | 16384²: 587 → 218 ms (together with the two rows around it) |
| `opt -O3` before `llc` | runtime | index arithmetic in inner loops | not measured separately |
| `stencil-fold-ring` | S4 | wave equation could not be diamond-tiled | wave2d × 100, pass off → on (`fold_ring=0/1`, re-measured 2026-10-08): 16384² 1837 → 202 ms; 4096² 57.3 → 14.3 ms |
| `stencil-skew-inner` | S5b | long rows fall out of L2 on big grids | not measured separately (16384²: 218 → 170 ms with all later passes) |
| `stencil-ring-endpoints` | S5b | copy-in / copy-out of the ring | about 1.5 ms (4096²), 25 ms (16384²) |
| `stencil-inline-fusion` | S3 | producer–consumer chain kept a temporary and could not be diamond-tiled | chain 16384²: 141 ms (hand-fused OpenMP 707) |
| Padded ring | S5b | boundary strips a large share of small grids | — |
| Exact per-index versioning | S5b | 2× slower when the grid is not a multiple of the tile | — |
| Bound versioning + remainder fix | S6 | boundary / skewed blocks ran scalar | heat2d 16384²: 0.95× → 1.07× of Pluto |
| Thread pinning | runtime | unpinned threads | 21.6 → 12.7 ms (4096²) |

**Final comparison** (2026-10-03; f64, zero halo, 64 threads pinned, kernel ms, best of 3 on a shared machine; Pluto at its best tile of a sweep, ours on `auto`; `baselines/bench/suite.py`, `fusion.py`; logs `suite_large.log`, `suite_small.log`, `fusion_*.log`)

| Stencil (100 steps unless noted) | Memory-bound: ours / Pluto | In cache: ours / Pluto |
|---|---|---|
| heat1d (67M points; 4M points × 400 steps) | **20.5** / 23.5 | 8.9 / **5.7** |
| heat2d (16384², 4096²) | **170** / 183 | 10.8 / 10.8 |
| box2d 9-point (16384², 4096²) | **209** / 251 | **13.3** / 14.9 |
| heat3d (512³, 256³) | 321 / **265** | **23.7** / 48.6 |
| star3d radius 4 (512³, 256³; 50 steps) | **1045** / 1474 | **57.5** / 218 (plain OpenMP 43.1) |
| wave2d (16384², 4096²) | 202 / 201 | 14.7 / **12.3** |

| Fusion chain (lap → blur → update), 50 steps | 4096² | 16384² |
|---|---|---|
| **ours** | **9.98** | **141** |
| plain OpenMP, 3 loops per step | 83.9 | 2218 |
| plain OpenMP, fused by hand | 12.5 | 707 |
| Pluto | 35.7 | 561 |
| xDSL | 42.3 | 1548 |
| Devito (result differs: boundary semantics) | 63.8 | 1662 |
| PyTorch `torch.compile` / eager | 1568 / 9999 | 19248 / 128265 |

- Other tools on heat2d 16384² × 100 (from the first-round run and their tuning sweeps): YASK 177, xDSL 1336, Devito 1551, plain OpenMP 1336.
- Our results are bit-identical to plain OpenMP and to Pluto in every row (C baselines built with `-ffp-contract=off`). Devito's differ (heat2d: 6.7e-16; fusion chain: 0.94).
- **32 threads (2026-10-06), heat2d 4096² × 100:** ours auto 20.7 ms, Pluto re-tuned 22.2 ms, plain OpenMP 103 ms (`examples/pluto.py`; best of 4 rounds).
- **Still behind:** heat3d 512³ (0.83× of Pluto), heat1d in cache (0.64×), wave2d in cache (0.84×), star radius 4 in cache vs plain OpenMP (57.5 vs 43.1 ms), small grids (heat2d 1024² × 100 on 32 threads: ours 1.9 ms, Pluto 1.1 ms, plain OpenMP 1.6 ms).
- **Known runtime issue:** the kernel's OpenMP runtime pins the calling thread to one core; PyTorch / NumPy work started afterwards on that thread inherits it.

**Not implemented yet**
- The `gpu-tensor` route (ConvStencil on nvgpu, MFMA on AMD). The route choice records this and uses the direct route (`gpu-cuda` / `gpu-rocm`). On the H200 the direct f64 kernel measured 301 GPoint-steps/s against 152 for the earlier ConvStencil implementation.
- The TMA lowering: `copy_async` uses `cp.async` on sm_90.
- Folding several steps into one (S4). `stencil-fold-ring` is a different thing: it folds the storage, not the steps. GPU diamond with fewer phases (e.g. diamond along one axis, overlapped along the others, so a gap block is not thin).
- Rules and passes for in-place (Gauss-Seidel) updates.
- Inline fusion with a `periodic` temporary; fold-ring, skew and ring endpoints on the GPU routes; roll-steps on GPU exists only behind `diamond_roll_gpu=1` (no gain).
- Inline fusion on the TileIR route (its lowering does not resolve the `off-grid` / `via` placeholders), and a cost model that selects inline fusion on GPU (today only `inline_fusion_gpu=1` enables it).
- A recognizer for plain affine loops: only nests marked by S2 are optimized; other affine code passes through unchanged.

**Measured (heat2d, 4096² × 100 steps; before the passes added after the comparison, so the CPU rows are out of date: see the final comparison above)**

| Target | f64 | f32 | f16 |
|---|---|---|---|
| Zen 5, 64 cores (default: diamond, heap buffers) | 23 | 66 | 89 (no native f16) |
| Zen 5, 64 cores, `buffers=static` | 35 | 137 | 158 |
| H200, gpu-cuda | 301 | 405 | 316 |
| H200, TileIR window | — | — | 106 |

GPoint-steps/s, including allocation and copy-in/out. The GPUs and the CPU host are shared (CPU load average 16–24 during the Zen 5 runs), so the numbers are best of short windows. With heap buffers the f64 run is dominated by allocating and first-touching the 256 MB ring on every call.

---

## Appendix A — Example: 2D heat on four targets

**Note:** this is the design-time example (a 16-core CPU, overlapped tiles). The CPU tile choices shown here are not what the current chooser picks; current CPU IR is in Appendix B.

**Program:** as in [`passdesign2.md` Appendix A](passdesign2.md#appendix-a--worked-example-2d-5-point-heat-diffusion). Grid 4096², T = 100, one step `u + 0.1·(n + s + w + e − 4c)`, boundary `constant(0)`.

| Run | Chip | Type | fp-mode | Route |
|---|---|---|---|---|
| CPU | 16 cores, AVX-512 | `f64` | exact | `cpu` |
| NV | H200 NVL (`sm_90`) | `f64` | exact | `gpu-cuda` (f64 tensor route needs `relaxed`) |
| AMD | MI300X (`gfx942`) | `f64` | exact | `gpu-rocm` |
| TIR | H200 NVL | `f16` | relaxed | `tileir` |

Only the lines that change are shown. `...` is unchanged IR.

### S0 — `stencil-select-target`

```mlir
// CPU
module attributes {stencil.target = #stencil.cpu, dlti.target_system_spec = #dlti.target_system_spec<
  "CPU" = #dlti.target_device_spec<"l1" = 49152, "l2" = 2097152, "l3" = 33554432, "simd_bits" = 512,
                                   "vregs" = 32, "cores" = 16, "numa_nodes" = 2>>}
// NV and TIR (same chip; TIR differs only in the route chosen after S5a)
"GPU" = #dlti.target_device_spec<"units" = 132, "fast_mem" = 233472, "warp" = 32, "async_copy" = "tma",
                                 "bw_gbs" = 4800, "tileir_mma" = ["f16", "bf16", "f8E4M3", "f8E5M2"]>
gpu.module @k [#nvvm.target<chip = "sm_90">] { }
// AMD
"GPU" = #dlti.target_device_spec<"units" = 304, "fast_mem" = 65536, "warp" = 64, "async_copy" = "none", ...>
gpu.module @k [#rocdl.target<chip = "gfx942">] { }
```

### S1 — `stencil-classify` (all runs)

```mlir
%lap = stencil.apply ... attributes {stencil.halo = ..., stencil.bc = ...,     // as in the CPU design
         stencil.class = {elt = f64, linear = true, shape = "star", radius = 1, time_loop = true}}
// TIR: elt = f16
```

### S2 — `stencil-to-affine`

- CPU, NV, AMD: as in the CPU design.
- TIR (`relaxed`): `stencil-group-weights` gives `0.6·c + 0.1·(n + s + w + e)`, accumulated in `f32`.

```mlir
%sum = arith.addf ... : f32                     // n + s + w + e, after arith.extf
%r   = ... 0.6 * c + 0.1 * %sum ... : f32
%h   = arith.truncf %r : f32 to f16
```

### S3 — fusion (all runs)

- `lap` and `u_new` fuse pointwise into one nest (same as the CPU design). No tile is set yet.

### S4 — `stencil-temporal-tile`

| Run | Tile (i × j) | b_t | Why |
|---|---|---|---|
| CPU | 128 × 256 | 2 | See `passdesign2.md`. |
| NV | 32 × 64 | 3 | Budget 57 KB (4 blocks/SM): 2 × 38 × 70 × 8 B = 42.6 KB. Extra work 596 / 6144 = 9.7%. Traffic 0.77 per point-step; 64 × 64 at b_t = 1 would be 2.06. |
| AMD | 32 × 32 | 2 | Budget 32 KB (2 workgroups/CU): 2 × 36 × 36 × 8 B = 20.7 KB. Extra work 132 / 2048 = 6.4%. b_t = 3 would be 13.2% > τ. |
| TIR | 126 × 126 (window 128) | 1 | Folding 2 steps: 896 multiply-adds per output (compute ≈ 1.07 ps per point-step at ~835 TFLOPS f16) is slower than b_t = 1 (memory ≈ 0.86 ps at 4.8 TB/s). |

```mlir
// NV: time loop in blocks of 3; the last block runs 1 step
affine.for %tb = 0 to 100 step 3 {
  affine.for %ti = 0 to 128 { affine.for %tj = 0 to 64 {                         // 32 × 64 tiles
    affine.for %s = 0 to min(3, 100 - %tb) { ... region shrinks by 1 per step ... } } } }
// AMD: step 2, 128 × 128 tiles of 32 × 32
```

### S5a — `stencil-tile` + route choice

```mlir
// NV: 7812 interior tiles (126 × 62), 380 boundary tiles (4.6%)
affine.if #interior(%ti, %tj) { ... } else { ... }
// AMD: 15876 interior tiles (126 × 126), 508 boundary tiles (3.1%)
// TIR: tiles of 126 × 126 → 33 × 33 = 1089 windows per step
func.func @heat(...) attributes {stencil.route = "tileir"}   // CPU: "cpu", NVIDIA: "gpu-cuda", AMD: "gpu-rocm"
```

### S5b — `stencil-pack` + `stencil-lower-pack` (not TIR)

```mlir
// CPU: boundary tiles only, as in passdesign2.md
// NV: every tile; TMA fills off-grid cells with 0 = constant(0)
%S0 = memref.alloca() : memref<38x70xf64, #gpu.address_space<workgroup>>
%S1 = memref.alloca() : memref<38x70xf64, #gpu.address_space<workgroup>>       // ping-pong inside the block
stencil.copy_async %U[%slot, %ti * 32 - 3, %tj * 64 - 3] -> %S0 {box = [38, 70]}
stencil.wait
// boundary tiles only: after each step, zero the off-grid cells of the intermediate buffer
// AMD: plain loads; off-grid cells written as 0
%S0 = memref.alloca() : memref<36x36xf64, #gpu.address_space<workgroup>>
affine.for %r = 0 to 36 { affine.for %c = 0 to 36 { ... in grid ? load : 0.0 ... } }
```

### S7 — `stencil-map-parallel` + lowering

```mlir
// Shared result (NV shown)
scf.forall (%ti, %tj) in (128, 64) {
  scf.forall (%ty, %tx) in (4, 64) { ... 8 rows × 1 column per thread ... }
    {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
} {mapping = [#gpu.block<y>, #gpu.block<x>]}
// NV after stencil-forall-to-gpu: 34 launches (33 × 3 steps + 1 × 1 step), 8192 blocks × 256 threads
gpu.launch_func @k::@heat_tb blocks in (64, 128, 1) threads in (64, 4, 1) args(...)
// AMD: forall (128, 128) × (8, 32), 4 rows per thread → 50 launches, 16384 workgroups × 256 (4 waves)
// CPU: stencil-forall-to-omp → one omp.parallel, two static loops (passdesign2.md S7)
```

### S6 — inner code

```mlir
// NV: align 38x70 → 38x72 (row = 576 B); no bank padding (a warp reads 32 consecutive f64)
%S0 = memref.alloca() : memref<38x72xf64, #gpu.address_space<workgroup>>
// NV: unroll-jam 8 rows per thread → shared loads per output 5 → (8 + 2)/8 + 2 = 3.25
// AMD: 4 rows per thread → (4 + 2)/4 + 2 = 3.5; pitch 36 unchanged (288 B, 16-B aligned)
// vectorize: skipped on GPU (1 column per thread is already coalesced)
// async-pipeline: skipped (one window per block, no loop to pipeline)
// CPU: as in passdesign2.md S6
```

### S8 — lower

```mlir
// NV: copy_async → TMA; descriptor built on the host
%d = nvgpu.tma.create.descriptor %U box[%c38, %c70] : ...
nvgpu.tma.async.load %d[%x, %y], %mbar to %S0 : ...
// AMD: stencil.barrier → gpu.barrier → s_barrier; binary from #rocdl.target<chip = "gfx942">
// all: fp-contract off (exact)
```

### TIR — `stencil-to-tileir{form=window}` (instead of S5b–S8)

- Star stencil → two products: `out = L·X + X·R`.
  - `L`: 0.1 on the two off-diagonals (n, s).
  - `R`: 0.6 on the diagonal, 0.1 on the off-diagonals (c, w, e).
- 2 × 128 = 256 multiply-adds per output. Loads grow by (128/126)² − 1 = 3.2%.

```mlir
%sv = make_strided_view %tv : strided_view<tile=(128x128), traversal_strides=[126,126], tensor_view<4096x4096xf16, ...>>
%X, %t0 = load_view_tko weak %sv[%wi, %wj] : ... -> tile<128x128xf16>, token      // off-grid → 0
%A  = mmaf %L, %X, %zero : tile<128x128xf16>, tile<128x128xf16>, tile<128x128xf32>
%B  = mmaf %X, %R, %A    : tile<128x128xf16>, tile<128x128xf16>, tile<128x128xf32>
%O  = ftof %B : tile<128x128xf32> -> tile<128x128xf16>
store_ptr_tko weak %ptrs, %O, %inner_mask : ...                                   // rows/cols 1..126 only
// host: 100 launches, 33 × 33 blocks each, ping-pong between two global buffers
```

---

## Appendix B — IR examples for the new passes

From the pipeline's output (`chip=host`, Zen 5, f64) and the lit tests, shortened and with readable names: only the lines a pass changes. B.5 and B.8 are schematic, and loop bounds are written inline instead of as named maps (the real IR has one copy per case). `examples/heat2d_demo.py --dump DIR` writes the IR after every pass.

### B.1 `stencil-inline-fusion` (S3) — chain `lap → blur`, temporary with `constant(0)`

```mlir
// before: two nests per step and a temporary
%tmp = memref.alloc() {stencil.buffer = "temp"} ...
affine.for ... { ... store lap(u) -> %tmp ... }
affine.for ... { ... blur reads %tmp[i, j-1], %tmp[i, j], %tmp[i, j+1] ... }

// after: one nest; u is read through a view grown to lap's reach + blur's reach
stencil.halo_view %u {bc = {kinds = ["constant", "constant"], value = 0.0}, lo = [0, 1, 2], hi = [0, 1, 2]}
%off = arith.constant {stencil.offgrid = array<i64: 0, -1>} false   // placeholder: is (i, j-1) off the grid?
%v   = arith.select %off, %c0, %lap_at_j_minus_1 : f64             // the temporary's rule, or the producer
... {stencil.nest = {inlined, kind = "apply"}}
```

- After S5b no `stencil.offgrid` is left: `true` / `false` in the boundary strips, folded to `false` in the centre.

### B.2 `stencil-fold-ring` (S4) — wave equation

```mlir
// before: three time levels
%ring = memref.alloc() {stencil.buffer = "ring"} : memref<3x64x64xf64>

// after: two, written in place
%ring = memref.alloc() {stencil.buffer = "ring", stencil.ring_inplace} : memref<2x64x64xf64>
%u   = affine.load  %v[%t mod 2,       %i + 1, %j + 1]     // u(t)
%um  = affine.load  %v[(%t + 1) mod 2, %i + 1, %j + 1]     // u(t-1): the slot about to be written
affine.store %new,  %v[(%t + 1) mod 2, %i + 1, %j + 1]     // u(t+1) overwrites u(t-1)
```

### B.3 `stencil-temporal-tile` → `stencil-roll-steps` (S4) — heat2d 1024², tile 16 × 1024, depth 8

```mlir
// after stencil-temporal-tile: one step per tile, the step is a placeholder
affine.for %blk = 0 to 12 {
  %s = arith.constant {stencil.step_placeholder} 0 : index
  affine.for %t = 0 to 64 { ... }          // tile phase
  affine.for %g = 0 to 65 { ... }          // gap phase: one more region than tiles
} {stencil.strategy = "diamond", ...}

// after stencil-roll-steps: a step loop inside each tile
affine.for %blk = 0 to 12 {
  affine.for %t = 0 to 64 {
    affine.for %s = 0 to 8 {
      affine.for %i = max(%t * 16 + %s, 0) to min(%t * 16 - %s + 16, 1024) {    // tile shrinks
        affine.for %j = 0 to 1024 {
          %c = affine.load %ring[%s mod 2, %i + 1, %j + 1] ...
    } } } {stencil.step_loop = 8}
  }
  affine.for %g = 0 to 65 {
    affine.for %s = 0 to 8 {
      affine.for %i = max(%g * 16 - %s, 0) to min(%g * 16 + %s, 1024) {         // gap grows
        ...
} } } }
affine.for %blk = 12 to 13 { ... step loop 0 to 4 ... }                          // last block: 100 = 12 × 8 + 4
```

### B.4 `stencil-tile` sweep slabs (S5a) — `temporal=off`

```mlir
// cost attribute of the chosen tile (1024², 64 cores; 4096² on 32 cores gives 128x4096)
stencil.tiles = {bt = 1, cost = "sweep tile=[16x1024] bt=1 ...", nopack, ...}
```

### B.5 `stencil-pack` (S5b) — versioning and padded ring

```mlir
// versioning, no pack: per tile index
affine.if #first_or_last(%t) {
  ... edge strip: rule applied on each read ...
  ... clean centre: constant bounds, plain reads ...
} else {
  affine.for %i = %t * 16 + %s to %t * 16 - %s + 16 { ... }      // middle tiles: clamps dropped
}

// padded ring (64², reach 1: strips 6.25% ≥ 4%): a real halo, filled once
%ring = memref.alloc() {stencil.buffer = "ring", stencil.pad = array<i64: 1, 1>} : memref<2x66x66xf64>
```

### B.6 `stencil-skew-inner` (S5b) — heat2d 16384², tile 128 rows, skew blocks of 512 columns, depth 64

```mlir
// S4 recorded: roll = {reach = [1, 1], skew = 512, skew_axis = 1, steps = 64}
#blo = affine_map<(k, s) -> (k * 512 - s)>
#bhi = affine_map<(k, s) -> (k * 512 - s + 512)>
affine.for %k = 0 to nblocks {                        // blocks of the contiguous axis
  affine.if #middle(%k) {
    affine.for %s = 0 to 64 {                         // step loop
      affine.for %i = ... {
        affine.for %j = #blo(%k, %s) to #bhi(%k, %s) { ... }    // block moves left by the reach per step
  } } } else { ... first / last blocks: the same, intersected with the grid ... }
} {stencil.skew_loop = 512}
```

### B.7 `stencil-ring-endpoints` (S5b)

```mlir
// before: one ring, copy-in before the time loop, copy-out after it
%ring = memref.alloc() {stencil.buffer = "ring"} : memref<2x1024x1024xf64>

// after: two buffers; the array is selected by the time of the read / write
%A = memref.alloc() {stencil.buffer = "ring"} : memref<1024x1024xf64>
%B = memref.alloc() {stencil.buffer = "ring"} : memref<1024x1024xf64>
%r0  = arith.select %even,    %A,   %B  : memref<1024x1024xf64>
%src = arith.select %is_t0,   %in,  %r0 : memref<1024x1024xf64>     // time 0: the input argument
%dst = arith.select %is_last, %out, %w0 : memref<1024x1024xf64>     // time T: the output argument
// both copy nests are gone
```

### B.8 Bound versioning in `stencil-vectorize` (S6)

```mlir
// before: two-result bounds, not vectorized
affine.for %j = max(%a, %b) to min(%c, %d) { ... scalar ... }

// after: one copy per case, each with single bounds, each vectorized (main loop + remainder)
affine.if #a_ge_b(...) {
  affine.for %j = %a to %split step 8 { ... vector<8xf64> ... }
  affine.for %j = max(%split, %a) to %c { ... remainder ... }
} else { ... }
```
