# Stencil DSL — Pass Design

**Status: DRAFT (2026-11-20).** One document for the whole compiler: the pass framework, every pass, every optimization, the four routes, and the ideas we tried and dropped. It replaces `passdesign1_1.md`, `passdesign2.md`, `PassesDesign.md` and `passdesign_generic.md`. The IR after each pass for heat2d on each route is in `cpuir_heat2d.md`, `tensorir_heat2d.md`, `cudatileir_heat2d.md` and `gpucudair_heat2d.md`. Worked examples are in the appendices.

**How to read it.** Section 1 gives the idea and the decisions. Section 2 lists every pass in pipeline order. Sections 3–4 describe the target description and the options. Section 5 walks through the stages S0–S8, one pass at a time: *purpose, what the pass does, the decision it takes, why, and when it does nothing*. Section 6 describes the four routes. Section 7 covers data types and boundary rules, which cut across all stages. Section 8 lists the ideas that did not work. Section 9 is the implementation status.

---

## 1. The idea and the decisions

### 1.1 The idea in seven points

1. **One pipeline, no duplicated passes.** A pass does the same work on every target. The numbers and capabilities it needs (cache sizes, shared memory, vector width, matrix shapes) come from a **target description**.
2. **Rule:** a pass may branch on a *capability* (`hasAsyncCopy`, `matrixShapes(type)`), never on a *vendor name*.
3. **S0–S5a are shared by all targets.** After S5a each kernel takes one **route**: `cpu`, `gpu-cuda` (NVIDIA), `gpu-rocm` (AMD), `gpu-tensor` (tensor-core methods; ConvStencil is the one implemented) or `tileir` (NVIDIA through `tileiras`).
4. **Shared boundary IR:** parallel loops are `scf.forall` with mapping attributes, not `gpu.launch` or `omp` directly. Each target lowers from there.
5. **All data types:** `i8`–`i64`, `fp8`, `f16`, `bf16`, `f32`, `f64`. The route depends on the type, because matrix units exist only for some types on some chips.
6. **The default GPU route is the direct kernel** (tile in shared memory / LDS, per-thread register blocking). Matrix units are used when the type, `fp-mode` and the cost model allow, or when the user asks for the route.
7. **Under `fp-mode=exact` the result is bit for bit the same as the plain loop** (fp-contract is off in all backends). Anything that reorders float sums needs `relaxed`.

### 1.2 Design decisions

| # | Decision | Why |
|---|---|---|
| 1 | Shared boundary = `scf.forall` + target-specific mapping attributes (`#gpu.block`, `#stencil.cpu_thread`, `#chip.pe`) | CPU and GPUs share S6; it also fits non-SIMT hardware. The `gpu` dialect is used one level lower, shared by NVIDIA and AMD. |
| 2 | Target description = **DLTI** for numbers + **`StencilTargetInterface`** for rules | DLTI is standard and scoped (CPU spec on the module, GPU spec on `gpu.module`). Rules like occupancy are functions, not numbers. A verifier catches missing keys. |
| 3 | Numbers come from a **chip table** (`sm_90`, `gfx942`, `zen5`, …) plus `probe.py` results, overridable by options | Works when cross-compiling; builds are reproducible. |
| 4 | Float matrix units (ConvStencil, TileIR `mmaf`) only with `fp-mode=relaxed` | They reorder sums, and zero weights turn `Inf` into `NaN`. `exact` stays bit for bit. |
| 5 | Integer matrix units are allowed under `exact` if overflow is `wrap` | Wrapping integer arithmetic gives the same bits in any order. |
| 6 | TileIR forks **after S5a**; no pack, no S7 inner mapping, no S6 | `tileiras` owns everything inside a tile. Packing into shared memory ourselves measured ~6 GPt/s against 46 GPt/s for direct loads. |
| 7 | TileIR loads an **overlapping power-of-2 window** (128 × 128, step 126) | Tile shapes must be powers of 2. Doubling (32 → 64) wastes 75% of loads; overlapping wastes ~3%. |
| 8 | Two matmul forms per tensor route: **window + banded matrices** and **gather + stencil2row** | Their costs cross over with radius, rank and type. `auto` picks window for TileIR when eligible and gather for `gpu-tensor`; there is no cost model that chooses between them yet. |
| 9 | f64 tensor route: `nvgpu` DMMA (NVIDIA), MFMA (AMD), only under `relaxed`; otherwise `gpu-cuda` / `gpu-rocm` | `tileiras` lowers f64 `mmaf` to scalar code (measured). |
| 10 | S5 is split: **S5a `stencil-tile`** (shared) and **S5b `stencil-pack`** (not on TileIR) | TileIR needs tiles but not packing. |
| 11 | S6 runs **before** S7 lowers the loops | S6 needs affine loops; after S7 they are `scf` / `omp` / `gpu`. |
| 12 | The CPU thread count is **one number given at compile time** (`threads=N`) | Tiling and the OpenMP `num_threads` clause then agree (see 5.8). |

### 1.3 Routes at a glance

| Route | Hardware | Used for | Status |
|---|---|---|---|
| `cpu` | x86 (AVX-512 / AVX2) | everything on a CPU chip | implemented, measured |
| `gpu-cuda` | NVIDIA (PTX) | default GPU route, every type | implemented, runs on H200 |
| `gpu-rocm` | AMD (ROCDL) | default AMD route | compiles to ROCDL / AMDGPU LLVM IR; no AMD machine here, the AMD plan is checked with `plan_as=gfx942` on NVIDIA |
| `gpu-tensor` | NVIDIA tensor cores via `nvgpu.mma.sync` | f64, f32 (TF32), f16, bf16, i8 | implemented: gather form (ConvStencil) and window form |
| `tileir` | NVIDIA via `tileiras` (cuda_tile bytecode) | f16 / bf16 / i8 on matrix units; any body through the direct form | implemented: direct, window and gather forms |

---

## 2. Pipeline

### 2.1 At a glance

```
stencil IR
  │  S0   target description, fp-mode, overflow mode
  │  S1   analysis + classify each kernel
  │  S2   lower to one affine nest per stencil
  │  S3   fusion: pointwise, inline (producer into consumer), tile-and-fuse decision
  │  S4   fold the ring, temporal blocking (diamond / overlapped), roll the steps into a loop
  │  S5a  tile every nest (sweep slabs on CPU), split interior / boundary tiles, choose the route
  ├─ route per kernel ─────────────────────────────────────────────────────────────┐
  │  cpu / gpu-cuda / gpu-rocm                gpu-tensor          tileir            │
  │  S5b  pack or version boundary tiles,    stencil-to-         stencil-to-tileir  │
  │       skew the long axis,                 gpu-tensor          cuda-tile-translate│
  │       drop the copy-in / copy-out         (own kernel body)   tileiras → cubin   │
  │  S6   align, prefetch, unroll-and-jam, vectorize                                 │
  │  S7   map scf.forall → omp | gpu.launch                                          │
  ▼  S8   lower to LLVM IR → CPU object (opt -O3, llc) | PTX / cubin | HSACO
```

- The pass list is the **same for every chip** up to S7. Route-specific passes do nothing on the other routes. Only the S8 tail depends on the chip kind (CPU → LLVM, NVIDIA → NVVM / PTX, AMD → ROCDL).
- `gpu-tensor` and `tileir` build their kernels inside S5a (`stencil-to-gpu-tensor`, `stencil-to-tileir`); the S5b–S7 passes then find nothing to do on them.
- The stage names are given to `stop-after=S0..S8` (inspection).

| Stage | CPU | NVIDIA (`gpu-cuda`) | AMD (`gpu-rocm`) | `gpu-tensor` | `tileir` |
|---|---|---|---|---|---|
| S0–S5a | shared | shared | shared | shared | shared |
| S5b pack | boundary tiles only; diamond and sweep tiles are versioned, not packed | all tiles → shared memory (`cp.async`) | all tiles → LDS | — | — |
| S6 | SIMD, unroll-jam | per-thread blocking | per-thread blocking | — | — (`tileiras`) |
| S7 | forall → `omp` | forall → `gpu.launch` | forall → `gpu.launch` | built in S5a | tile → one TileIR block |
| S8 | `llvm` | `nvvm` → PTX | `rocdl` → HSACO | `nvvm` → PTX | `cuda_tile` bytecode |

### 2.2 All passes, in pipeline order

Passes marked **(new)** were added after the first comparison with Pluto / YASK / Devito / xDSL. Pass names are those of `include/StencilDSL/Transforms/Passes.td`; the pipeline text is built in `lib/Pipelines/Pipelines.cpp`.

| Stage | Pass | What it does | Applies to |
|---|---|---|---|
| S0 | `stencil-select-target` | Attaches chip numbers (DLTI), fp-mode, overflow mode, strategy options and overrides | all |
| S0 | `stencil-verify-target` | Fails if a number the target rules need is missing | all |
| S1 | `stencil-infer-halo` | Reach of each stencil per axis | all |
| S1 | `stencil-model-boundary-conditions` | Records the boundary rule per axis | all |
| S1 | `stencil-decide-bounds-mode` | Records static or dynamic shapes | all |
| S1 | `stencil-infer-layout` | Records the contiguous axis of each field | all |
| S1 | `stencil-alias-contract` | Which arguments must not overlap, and the run-time action (`on-overlap=copy|error`) | all |
| S1 | `stencil-classify` | Element type, linear or not, star or box, radius, time loop | all |
| S2 | `stencil-group-weights` | `fp-mode=relaxed`: adds the points that share a weight, multiplies once | all |
| S2 | `stencil-to-affine` | One affine nest per stencil over halo views; time loop over a ring buffer; marks `stencil.nest`, `stencil.time_loop` | all |
| S3 | `stencil-affine-fusion` | Fuses nests that read the same point | all |
| S3 | `stencil-inline-fusion` **(new)** | Substitutes a producer into the consumer that reads it with a halo; the temporary disappears | CPU (on); `gpu-cuda`, `gpu-rocm`, `gpu-tensor` (opt-in, off by default); never TileIR |
| S3 | `stencil-tile-and-fuse` | Decides the remaining fused chains and their tile (S5a builds them) | all |
| S4 | `stencil-fold-ring` **(new)** | 3 time levels → 2, written in place (wave equation) | CPU, diamond |
| S4 | `stencil-temporal-tile` | Chooses strategy, tile and depth; emits diamond or overlapped tiles | all |
| S4 | `stencil-roll-steps` **(new)** | One step loop per tile instead of one code copy per step | CPU (GPU opt-in) |
| S5a | `stencil-tile` | Tiles nests that are not time-blocked (CPU: sweep slabs); interior / boundary split | all |
| S5a | `stencil-choose-route` | `cpu`, `gpu-cuda`, `gpu-rocm`, `gpu-tensor` or `tileir` per kernel | all |
| S5a | `stencil-to-tileir{form}` | Builds the TileIR module for kernels on that route. `form`: `direct` (plain loads, any body), `window` (overlapping window × banded matrices), `gather` (stencil2row gather × weight matrix); `auto` takes window when eligible, else direct | TileIR |
| S5a | `stencil-to-gpu-tensor{form}` | Builds the tensor-core kernels (gather or window) for kernels on that route | `gpu-tensor` |
| S5b | `stencil-pack` | Packs tile inputs, or versions boundary tiles without packing; padded ring | CPU, `gpu-cuda`, `gpu-rocm` |
| S5b | `stencil-lower-pack` | Pack ops → loops or async copies | same |
| S5b | `stencil-verify-no-halo-view` | Fails if a halo view is left | same |
| S5b | `stencil-skew-inner` **(new)** | Cuts the long (contiguous) axis into time-skewed blocks | CPU, memory-bound |
| S5b | `stencil-ring-endpoints` **(new)** | First step reads the input, last step writes the output; copies removed | CPU |
| S6 | `stencil-align-buffers` | Buffer pitch to the vector width / copy alignment | CPU, `gpu-cuda`, `gpu-rocm` |
| S6 | `stencil-prefetch` | Prefetch the next tile's window (`prefetch=on`, default off) | CPU |
| S6 | `stencil-unroll-jam` | Several outputs per inner iteration | CPU, `gpu-cuda`, `gpu-rocm` |
| S6 | `stencil-vectorize` | SIMD on the contiguous axis with a scalar remainder; splits max/min bounds so boundary blocks vectorize too | CPU |
| S7 | `stencil-map-parallel` | Loops → parallel levels (`scf.forall` + mapping) | CPU, `gpu-cuda`, `gpu-rocm` |
| S7 | `stencil-forall-to-omp` | `scf.forall` → OpenMP, gap affinity | CPU |
| S7 | `stencil-forall-to-gpu` | `scf.forall` → `gpu.launch` | GPU |
| S8 | `stencil-lower-target-ops` | Lowers `copy_async`, `barrier`, `matrix_contract` | all |
| S8 | `stencil-static-buffers` | Compiler-owned buffers kept across calls (`buffers=static`) | CPU |
| S8 | `stencil-emit-alias-check` | Entry wrapper that checks argument overlap at run time (copy or error) | CPU |
| S8 | `stencil-align-workgroup-mem` | Aligns the shared-memory (workgroup) allocations of the outlined kernels | GPU |
| S8 | `stencil-emit-host-schedule` | Records buffers, loops and launches of the host function as JSON | GPU |
| S8 | upstream MLIR lowering | affine → scf → cf → LLVM dialect on CPU; `gpu-module-to-binary` on GPU | all |

- **Not every pass fires on every stencil.** Each pass checks its own condition and otherwise leaves the IR unchanged (for example fold-ring only fires on a 3-level ring, skew-inner only when the grid does not fit in cache). The per-route IR files list only the passes that change the IR.
- **Design-only, not built:** `stencil-map-to-matrix-units`, `stencil-async-pipeline`.

---

## 3. Target description

**Numbers (DLTI keys, per device).** Memory levels (size, hardware- or software-managed), parallel levels (units, warp / wave size, max threads), SIMD bits, vector registers, NUMA nodes, bandwidth, peak FLOP/s per type, async-copy kind (`none`, `tma`, `cp_async`, `lds_direct`), matrix types per route (from `probe.py`), and the measured route efficiencies `eff_direct_pct`, `eff_tileir_pct` used by the route choice. Source: `lib/Target/ChipTable.cpp`.

**Rules (`StencilTargetInterface`).**

| Method | Used by | CPU | SIMT GPU (NVIDIA + AMD) |
|---|---|---|---|
| `fastMemory()` | S3, S4, S5b | L2, hardware-managed | shared memory / LDS, software-managed |
| `occupancy(bytes, regs)` | S3, S4 | always 1 | blocks per SM / CU |
| `innerWidth(type)` | S3, S5a | vector width | 128-byte coalescing |
| `padPitch(pitch, type)` | S6 | vector width | async-copy alignment + bank rule |
| `parallelLevels()` | S7 | thread (+ SIMD) | block, thread |
| `mappingAttr(level)` | S7 | `#stencil.cpu_thread` | `#gpu.block`, `#gpu.thread` |
| `matrixShapes(type, route)` | route choice, S6 | none | DMMA / MFMA / `mmaf` shapes |

- NVIDIA and AMD share **one** implementation (`#stencil.simt_gpu`); they differ only in numbers.
- The LLVM backend is picked by the `gpu.module` target attribute (`#nvvm.target`, `#rocdl.target`), not by the interface.

---

## 4. Options

### 4.1 Pipeline options (`stencil-pipeline-auto`, also the `compile()` arguments)

| Option | Values (default first) | Meaning |
|---|---|---|
| `chip` | `host`, `sm_90`, `gfx942`, … | Which chip-table entry to use |
| `fp-mode` | `exact`, `relaxed`, `lowp` | `exact`: bit for bit. `relaxed`: reordering allowed (weight grouping, matrix units). `lowp`: also TF32 for f32 |
| `overflow` | `wrap`, `saturate` | Integer overflow rule. `saturate` only keeps operations in written order (the DSL has no saturating arithmetic) |
| `backend` | `auto`, `cpu`, `gpu-cuda`, `gpu-rocm`, `gpu-tensor`, `tileir` | Force a route |
| `tau` | 0.10 | Most repeated work accepted by fusion and temporal blocking |
| `on-overlap` | `copy`, `error` | What the entry wrapper does when arguments overlap |
| `threads` | 0 (= chip table `units`) | CPU thread count; ignored on GPU |
| `temporal` | `auto`, `overlapped`, `diamond`, `off` | Temporal-blocking strategy |
| `nontemporal` | `auto`, `on`, `off` | Streaming stores for large outputs (CPU) |
| `prefetch` | `off`, `on` | Prefetch the next tile (CPU) |
| `buffers` | `heap`, `static` | Compiler-owned buffers per call or kept across calls |
| `tileir-form` | `auto`, `direct`, `window`, `gather` | TileIR kernel form |
| `tensor-form` | `auto`, `gather`, `window` | `gpu-tensor` kernel form (`auto` = gather) |
| `cluster` | `off`, `on` | Hopper thread-block clusters for `gpu-cuda` (off; see section 8) |
| `override` | `key=value,…` | Chip-table overrides, e.g. plan an AMD kernel on NVIDIA |
| `stop-after` | `S0`…`S8` | Stop after a stage (inspection) |

### 4.2 Chip-table overrides (tuning and experiments)

`inline_fusion` (CPU, 1), `inline_fusion_gpu` (0), `fold_ring` (1), `diamond_roll` (1), `diamond_roll_gpu` (0), `diamond_skew` (1), `diamond_tile<d>`, `diamond_bt`, `diamond_smem` (0), `diamond_threads`, `diamond_affinity`, `diamond_penalty_pct`, `sweep_slabs` (1), `ring_endpoints` (1), `pad_ring_pct` (4), `eff_direct_pct`, `eff_tileir_pct`, `units`. They are read by the passes through the target description, so a build with an override is reproducible.

---

## 5. The stages, pass by pass

### 5.1 Stage 0 — Select target

**Purpose:** attach everything later stages need.
**Passes:** `stencil-select-target{chip, fp-mode, overflow, backend, …}`, `stencil-verify-target`.

- Attaches the interface attribute, the DLTI numbers, and `#nvvm.target` / `#rocdl.target` on each `gpu.module`.
- **Decision:** which numbers. **Why:** from the chip table, so a build is reproducible and can target another machine.
- `stencil-verify-target` fails if a key the interface needs is missing (for example `vregs`).

### 5.2 Stage 1 — Analysis and classification

**Purpose:** collect the facts every later decision needs.
**Passes:** `stencil-infer-halo`, `stencil-model-boundary-conditions`, `stencil-decide-bounds-mode`, `stencil-infer-layout`, `stencil-alias-contract`, `stencil-classify`.

- **infer-halo** computes the reach of each stencil per axis from its offsets. The reach decides the halo of every view, the size of every tile window, and how fast a diamond tile shrinks.
- **model-boundary-conditions** records the boundary rule per axis (`constant(c)`, `replicate`, `reflect`, `periodic`, or none). A `constant(c)` value must be representable in the element type.
- **decide-bounds-mode** records whether the shapes are static or dynamic. Static shapes allow constant tile counts and bounds.
- **infer-layout** records the contiguous axis of each field; vectorization and the tile shape keep that axis long.
- **alias-contract** decides which arguments must not overlap and what the entry wrapper does if they do (`on-overlap`).
- **classify** records per kernel: element type, linear with constant weights or not, star or box, radius, time loop, in-place, overflow mode. The route choice (S5a) reads it.
- **Decision:** none yet. The route is chosen after S5a, when fusion is known.

### 5.3 Stage 2 — Lower to affine

**Purpose:** one affine nest per stencil, reading its inputs through halo views; the time loop runs over a ring buffer.
**Passes:** `stencil-group-weights`, `stencil-to-affine`.

- **group-weights** (`fp-mode=relaxed` only): adds the points that share a weight, then multiplies once. Heat: `0.6·c + 0.1·(n + s + w + e)` instead of five multiplies.
- **to-affine** emits the nest, marks it with `stencil.nest` and the time loop with `stencil.time_loop`, and allocates the **ring buffer** (two or three time slots). A 100-step program becomes one loop over time that reads slot `t mod 2` and writes slot `(t+1) mod 2`.
- **Data types:** arithmetic stays in the source type under `exact`.
  - Under `relaxed`, `f16` / `bf16` / `fp8` accumulate in `f32`.
  - Integers with `wrap` may accumulate in a wider type (`i8` → `i32`) and truncate afterwards. The result is identical, so this is allowed under `exact`.
  - `saturate` is not associative: no widening, no weight grouping, no folding.

### 5.4 Stage 3 — Fusion

**Purpose:** fewer passes over memory.
**Passes:** `stencil-affine-fusion`, `stencil-inline-fusion` (new), `stencil-tile-and-fuse`.

- **`stencil-affine-fusion`** fuses nests that read each other at the same point (pointwise fusion), for example `lap` and `u_new` of the heat step become one nest.
- **`stencil-inline-fusion`** (CPU on, off with `inline_fusion=0`; GPU off by default, `inline_fusion_gpu=1` turns it on; never on the TileIR route):
  - A producer that writes a temporary which is read with a halo is substituted into its consumer, once per read of the temporary. The consumer must be the only reader. Size limits: at most 9 reads of the temporary, 16 reads in the producer, product ≤ 64.
  - Cost: the producer is recomputed once per read. Gain: the temporary disappears and the step becomes **one** nest, so S4 can use diamond tiles on it.
  - The halo view of the input grows by the shift (consumer's reach + producer's reach).
  - Boundary rule of the temporary: `constant` / none: the read becomes `select(off-grid, c, producer)`, where `off-grid` is a placeholder that S5b resolves in the boundary strips and folds to `false` elsewhere. `replicate` / `reflect`: the copied reads are tagged with the shift and the rule, and S5b remaps them in the strips. `periodic`: not inlined (`stencil-tile-and-fuse` handles it).
  - **GPU:** the rewrite is the same; only the gate differs. It is off by default until a cost model chooses between inlining and tile-and-fuse. That model does not exist. **TileIR never inlines**: its lowering does not resolve the placeholders, and boundary cells came out wrong when it was forced (see section 8). Checked bit for bit on `auto`, `gpu-cuda`, `gpu-tensor` and `tileir` (`test_inline_fusion_gpu`, 20/20); the speed benefit on GPU was **not measured**.
- **`stencil-tile-and-fuse`** only decides: it marks a chain and its tile. S5a builds the chain with the same generator S4 uses.
- **Decision (tile size)**, the same rule for every target with the numbers from the interface: footprint ≤ the `fastMemory()` share (CPU: 50% of L2; GPU: shared memory / LDS ÷ target blocks per unit); inner width a multiple of `innerWidth(type)`; repeated work ≤ τ; enough tiles for all units.
- The rolling buffer of S2 is the same transformation as GPU 2.5D streaming (walk the outer axis, keep a few planes).

### 5.5 Stage 4 — Temporal blocking

**Purpose:** run b_t time steps per tile to cut memory traffic.
**Passes:** `stencil-fold-ring` (new), `stencil-temporal-tile`, `stencil-roll-steps` (new).

- **`stencil-fold-ring`** (CPU; off with `fold_ring=0`). Applies when a step keeps 3 time levels and the oldest, u(t−1), is read only at the point being written (wave equation). The new value overwrites u(t−1) in place, so 3 slots become 2 and the step becomes diamond-tileable. The overlapped form and the GPU routes keep 3 slots.
- **`stencil-temporal-tile`.** `temporal=off | overlapped | diamond | auto`; `off` leaves the step loop alone (S5a then cuts sweep slabs).
  - **Overlapped tiles:** each tile loads its window grown by b_t · reach, computes b_t steps on shrinking regions in scratch buffers, and writes only its own cells. Some work is repeated at the tile edges (the *redundancy*).
  - **Diamond tiles:** along each split axis the tile region shrinks by the reach every step while the gap region between two tiles grows. 2^(split axes) **phases** (a tile-or-gap bit per axis) are launched in rank + 1 **waves** (wave = number of gap axes). There is no repeated work, and the tiles read the arrays directly (no pack). Terminology: *phase* = which axes are tile or gap (the thing that is launched); *wave* = how many axes are gap (what fixes the launch order). A gap axis has one more region than tiles.
  - **Decision:** strategy, tile and b_t. Overlapped: the tile and b_t with the lowest modelled memory traffic per point per step that pass τ and the footprint. Diamond: the ranking below. `auto` takes diamond if its modelled traffic × `diamond_penalty_pct` is below overlapped's traffic × (1 + redundancy).
  - In-place (folded) or inlined steps: diamond or no blocking only. A forced diamond shape that cannot be built gives a warning.
  - **CPU diamond chooser** (numbers from the chip table). Every tile must fit L2 plus this core's share of L3 (both ring slots of the tile's window) and be wider than 2 · reach · (b_t − 1).
    - *Fits in cache* (both ring slots ≤ all L2 + all L3 of the machine). Ranking: keep the contiguous axis whole, split as few axes as possible, largest depth (up to 8), about 4 tiles per core, lowest traffic.
    - *Memory-bound* (anything larger). Ranking: contiguous axis not cut below 512 points, largest depth (up to 64), power-of-two widths before in-between ones, tile footprint closest to L2, longest contiguous axis. In-between widths (12 … 124, step 4) are also tried on the outer axes.
    - *1-D:* both slots of the tile in L1, then the largest depth (up to 1024), then the widest tile.
    - *Skewed axis:* on a memory-bound grid, if the chosen shape cuts the contiguous axis and at least one other axis, the contiguous axis is not diamond-split; it is recorded as a skew block width for `stencil-skew-inner` (`diamond_skew=0` uses a diamond split instead).
    - Shapes chosen on the 64-core Zen 5: heat2d 4096² 16 × 4096, depth 8; heat2d 16384² 128 rows × the whole row, skewed in blocks of 512, depth 64; heat3d 512³ 20 × 20 × 512, depth 10; heat1d 2048 wide, full depth.
- **`stencil-roll-steps`** (CPU; off with `diamond_roll=0`). S4 emits **one** step per tile with a step placeholder; this pass wraps it in a step loop. Tile bounds shrink by the reach per step, gap bounds grow: tile `[tW + s·r, tW + W − s·r)`, gap `[tW − s·r, tW + s·r)`, clamped to the grid. It removes the old depth cap of 8 (each step used to be a separate copy of the code).
  - **GPU** (`diamond_roll_gpu=1`, default off): the step loop runs inside the block, a `stencil.barrier` ends each step, the depth cap is 32. It is bit-exact but gave **no speed gain** (section 8).
- **Strategies allowed per target:**

| Target | Allowed | Why |
|---|---|---|
| CPU | overlapped, diamond | `auto` takes diamond (measured faster). |
| `gpu-cuda` | overlapped, diamond | `auto` takes overlapped: diamond costs one launch per phase and a barrier per thin gap step. |
| `gpu-tensor` gather form | overlapped chains, diamond (one launch per stage) | Mirrors `gpu-cuda`; the window form is single-step only. |
| `tileir` | b_t = 1 | Step-to-step shifted reads inside a tile go through memory (measured slow). |

- **Time loop when `temporal=off`:** S5a cuts the single step into sweep slabs (CPU) or tiles (GPU).

### 5.6 Stage 5a — Tile, choose the route, build the special routes

**Purpose:** every nest tiled; each tile classified once as interior or boundary; each kernel gets a route.
**Passes:** `stencil-tile`, `stencil-choose-route`, `stencil-to-tileir`, `stencil-to-gpu-tensor`.

- **`stencil-tile`** tiles the nests S4 did not block (b_t = 1) and builds the chains S3 marked.
  - **CPU sweep slabs** (single-stage nest; off with `sweep_slabs=0`): outer axis: one slab per core; contiguous axis: kept whole; middle axes (3D): cut only until the moving front fits L2 + this core's share of L3; slabs are not packed (boundary slabs are versioned in S5b).
  - On TileIR the tile is `window − 2R` (window a power of 2), for example 126 × 126 from a 128 × 128 window.
  - Interior tiles and boundary tiles are told apart once here (`affine.if #interior`), so later passes treat them separately.
- **`stencil-choose-route`** (per kernel):
  1. `backend=` given → that route.
  2. CPU → `cpu`.
  3. GPU → a **tensor route** is possible if all hold: the fused body is linear with constant weights; `matrixShapes(type, route)` is not empty; floats need `fp-mode=relaxed`; integers need `overflow=wrap`; `f32` through TF32 needs `fp-mode=lowp`.
  4. Take the tensor route unless the cost model predicts the direct kernel more than 10% faster, else the direct route (`gpu-cuda` on NVIDIA, `gpu-rocm` on AMD).
  - **Measured efficiencies** are in the chip table (`eff_direct_pct`, `eff_tileir_pct`). On the H200 the direct kernel is predicted faster than the TileIR window form (measured 316 against 106 GPoint-steps/s, f16 heat 4096²), so `auto` takes `gpu-cuda`; `backend=tileir` forces TileIR.
  - **Automatic choice of `gpu-tensor`** happens only for f64 and f32 (`relaxed`, linear one-stage stencil). f16, bf16 and i8 run on it only with an explicit `backend=gpu-tensor`, because the native kernels are not yet faster than the direct kernel.

| Type | NVIDIA tensor route | AMD tensor route (gfx942) | Otherwise |
|---|---|---|---|
| `f64` | `nvgpu` DMMA (`gpu-tensor`) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `f32` | TF32 (`lowp` only) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `f16`, `bf16`, `fp8` | `tileir` (and `gpu-tensor` for f16 / bf16) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `i8` | `tileir` (native s8 matrix unit; also `gpu-tensor` m16n8k32) | MFMA | `gpu-cuda` / `gpu-rocm` |
| `i16`, `i32`, `i64` | none | none | `gpu-cuda` / `gpu-rocm` |

  - Kernels of one program may take different routes; the host code launches each.
- **`stencil-to-tileir{form=auto|direct|window|gather}`** builds the TileIR kernel. The **window** form loads one overlapping window and multiplies by banded matrices; the **gather** form gathers the neighbours into a matrix and multiplies by a weight matrix (both in 6.5).
- **`stencil-to-gpu-tensor{form=auto|gather|window}`** builds the tensor-core kernel. The **gather** form (default) is ConvStencil's stencil2row; the **window** form stages a shared-memory window and multiplies by banded tables (both in 6.4).

### 5.7 Stage 5b — Pack and boundary versioning

**Purpose:** give each tile its input with the boundary rule applied, at the lowest cost.
**Passes:** `stencil-pack{scope, memspace}`, `stencil-lower-pack`, `stencil-verify-no-halo-view`, `stencil-skew-inner` (new), `stencil-ring-endpoints` (new). Not run on `tileir` or `gpu-tensor` kernels.

- **`stencil-pack`**, three ways to handle the boundary:
  - **Pack** (overlapped tiles; all GPU tiles): copy the window into a buffer with the rule applied. CPU: boundary tiles only, thread-local. GPU: every tile, into shared memory / LDS.
  - **Version without packing** (CPU diamond tiles and sweep slabs): per tile index, first and last tiles (and any tile that reaches the edge when the grid is not a multiple of the tile) get constant bounds and are cut into edge strips plus a clean centre; the rule is applied only in the strips; middle tiles drop the clamps, so their loops have single-expression bounds. Without this, boundary tiles applied the rule on every read in scalar code (the cause of an early 0.8× result, even on one thread).
  - **Padded ring** (CPU, `constant` rule, only when no tile is packed; `pad_ring_pct`, default 4): if the strips would cover ≥ 4% of the grid (sum over axes of 2 · reach / extent), the ring gets a real halo filled once with the constant; no rule code is left at all. Cost: a ring slot no longer has the shape of the arguments, so the copy-in and copy-out stay.
  - Also resolves the `off-grid` placeholders and tagged reads left by `stencil-inline-fusion`.
- **`stencil-lower-pack`** emits `stencil.copy_async` when `asyncCopy != none`, else plain loops.
- **`stencil-verify-no-halo-view`** fails the pipeline if a `stencil.halo_view` is left, so no rule is silently dropped.
- **`stencil-skew-inner`** (CPU, rolled diamond tiles for which S4 recorded a skewed axis). It cuts the contiguous axis into blocks of width B around the step loop. Block k at step s covers `[kB − s·r, kB + B − s·r)`, so a block's data stays in L2 for all steps. Middle blocks get single-expression bounds; first and last intersect with the grid.
- **`stencil-ring-endpoints`** (CPU; not with a padded ring or the overlapped strategy; off with `ring_endpoints=0`). The 2-slot ring becomes two buffers A and B. The array of each read and write is selected by the time, outside the point loops: time 0 reads the input argument, time T writes the output argument. The copy-in and copy-out nests are deleted.
- **GPU boundary rule:** `constant(0)` needs nothing more when the async copy fills off-grid cells with zero; other rules add a small fix-up loop in boundary tiles only.

### 5.8 Stage 6 — Inner code

**Purpose:** fast code inside a tile. Runs **before** S7 lowers the loops.
**Passes:** `stencil-align-buffers`, `stencil-prefetch`, `stencil-unroll-jam`, `stencil-vectorize`.

| Pass | CPU | GPU |
|---|---|---|
| `align-buffers` | pitch to vector width | pitch to async-copy alignment; bank padding only if a warp reads down a column |
| `prefetch` | next tile's window into L2 (`prefetch=on`; default off, no gain measured) | — |
| `unroll-jam` | U outputs along the reuse axis | U rows per thread (register blocking) |
| `vectorize` | SIMD, main loop + scalar remainder | up to 128-bit loads per thread; thread guard instead of masks |

- **Bound versioning** (in `stencil-vectorize`): a loop whose bounds are still `max(a, b)` / `min(c, d)` is copied under `affine.if` into single-bound versions, so boundary and skewed blocks vectorize too. The remainder loop starts at `max(split, lower bound)`, so it never runs on an empty range.
- **Streaming stores** (`nontemporal=auto`): for final outputs larger than L3, also through a `select`; 64-byte aligned full vectors only, `sfence` after the nest.
- Unroll-and-jam and vectorization need single-expression loop bounds. Interior tiles have them; CPU boundary tiles get them from S5b's versioning and S6's bound versioning. Packed boundary tiles stay scalar.

### 5.9 Stage 7 — Parallelize

**Purpose:** map tiles and work inside a tile onto the hardware.
**Passes:** `stencil-map-parallel` (shared), then `stencil-forall-to-omp` (CPU) or `stencil-forall-to-gpu` (NVIDIA, AMD; uses upstream forall-to-block / thread mapping).

- **Thread count (CPU) — design choice.** The thread count is ONE number given by the user at compile time (`compile(..., threads=N)`; the same as `override=units=N`). The chip table's `units` (zen5: 64) is only the default. GPU chips ignore it.
  - *How:* S0 stores N as `units`. The tile passes (S4, S5a, cost model) read `units` to cut the grid; `stencil-forall-to-omp` reads the same `units` and writes `num_threads(N)` on `omp.parallel`. The generated code therefore contains N.
  - *Why:* before, tiling used the chip table and the thread count came from `OMP_NUM_THREADS` at run time, so they could disagree (64 tiles on 48 threads: 16 threads get two tiles, and the run waits for the slowest).
  - *Consequences:* `OMP_NUM_THREADS` is not read by the generated code. Changing N recompiles (the code cache is keyed by the code). The tile count is about N, not always exactly N.
  - *Trade-off:* a thread-count change costs a recompile. A run-time choice would avoid it but brings the uneven load back, so it was rejected.
- **Decision (shared):** which loop goes to which parallel level, and the per-thread work (rows per thread = the S6 unroll factor).
- **Time loop:** CPU keeps it inside one parallel region. GPU makes one kernel launch per time block, because there is no synchronization across blocks inside a kernel.
- **Scratch buffers:** CPU once per thread; GPU in the block's shared memory / LDS.
- **Interior and boundary tiles:** CPU one worksharing loop over a linear tile index; GPU one launch with a branch per block (no divergence: the whole block takes one side).
- **Diamond phases:** each phase is one parallel loop (CPU) or one launch (GPU). **CPU gap affinity:** a gap phase reuses the tile phase's static blocks, so a gap runs on a thread that computed one of its neighbours.
- **GPU diamond blocks:** gap blocks are sized from their mean region (about 4 points per thread, at least one warp), not from the tile.

### 5.10 Stage 8 — Lower

**Purpose:** machine code.
**Passes:** `stencil-lower-target-ops`, then per target.

- `stencil-lower-target-ops` lowers the three target-neutral ops: `stencil.copy_async` / `wait`, `stencil.barrier`, `stencil.matrix_contract`.
- **CPU only:** `stencil-static-buffers` (`buffers=static`), `stencil-emit-alias-check`, then the upstream lowering to the LLVM dialect (affine → scf → cf → LLVM, vectors, math, memref with bare pointers, OpenMP).
- **GPU only:** `gpu-kernel-outlining`, `stencil-align-workgroup-mem`, `convert-gpu-to-nvvm` / `-rocdl`, `gpu-module-to-binary`, `stencil-emit-host-schedule`.

| | copy_async | barrier | matrix_contract | then |
|---|---|---|---|---|
| CPU | — | — | — | `omp` → `llvm`, native compile with `-mcpu` |
| NVIDIA | `cp.async` (TMA when the chip has it) | `gpu.barrier` | `nvgpu.mma.sync` (DMMA) | `convert-gpu-to-nvvm`, `gpu-module-to-binary` |
| AMD | `amdgpu.gather_to_lds` or loads | `gpu.barrier` | `amdgpu.mfma` | `convert-gpu-to-rocdl`, `gpu-module-to-binary` |

- Under `exact`, fp-contract is forced off in all three backends (NVPTX fuses multiply-add by default).
- **CPU runtime (`runtime_cpu.py`):** runs LLVM `opt -O3` before `llc` (`STENCIL_LLVM_OPT=none` skips it; without it index arithmetic stayed inside the inner loops); medium code model, so static buffers above 2 GB link; pins one thread per physical core (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`) unless the user set it.
- **GPU runtime (`runtime_gpu.py`):** interprets the `stencil.host_schedule` JSON (buffers, loops, launches) on the PyTorch stream.
- **TileIR route:** `stencil-to-tileir{form=window|gather}` → `cuda-tile-translate` → `tileiras`. Print with `--mlir-print-elementsattrs-with-hex-if-larger=-1`.

---

## 6. The routes in detail

### 6.1 `cpu`

The full S0–S8 path above. What makes it fast: diamond or overlapped time tiles with whole contiguous rows, versioned boundary tiles (no scalar rule code in the centre), skewing for memory-bound grids, ring endpoints without copies, vectorized inner loops with bound versioning, static buffers, one pinned thread per core, gap affinity.

CPU memory-traffic options (measured on the 64-core Zen 5, interleaved runs, `test/python/bench_cpu.py`):

| Option | What it does | Gain | Default |
|---|---|---|---|
| `nontemporal=auto` | Streaming stores for final outputs larger than L3 | 1.04–1.08× on a single-step memory-bound stencil; none when blocked; 0.83× if forced on data that fits in L3 | `auto` |
| `prefetch=on` | Prefetch the next tile's window into L2 | none measurable (Zen 5's hardware prefetchers already cover it) | `off` |
| `buffers=static` | Rings and temporaries kept across calls | per-call cost 22.8 → 1.7 ms; heat 100 steps 1.47× | `heap` (static is not reentrant) |

The time-blocked kernel is compute-bound at about 6 cycles per point.

### 6.2 `gpu-cuda` and `gpu-rocm` (direct kernels)

Tile in shared memory (NVIDIA, `cp.async`, TMA descriptor on chips that have it) or LDS (AMD), per-thread register blocking from `unroll-jam`, one launch per time block (overlapped) or per diamond phase. Strategy `auto` takes overlapped tiles: diamond costs one launch per phase and a barrier per thin gap step.

- **Diamond on GPU:** shared-memory staging (two buffers per tile; fill only what the tile did not compute; write through to global) exists but is opt-in (`diamond_smem=1`), because L1 already caches the direct reads. Nsight Compute (f64, 8 steps, 64 × 32 tiles): the tile phase is 2.4× faster per step than `off`; the three gap phases do about 12% of the work but take about as long again; barrier stalls are 52–60%. Right-sizing gap blocks recovered part of this; the rest is the per-step barrier on thin regions.
- **Auto rule:** diamond if modelled traffic × `diamond_penalty_pct` is below overlapped's traffic × (1 + redundancy). The penalty is 1.0 on CPUs and 6.9 on GPUs (measured f64 time ratio 1.52 / traffic ratio 0.22).
- **AMD:** no AMD machine is available; the plan is checked with `plan_as=gfx942` on NVIDIA and the code compiles to ROCDL and AMDGPU LLVM IR. This LLVM build lacks the AMDGPU target for the final binary.

### 6.3 Route choice summary

See 5.6. A program may mix routes: each kernel is routed independently.

### 6.4 `gpu-tensor` (ConvStencil on tensor cores)

`lib/GpuTensor/StencilToGpuTensor.cpp` (gather form), `lib/GpuTensor/WindowForm.cpp` (window form), `TensorBuilder.h` (shared helpers), `ConvStencilCostModel.cpp` (planner). The pass is `stencil-to-gpu-tensor{form=gather|window}`; the option `tensor-form` picks it, and every kernel launch carries `stencil.tensor_form`.

**Data types.** f64 (`mma m8n8k4`), f32 (TF32, `m16n8k4`), f16 and bf16 (`m16n8k16`, f32 accumulate), i8 (`m16n8k32`, s32 accumulate; the weights are taken modulo 256 into [−128, 127]). The native kinds (f16, bf16, i8) run on the unconverted data; their tile fill is a predicated load and store because `cp.async` cannot copy 1- or 2-byte elements. `i16`, `i32`, `i64` have no tensor-core instruction. The route is off in `fp-mode=exact`, even when forced.

**Boundary rules.** Every rule works. The tile fill applies the rule with `emitBoundaryLoad`. Between fused steps of an overlapped chain, ghost cells are pinned to the constant, left as the real wrapped cells (periodic) or copied from the interior after the stage (replicate, reflect). Periodic is not taken in a diamond tile.

**Gather form (default; ConvStencil).** Stencil2row: for each output tile a data matrix `A[M, K]` of gathered neighbours is multiplied with a weight matrix on the tensor cores. The cost model (`ConvStencilCostModel`) searches the points per MMA, chunks, lead tile, padding and warps. Rank 1 to 3.
- **Temporal blocking:** an overlapped fused chain (`lowerFusedChain`: ring slot computed on the device as `(rawIv + shift) % slots`, ping-pong shared buffers sized by a backward recurrence from the known-good single-step plan) and diamond tiles (`lowerDiamondPhase`: one `gpu.launch_func` per stage, tile shape forced to the diamond tile through `planForTile`, epilogue store clipped to the stage's own output region).
- A stage with several ring reads at different (shift, slots) pairs (leapfrog, `wave`) is rejected explicitly, not mis-lowered: one ring slot is loaded per launch.

**Window form.** The input window is staged in shared memory with plain loads, and `out = Σ_i X_i · T_i`: `X_i` is the window shifted by row tap `i` (a plain M × K block, so an A fragment is consecutive cells), `T_i[k][n] = w[i][k − n]` is a banded compile-time table (one B fragment per lane). Per type: f64 m8n8k4, f32 TF32 m16n8k4, f16 / bf16 m16n8k16, i8 m16n8k32; N = 8 outputs per MMA; `ceil((8 + nLast − 1) / K)` steps per tap tuple; M = 16 rows (f64 8) along the second-to-last axis (rank 1 treats the line as rows of S cells that overlap by the halo). A block is 32 rows × 128 columns (smaller if the shared tile does not fit); the shared-memory row stride is the one with the fewest simulated bank wavefronts. Rank 1 to 3, every boundary rule, the five types.
- **One step per launch:** it needs `temporal=off`. A fused chain (b_t > 1) or a diamond tile is an error that names `temporal=off` / `tensor-form=gather`, not a silent fallback.
- Measured once (H200 shared, heat2d 4096² × 100, `temporal=off`, window / gather speed): f16 1.00×, bf16 0.95×, i8 1.34×, f32 1.11×, f64 1.42×; heat3d f16 1024³ × 50 1.67×. Not tuned and not repeated: the tile is a first choice.

**Other limits:** one stage and one ring slot per launch (no `wave`); linear weights only; the f32 tolerance grows with the step count (TF32 truncation repeats per step: 1.8e-3 after 1 step, 1.2e-2 after 16).

**Open:** the native f16 / bf16 / i8 kernels are not tuned (A fragments are scalar shared-memory loads with no `ldmatrix`; heat2d f16 4096² × 100 takes 28.7 ms against 5.3 ms for the direct kernel); `mmaCostScale` is an estimate; a window form for fused chains and diamond tiles; a cost model to choose the form. Benchmark: `test/python/bench_gpu_tensor.py DTYPE N STEPS --form gather,window`.

### 6.5 `tileir` (NVIDIA through `tileiras`)

`lib/TileIR/StencilToTileIR.cpp` (direct and window forms), `GatherForm.cpp` (gather form). Pipeline: `stencil-to-tileir` → `cuda-tile-translate` → `tileiras` → cubin.

- **Window load:** a `strided_view` with tile 128 × 128 and step 126 per axis. The outer ring of each result is invalid; a masked store writes the inner 126 × 126.
- **Boundary rules:** `constant(0)`: out-of-grid view loads return zero. Other rules: boundary windows use a gather with a remapped index (clamp, mirror, mod); interior windows keep the view load.
- **Matmul, window form:** `out = Σ_a S_a · X · T_a` over row offsets a. `S_a` shifts rows; `T_a` is banded with row a's weights. A star stencil needs only two products: `out = L·X + X·R`. Cost: about 2(2R+1)·N multiply-adds per output (2N for a star). The constant matrices count in the footprint. Limits of the window form: 2D, one input, `constant(0)` or no rule, float, `relaxed`; the first rows and columns use the direct form.
- **Matmul, gather form** (`tileir-form=gather`): stencil2row, a data matrix `A[M, K]` of gathered neighbours × a weight matrix. Fewer multiply-adds (about K) but pointer gathers instead of view loads; no chain parameter and no border strips. Rank 1 to 3, any linear kernel; f16, bf16 and i8 on matrix units, f32 / f64 as scalar code. Measured once (H200, heat2d 4096² × 100, f16): gather is 2.16× the window form without a chain and about equal to the best chained window; f32 about equal, f64 0.59×. Bench: `bench_tileir_chain.py DTYPE N STEPS --form window|gather|both`.
- **Direct form:** used for everything else and with `exact`; any body.
- **`tileiras` 13.3 facts:** `mmaf` reaches tensor cores only for f16 / bf16 (HGMMA / HMMA), i8 (IMMA) and fp8 (QGMMA); f64 / f32 lower to scalar code. `allow_tma` on `load_view_tko` is ignored. Grep SASS with `HMMA|HGMMA|QGMMA|IMMA|DMMA`.
- **Time:** b_t = 1 only (shifted reads between steps inside a tile go through memory). Folding two steps costs 896 multiply-adds per output, which is slower than b_t = 1 on f16.
- **Limits:** only linear bodies with constant weights (the body is rebuilt from the S5a loops). Inline fusion is not available (section 8).

---

## 7. Cross-cutting: data types, boundary rules, fp-mode

| Topic | Rule |
|---|---|
| `fp-mode=exact` | Same bits as the plain loop. fp-contract off, no weight grouping, no matrix units for floats, no fold. |
| `fp-mode=relaxed` | May regroup weights, accumulate low types in `f32`, use float matrix units. |
| `fp-mode=lowp` | Also allows TF32 for f32. |
| `overflow=wrap` | Integer matrix units and wider accumulation allowed under `exact`. |
| `overflow=saturate` | Operations keep their written order; no widening, grouping or folding. |
| Boundary rules | `constant(c)`, `replicate`, `reflect`, `periodic`, per axis. Applied in boundary strips (CPU), by the tile fill (GPU), by the async copy (`constant(0)`), or by remapped gathers (TileIR). |
| Types | `i8`–`i64`, `fp8`, `f16`, `bf16`, `f32`, `f64`; the route depends on the type (5.6). |

---

## 8. Other options tried and dropped

Each entry: what we tried, why it did not work, and where it ended up. Sources are the dated reports in `stencil-dsl/report/`, `decision.md`, and the measurements recorded in earlier versions of this document. Entries marked *(reason not documented)* need a decision from the author before this section is final.

| Idea | What happened | Where it ended up |
|---|---|---|
| **Thread-block clusters with distributed shared memory (Hopper)**, `cluster=on` | Raw DSM access and cluster barriers measured ~8.7× slower than plain global memory; the full implementation also failed with `CUDA_ERROR_LAUNCH_FAILED` and was not root-caused. A Phase 0 test had already shown the negative result. | Off by default; three real bugs found on the way were fixed and kept. `report/hopper-clusters-2026-10-15.md` |
| **Bank-conflict padding for `gpu-tensor`** | Cut conflicts for 1-D, but made rank ≥ 2 about 10% slower (42.9 → 38.4 GStencils/s). | Reverted in full. `report/convstencil-gpu-tensor-bank-conflict-2026-10-24.md` |
| **Rolling the GPU diamond steps into one loop** (`diamond_roll_gpu=1`) | Bit-exact, but no speed gain at equal depth (46.2 vs 46.5 ms, heat2d f64 4096² × 100); deeper tiles gained ~4% and stayed behind overlapped tiling (41.8 ms). Shared-memory staging does not apply to rolled steps. | Kept behind the flag, off. `report/status_2026-11-17.md` |
| **Inline fusion on the TileIR route** | Wrong results (1000 cells wrong on a 100 × 72 chain with replicate / constant). The TileIR lowering does not resolve the `off-grid` / `via` placeholders inlining leaves. Not traced further. | The pass returns early on TileIR. |
| **Inline fusion on GPU by default** | Correct, but no measurement shows it is faster: an inlined step can only use diamond tiles or no time blocking, and GPU diamond tiles carry per-step barriers. | Off; `inline_fusion_gpu=1` enables it; no cost model. |
| **All diamond stages in one `gpu-tensor` kernel with block barriers** | A cross-thread read right after the barrier returned stale data for b_t ≥ 3 even with an explicit memory barrier; `racecheck` found nothing; cause never found. | One launch per stage instead (adds launch overhead). `decision.md`, 2026-10-20 |
| **Prefetch of the next tile (CPU)** | No measurable gain (0.99–1.04×): Zen 5's hardware prefetchers already cover it. | `prefetch=off` by default. |
| **Round-robin tile schedule (CPU)** | 0.75×: loses NUMA and cache locality. | Dropped. |
| **Unroll-and-jam of the scratch-buffer stages** | No gain. | Applied but not credited. |
| **Run-time choice of the thread count** | Brings back uneven load (tiles vs threads disagree). | Compile-time `threads=N`; a change recompiles. |
| **Exact per-index versioning of boundary tiles** (first version) | 2× slower when the grid is not a multiple of the tile. | Replaced by bound versioning with a remainder fix. |
| **Square diamond tiles on CPU** (first chooser) | The worst shape: 1.9× vs 3.1× for 16 × 4096 whole-row tiles (64 threads, vs overlapped). | Chooser now keeps the contiguous axis whole. |
| **Packing into shared memory on TileIR** | ~6 GPt/s against 46 GPt/s for direct loads. | TileIR forks before S5b. |
| **Doubling the TileIR tile (32 → 64) for power-of-2 shapes** | Wastes 75% of loads. | Overlapping 128-wide windows (~3% waste). |
| **Folding two steps into one MMA chain on TileIR** | 896 multiply-adds per output: compute-bound, slower than b_t = 1. | b_t = 1 only. |
| **GPU diamond with large tiles** | 128 × 128 measured 0.49×: fewer blocks outweigh thinner gaps. | `auto` takes overlapped on GPU. |
| **TileIR for f64 / f32 matrix work** | `tileiras` lowers f64 `mmaf` to scalar code (measured). | `gpu-tensor` (`nvgpu` DMMA) for f64; TileIR only for f16 / bf16 / i8. |
| **The earlier ConvStencil implementation (`stencil-dsl_old/`)** | On H200 the direct f64 kernel measured 301 GPoint-steps/s vs 152 for the earlier ConvStencil implementation. *(why the old implementation as a whole was replaced: reason not documented)* | Replaced by the generic pipeline. |

**Known problems, not dropped ideas** (from `report/status_2026-11-17.md` and `decision.md`; whether they are fixed has to be checked): rank-3 diamond on `gpu-tensor` fails with "no stencil nest in tile"; two lit failures noted on 2026-11-17 were not investigated.

---

## 9. Implementation status

**Implemented and tested** (full pytest 595 passed / 11 skipped, lit 29 / 29 on 2026-11-19; bit for bit under `exact`):
- S0–S8 as described, one pass list for all chips (`stencil-pipeline-auto`).
- Routes `cpu`, `gpu-cuda` (runs on the H200), `gpu-rocm` (compiles to ROCDL; not run), `gpu-tensor` (gather and window forms), `tileir` (direct, window, gather forms).
- All data types i8–i64, f16, bf16, f32, f64 (tf32 on `lowp`); all boundary rules, per axis.
- CPU thread count is user-given (`threads=N`); the tile count and the OpenMP `num_threads` use the same number (`test/lit/S7/parallelize.mlir`, `test_threads`).

**Not implemented / open:**
- Folding several time steps into one (S4). `stencil-fold-ring` is different: it folds the storage, not the steps.
- GPU diamond with fewer phases (diamond along one axis, overlapped along the others).
- Rules and passes for in-place (Gauss–Seidel) updates.
- Inline fusion with a `periodic` temporary; fold-ring, skew and ring endpoints on the GPU routes; a cost model for inline fusion on GPU.
- The TMA lowering for chips and shapes where `copy_async` still uses `cp.async`.
- `gpu-tensor`: window form for fused chains and diamond tiles; a cost model to choose the form; tuned native f16 / bf16 / i8 kernels; automatic choice of the native kinds.
- A recognizer for plain affine loops: only nests marked by S2 are optimized; other affine code passes through unchanged.
- AMD: this LLVM build lacks the AMDGPU target; `gather_to_lds` on gfx942 for 8-byte elements is unchecked.

**Performance.** The comparison with Pluto, YASK, Devito, xDSL, plain C + OpenMP and PyTorch is in the benchmark report and the presentation (`baselines/bench/`), because those numbers change whenever the baselines are re-tuned. As of this draft the baselines are being re-tuned at 32 threads; this document therefore carries no baseline table. Where it quotes a measurement it says what was measured and on which machine.

**Still behind (earlier 64-thread comparison):** heat3d 512³ against Pluto, heat1d and wave2d in cache, star radius 4 in cache against plain OpenMP, small grids. **Known runtime issue:** the kernel's OpenMP runtime pins the calling thread to one core, and PyTorch / NumPy work started afterwards on that thread inherits it.

---

## 10. Adding a new target

1. A chip-table entry (DLTI numbers).
2. An interface implementation, or reuse `#stencil.simt_gpu`.
3. Mapping attributes for its parallel levels.
4. Lowerings for the three target-neutral ops.
5. A serializer (`gpu::TargetAttrInterface`).

No shared pass changes.

---

## Appendix A — Example: 2D heat on four targets (design-time example)

**Program:** 2D 5-point heat diffusion (the CPU IR is in `cpuir_heat2d.md`). Grid 4096², T = 100, one step `u + 0.1·(n + s + w + e − 4c)`, boundary `constant(0)`.

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
| CPU | 128 × 256 | 2 | See `cpuir_heat2d.md`. |
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
// CPU: boundary tiles only, as in cpuir_heat2d.md
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
// CPU: stencil-forall-to-omp → one omp.parallel, two static loops (see cpuir_heat2d.md)
```

### S6 — inner code

```mlir
// NV: align 38x70 → 38x72 (row = 576 B); no bank padding (a warp reads 32 consecutive f64)
%S0 = memref.alloca() : memref<38x72xf64, #gpu.address_space<workgroup>>
// NV: unroll-jam 8 rows per thread → shared loads per output 5 → (8 + 2)/8 + 2 = 3.25
// AMD: 4 rows per thread → (4 + 2)/4 + 2 = 3.5; pitch 36 unchanged (288 B, 16-B aligned)
// vectorize: skipped on GPU (1 column per thread is already coalesced)
// async-pipeline: skipped (one window per block, no loop to pipeline)
// CPU: as in cpuir_heat2d.md S6
```

### S8 — lower

```mlir
// NV: copy_async → TMA; descriptor built on the host
%d = nvgpu.tma.create.descriptor %U box[%c38, %c70] : ...
nvgpu.tma.async.load %d[%x, %y], %mbar to %S0 : ...
// AMD: stencil.barrier → gpu.barrier → s_barrier; binary from #rocdl.target<chip = "gfx942">
// all: fp-contract off (exact)
```

### TIR — `stencil-to-tileir{form=window}` (instead of S5b–S8); the gather form is described after it

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

**TileIR gather form** (`tileir-form=gather`, same f16 heat2d): instead of loading a 128 × 128 window and multiplying by `L` and `R`, each output tile gathers its neighbours by pointer into a data matrix `A[M, K]` and multiplies it with a weight matrix on the matrix unit; there is no strided view, no chain parameter and no border strips. Real IR for both forms: `cudatileir_heat2d.md`.

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

---

## Appendix C — IR after every pass, per route

The IR of heat2d after each pass that changes it, generated from the real compiler (`mlir-print-ir-after-all`), is in:

- `cpuir_heat2d.md` — CPU route
- `tensorir_heat2d.md` — `gpu-tensor` route
- `cudatileir_heat2d.md` — `tileir` route
- `gpucudair_heat2d.md` — `gpu-cuda` route

Generated by `stencil-dsl/scripts/dump_pass_ir.py` (heat2d, 1024², 100 steps, `sm_90` / host chip). Each file has a stage index listing every pass and prints only the passes that change the IR. Appendix A and B above are hand-shortened examples.
