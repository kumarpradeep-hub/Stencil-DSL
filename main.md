# Pass Framework Architecture for the Stencil DSL

High-level overview of the pass structure for a general-purpose Stencil DSL compiler stack,
capable of handling arbitrary stencil computations and lowering to multiple hardware targets
through a shared, reusable pipeline.

## Input

The entry point for the pipeline is the **Stencil IR** (value semantics: `stencil.field`,
`stencil.temp`, `stencil.apply`, `stencil.access`, `stencil.return`, `stencil.load`,
`stencil.store`), extending the dialect introduced in Gysi et al. [OEC, 2021] and reused/extended
in the shared stencil compilation-stack work [arXiv:2404.02218].

Hardware/target information (target kind, HW spec parameters, decomposition topology) is attached
as **module/operation attributes** from the very start, alongside the Stencil IR — not discovered
later. Passes downstream read these attributes as hints; they do not need a dedicated pass to
"find out" the target.

## Output

Based on target-path dispatch (see Block 4), the pipeline emits one of:

1. **TileIR** — NVIDIA's tile-level compiler stack (mid-level, tile-abstraction target)
2. **nvgpu** — NVIDIA GPUs (low-level, affine-loop + tensor-core target)
3. **amdgpu** — AMD GPUs
4. **llvmir** — CPUs
5. **Custom hardware accelerators** (future extension point)

Note: TileIR sits at a different abstraction level than nvgpu/amdgpu/llvmir (closer to
Triton/Pallas than to raw affine+memref). It is a **branch point in the pipeline**, not merely a
different backend flag applied uniformly at the end — see Block 4.

## Architectural components (not passes, but required infrastructure)

- **Compiler driver / target dispatch layer**: analogous to PolyBlocks' C4. Owns pass-pipeline
  specification per target, target-info/attribute propagation, tensor↔memref (or tensor↔tile) ABI
  marshalling at the boundary, and JIT/AOT execution (kernel launch, synchronization, linking the
  right runtime — CUDA/ROCm/OpenMP/custom).
- **Cost model / fusion-and-tiling decision machinery**: shared utilities used by Blocks 2 and 5 to
  decide whether a candidate fusion or tile choice is legal and profitable — dependence/conflict
  checks, redundant-computation tolerance (e.g. 0–10%), buffer-fits-on-chip checks, parallelism
  and vectorizability preservation checks. This is infrastructure the passes call into, not a pass
  itself.
- **Canonicalization / CSE / DCE / LICM**: standard MLIR cleanup passes interleaved between every
  block below. Fully reusable across all targets.

## Pass Blocks

Each block below is tagged with its **reuse level**:
- **[R]** Reused as-is across all targets/frontends
- **[C]** Mostly reused, customized per target
- **[T]** Target-specific, not reused beyond standard cleanup passes

---

### Block 1 — High-level Fusion (unified: local + cross-kernel) **[R]**

Operates on Stencil IR in value semantics, using `stencil.access` relative-offset information
(not general affine analysis — the stencil dialect's simpler offset model makes this tractable
at this level, following OEC).

Merges what was previously split into two disconnected steps:
- **Local (horizontal) fusion**: sequential `stencil.apply` ops in the same block operating on the
  same data are grouped into one, when profitable.
- **Cross-kernel fusion**: fusion across function/kernel boundaries, using the SSA def-use /
  call graph of stencil-kernel invocations (not an AST — by the time kernels are expressed as
  stencil-dialect operations, the representation is already SSA+regions).

Both use the same underlying slicing/validity machinery (see cost-model component above):
dependence-conflict checks, redundant-computation bound, and whether the fusion eliminates a
buffer or just relocates it.

This block does **not** run once and hand off — it re-engages with Block 5 (see Two-Phase
Tile↔Fuse below). Running fusion once, in isolation, before tiling ever happens, throws away
information tiling needs to decide what to fuse into.

---

### Block 2 — Domain-Specific Structural Analysis **[R]**

Analyses that stay valid regardless of target:

- **Halo/ghost-region derivation**: scan `stencil.access` offsets across all `stencil.apply` ops to
  compute minimal halo shape and size, as in the shared-stack paper's dmp pass.
- **Boundary-condition modeling**: a **first-class representation**, not manually-encoded
  conditionals over memory accesses. (The reference stencil-dialect work explicitly flagged
  boundary conditions as unsupported at a high level — this is our opportunity to solve it
  properly rather than inherit the limitation.)
- **Decomposition-strategy selection**: choose the domain-decomposition strategy (1D/2D/3D
  slicing, or more exotic overlap patterns) and rank topology. This must happen **here**, not
  deferred to the distributed-system block at the end, because rank-local domain size directly
  determines what "a tile" means for Blocks 5 and 6.
- **Bounds-mode decision**: compile-time-known bounds vs. dynamic bounds, decided explicitly and
  recorded as an attribute, since it materially affects which downstream optimizations are legal
  (constant-folding of address computations, register pressure reduction) — a known tradeoff
  flagged in the reference stencil-dialect work.

---

### Block 3 — Target Path Dispatch **[C]**

Selects the lowering path and applies the corresponding pipeline configuration, using the
target attributes established at input.

- For **nvgpu / amdgpu / llvmir**: continue through the full affine-loop pipeline (Blocks 5–8
  below).
- For **TileIR**: **fork here**. TileIR wants tile-level ops emitted directly; forcing it through
  explicit affine-loop tiling (Block 5) and then re-deriving tile structure fights the target
  abstraction. The TileIR path should branch before Block 5 into a tile-native lowering that
  reuses Blocks 1–2's fusion/domain decisions but not the explicit loop-tiling machinery.
- For **custom accelerators**: a defined extension point — must specify at minimum a HW-spec
  descriptor (scratchpad size, parallelism granularity, supported matrix/vector units) that
  Blocks 5–7 can consume.

---

### Block 4 — Tile ↔ Fuse (two-phase, iterated) **[C]**

Not a single one-shot tiling pass. Follows the PolyBlocks-style two-phase interaction, since
tiling-then-fusing and fusing-then-tiling both fail in important cases:

1. **Phase A**: Tile the "anchor" nests first (the stencil computations that are fusion
   destinations), considering HW spec — scratchpad/shared-memory size, cache size, register
   count. Generate their on-chip scratchpad buffers as part of this phase (not deferred to
   Block 5).
2. **Phase B**: Re-invoke Block 1's fusion machinery to pull profitable producers/consumers into
   the now-tiled anchors, using the cost-model component to validate.
3. **Phase C**: Tile whatever nests remain untiled, for locality/parallelism.

Very few components in this block are reusable across targets — most tiling decisions are
HW-parameter-driven — but the two-phase *algorithm* structure is reusable; only the cost
model's HW inputs change per target.

---

### Block 5 — Memory Transformation **[C]**

Lowers to the `memref` dialect: materializes concrete `memref.alloc`s for the on-chip scratchpad
buffers implied by Block 4's tiling, and generates the copy-in/copy-out data-movement loops at
tile boundaries. If Block 4 already tiled the abstract stencil iteration domain in value
semantics, this block is where that gets made concrete in memory.

---

### Block 6 — Vectorization / Matrix-Unit Mapping **[C]**

Given its own block rather than folded silently into tiling:

- Affine/memref access-pattern (contiguity) analysis to determine vectorizability.
- Mapping to SIMD/vector types.
- Mapping eligible stencil computations onto matrix/tensor-core units where hardware supports it
  (im2col-like restructuring, analogous to how convolutions are mapped to matrix units in
  PolyBlocks) — a genuine research opportunity for stencils specifically, not yet demonstrated
  broadly in the reference literature.

---

### Block 7 — Kernel / Launch Fusion (late-stage) **[C]**

Distinct from Block 1's early producer/consumer fusion. Operates on already-tiled compute
regions and merges them into fewer kernel launches / parallel regions, specifically to address
known regressions:

- Synchronous per-loop kernel launches from `scf.parallel`→`gpu` lowering (observed to hurt
  tracer-advection-style workloads with many stencil regions in the reference evaluation).
- Kernel-launch synchronization overhead dominating smaller GPU kernels (observed in the
  reference evaluation's 3D kernel profiling).

This block should be designed in from the start rather than left as a known limitation to fix
later.

---

### Block 8 — Stencil for Distributed Systems (message-passing codegen) **[C]**

Given decomposition strategy already fixed in Block 2, this block focuses on:

- Lowering domain-decomposition/halo-exchange declarations (dmp-style) to concrete message-passing
  calls (mpi-style), following the shared-stack paper's dmp→mpi lowering pattern.
- **Communication/computation overlap**, designed in rather than deferred — the reference dmp
  design's one-exchange-per-halo limitation and lack of overlap should not be repeated here.
- Multi-halo batching (combining multiple simultaneous halo exchanges) as a first-class capability.
- Target-appropriate backend: MPI for CPU clusters, NCCL/RCCL-style collectives for multi-GPU,
  or custom interconnect primitives for accelerators, selected via the same target dispatch
  attributes from Block 3.

---

## Implementation Status (`stencil-dsl/`)

The design above is the target architecture; this table tracks what actually
exists today in `stencil-dsl/lib/StencilDSL/` and `stencil-dsl/tools/stencil-opt`,
one sentence per pass. "Implemented" means the pass does real, verified work
(exercised by `test/run_heat_diffusion_from_python.py` and the lit suite
under `stencil-dsl/test/`); "Stub" means it's registered and wired into the
pipeline but its `runOnOperation()` is still a documented no-op placeholder.

| Pass (CLI flag) | Block | Status | One-sentence description |
|---|---|---|---|
| `stencil-local-fusion` | 1 | Implemented | Vertically fuses a producer `stencil.apply` into a single-operand consumer by inlining the producer's body with its access offsets composed against the consumer's own access offset, eliminating the intermediate `stencil.temp`. |
| `stencil-cross-kernel-fusion` | 1 | Implemented | Inlines `func.call`s to private, stencil-only kernel functions, forwards/eliminates the `stencil.store`-then-`stencil.load` round trip through the scratch field between them, then re-runs `stencil-local-fusion` so the now-merged kernels' `stencil.apply` ops can fuse across the former kernel boundary. |
| `stencil-infer-halo` | 2 | Implemented | Scans every `stencil.apply`'s `stencil.access` offsets and records the per-axis maximum `\|offset\|` as a `stencil.halo` attribute for Blocks 4/5/8 to consume. |
| `stencil-model-boundary-conditions` | 2 | Implemented (single policy) | Attaches a default `stencil.boundary_condition = "copy"` attribute to any `stencil.apply` that doesn't already declare one (the Python frontend sets `"copy"` or `"dirichlet"` + a value directly; per-face/multi-policy boundary regions are not modeled yet). |
| `stencil-select-decomposition-strategy` | 2 | Implemented (single-rank only) | Records `stencil.decomposition_strategy = "single_rank"` on the module, since this project's only working backend today is single-node CPU; real multi-rank strategy selection is future work alongside Block 8. |
| `stencil-decide-bounds-mode` | 2 | Implemented | Records `stencil.bounds_mode = "static"` or `"dynamic"` on the module by checking whether any `stencil.field`/`stencil.temp` type reachable from a function signature has an empty (dynamic) bounds list. |
| Target dispatch (`stencil-pipeline-{tileir,nvgpu,amdgpu,llvm}`) | 3 | Implemented | Four named `PassPipelineRegistration`s in `Pipelines/TargetDispatchPipelines.cpp` chain Blocks 1–2 then fork: `tileir` skips straight to `stencil-to-tileir`, while `nvgpu`/`amdgpu`/`llvm` continue through Blocks 4–7 before their respective backend conversion pass. |
| `stencil-tile-fuse` | 4 | Stub (deliberate no-op for CPU) | Registered and wired into every affine-loop-path pipeline, but currently a documented no-op: the CPU/LLVM path iterates the whole domain in one `scf.for` nest per `stencil.apply`, so there is nothing to tile yet without a `--tile-size` option that doesn't exist. |
| `stencil-to-memref` | 5 | Implemented | Lowers `stencil.field`/`stencil.temp` to `memref`, turning each `stencil.apply` into an interior `scf.for` nest (bounded by the halo) of `memref.load`/arith/`memref.store`, writing in place into a directly-storing `stencil.store`'s destination when possible (else a fresh `memref.alloc`), and filling the halo border via `"copy"` or `"dirichlet"` boundary fills. |
| `stencil-vectorize` | 6 | Stub | Registered but not yet implemented; intended to do contiguity analysis and map eligible accesses to SIMD vector types. |
| `stencil-map-to-matrix-units` | 6 | Stub | Registered but not yet implemented; intended to map eligible stencil computations onto matrix/tensor-core units. |
| `stencil-kernel-launch-fusion` | 7 | Stub (N/A for CPU) | Registered but a no-op; this block is inherently GPU-specific (merging separate `gpu.func` launches), and the CPU/LLVM path never produces `gpu.func`s, so there is nothing for it to do on this backend yet. |
| `stencil-to-llvm` | 3 (llvm output) | Implemented | Runs the standard MLIR lowering pipeline (`convert-scf-to-cf`, `convert-arith-to-llvm`, `finalize-memref-to-llvm`, `convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}`, `convert-cf-to-llvm`, `reconcile-unrealized-casts`) to produce `llvm` dialect with a bare-element-pointer ABI, ready for `mlir-translate --mlir-to-llvmir` and calling from `ctypes`. |
| `stencil-to-memref`'s dirichlet path | 5 | Implemented | Specifically: fills the halo border with a constant materialized from the `stencil.boundary_value` attribute when `stencil.boundary_condition = "dirichlet"`, instead of copying the input through unchanged. |
| `stencil-to-tileir` | 3 (tileir output) | Stub | Registered (guarded by `STENCILDSL_ENABLE_TILEIR`) but not yet implemented; intended to emit `cuda_tile` dialect ops directly from `stencil.apply` nests. |
| `stencil-to-nvgpu` | output | Stub | Registered (guarded by `STENCILDSL_ENABLE_NVGPU`) but not yet implemented; intended to do thread-block/grid mapping and tensor-core lowering. |
| `stencil-to-amdgpu` | output | Stub | Registered (guarded by `STENCILDSL_ENABLE_AMDGPU`, off by default since this tree's LLVM wasn't built with the AMDGPU codegen target) but not yet implemented. |
| `stencil-to-distributed` | 8 | Stub | Registered (guarded by `STENCILDSL_ENABLE_DISTRIBUTED`) but not yet implemented; intended to lower decomposition-strategy/halo-exchange declarations to MPI/NCCL calls. |

Known correctness limitation surfaced by the CPU correctness harness
(`stencil-dsl/test/python/correctness/`): fusing across a kernel/apply
boundary correctly widens the halo and interior formula, but the fused
kernel's *boundary* region is not yet equivalent to independently
re-applying each original kernel's own boundary rule in sequence — see
`star_sum_fused_reference`'s docstring. Fixing this needs Block 2's
boundary-condition modeling to actually split interior/boundary into
separate regions (as designed above), not just tag an attribute, so they
can be fused independently.

---

## Summary reuse table

| Block | Name | Reuse |
|---|---|---|
| 1 | High-level Fusion (local + cross-kernel) | [R] |
| 2 | Domain-Specific Structural Analysis | [R] |
| 3 | Target Path Dispatch | [C] |
| 4 | Tile ↔ Fuse (two-phase) | [C] |
| 5 | Memory Transformation | [C] |
| 6 | Vectorization / Matrix-Unit Mapping | [C] |
| 7 | Kernel / Launch Fusion | [C] |
| 8 | Distributed Systems (message-passing codegen) | [C] |

Standard cleanup passes (canonicalization, CSE, DCE, LICM) run between every block, fully
reused **[R]** regardless of target.
