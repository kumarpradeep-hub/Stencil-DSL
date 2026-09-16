This file will be for the high level overview of the Pass structure for  the Stencil DSL

Input:- It will input take the input as the Stencil IR then go through many MLIR passes before lower to the one of ht e target
1. stencil IR [https://arxiv.org/html/2404.02218v2]

OutPUT:- Output has following 4 paths bassed on the path selection the output IR will be generate
1. TileIR ( It will target the tileir of the NVIDIA GPU)
2. NVGPU  (It will target nvidia GPU)
3. AMDGPU (It will target AMD GPU)
4. CPU (It will target CPU)


The Passes are group in the following blocks
1. High level algebraic simplification and Fusion ( Possiblity of the fusion inside the function block horizontal fusion)
2. Domain specific Structureal analysis and opt passes (Opt based on the types of the Stencil type
4. Classic passes (LICM, Dead code, CSE, etc)
5. Target path selection (Which path to select and corresponding opt)
6. Tiling passes (Opt tile selection based on the H/w scrathpad, reg size and other relevant paramenter)
7. Memory transformation passes (Memref access)
8. Kernal Level Fusion ( Stencil Graph)
9. Stencil for Distributed GPU  ( for multi GPU Stencil computation)

