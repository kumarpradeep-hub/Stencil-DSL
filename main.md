This file will be for the high level overview of the Pass structure for  the Stencil DSL

Input:- It will input take the input as the Stencil IR then go through many MLIR passes before lower to the one of ht e target
1. stencil IR [https://arxiv.org/html/2404.02218v2]

OutPUT:- Output has following 4 paths bassed on the path selection the output IR will be generate
1. TileIR
2. NVGPU
3. AMDGPU
4. CPU


The Passes are group in the following blocks
1. High level algebraic simplification and Fusion
2. Domain specific Structureal analysis and opt passes
3. Memory transformation passes
4. Classic passes (LICM, Dead code, CSE, etc)
5. Target selection
6. Tiling passes (Opt tile selection based on the H/w scrathpad, reg size and other relevant paramenter)
7. Kernal Level Fusion ( Stencil Graph)
8. Stencil for Distributed GPU  ( for multi GPU Stencil computation)

