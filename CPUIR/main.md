This md file descrive that how the passes IR will look like after each block of the pass for the CPU path
Example 
Stencil DSL Kernel


@stencil(
    neighborhood=star(radius=2, center=True),
    update=Jacobi,
    boundary=Dirichlet(0.0),
)
def heat_diffusion(u: Grid[f64], n: Neighborhood["u"]) -> f64:
    return n[-2] + n[-1] + n[0] + n[1] + n[2]


1. *stencilir* the input of the pipeline framework wiil like this
module {
  func.func @heat_diffusion(%u: !stencil.field<[128]xf64>, %out: !stencil.field<[128]xf64>) {
    %u_t = stencil.load %u : !stencil.field<[128]xf64> -> !stencil.temp<[128]xf64>
    %result = stencil.apply(%arg_u = %u_t : !stencil.temp<[128]xf64>) -> !stencil.temp<[128]xf64> attributes {stencil.boundary_condition = "dirichlet", stencil.boundary_value = 0.0 : f64} {
      %v0 = stencil.access %arg_u [-2] : (!stencil.temp<[128]xf64>) -> f64
      %v1 = stencil.access %arg_u [-1] : (!stencil.temp<[128]xf64>) -> f64
      %v2 = arith.addf %v0, %v1 : f64
      %v3 = stencil.access %arg_u [0] : (!stencil.temp<[128]xf64>) -> f64
      %v4 = arith.addf %v2, %v3 : f64
      %v5 = stencil.access %arg_u [1] : (!stencil.temp<[128]xf64>) -> f64
      %v6 = arith.addf %v4, %v5 : f64
      %v7 = stencil.access %arg_u [2] : (!stencil.temp<[128]xf64>) -> f64
      %v8 = arith.addf %v6, %v7 : f64
      stencil.return %v8 : f64
    }
    stencil.store %result to %out : !stencil.temp<[128]xf64> to !stencil.field<[128]xf64>
    return
  }
}

2. It will go through the 
