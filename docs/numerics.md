# Numerical implementation notes

## Verified high-Re path: LBM/LES

The verified production path is `solver.method=lbm`:

- D2Q9 BGK lattice-Boltzmann discretization
- Smagorinsky LES eddy viscosity
- second-order bulk accuracy in the low-Mach limit
- fixed timestep from initial CFL
- MPI slab decomposition in y plus OpenMP local loops
- momentum-exchange drag/lift on the solid mask

The current high-Re example uses `1081 x 271`, exactly 3x finer in each direction than the original `361 x 91` mesh, with `Re=2000`, `CFL=0.15`, `Cs=0.30`, and `6000` verified steps.

## Experimental FVM path

`solver.method=fvm` uses a cell-centered finite-volume projection method with:

- RK2 explicit advection-diffusion update
- second-order central finite-volume fluxes
- Smagorinsky LES viscosity
- pressure Poisson projection
- approximate pressure/viscous force integration on fluid-solid faces

This path is useful as a starting point for the requested FVM/FDM/FEM direction, but the current collocated immersed-mask pressure treatment is not yet robust enough for the verified high-Re production output. A production FVM version should use a MAC/staggered layout or Rhie-Chow coupling plus a better cut-cell/immersed-boundary pressure treatment.
