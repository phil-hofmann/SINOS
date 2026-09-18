# SINOS

Code accompanying the paper

**[Mitigating Numerical Stiffness in Least-Squares Formulations of Elliptic PDEs for Physics-Informed Neural Networks](https://arxiv.org/abs/2607.02726)**

This repository contains the numerical experiments for SINOS and D-SINOS, two negative-order Sobolev loss formulations for physics-informed neural networks.

The experiments compare standard mean-squared-error and L2 residual losses with SINOS and D-SINOS, as well as with established adaptive loss-balancing methods, including inverse-Dirichlet weighting, GradNorm, and neural-tangent-kernel-based balancing. They comprise spectral and computational benchmarks and PINN experiments for the Poisson equation, a convection–diffusion–reaction equation, and the stationary incompressible Navier–Stokes equations.

## Citation

If you use this code or build on the methods in this repository, please cite the accompanying paper.

The current arXiv preprint corresponds to the original manuscript, which has since been substantially revised. The updated preprint and final published version will be linked here once available.

```bibtex
@misc{Hofmann2026arXiv,
  title         = {Mitigating Numerical Stiffness in Least-Squares Formulations of Elliptic PDEs for Physics-Informed Neural Networks},
  author        = {Hofmann, Phil-Alexander and Hecht, Michael},
  year          = {2026},
  eprint        = {2607.02726},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA},
  doi           = {10.48550/arXiv.2607.02726},
  url           = {https://arxiv.org/abs/2607.02726}
}
```

## License

The project is licensed under the [MIT License](LICENSE.txt).