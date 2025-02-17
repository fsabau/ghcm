import jax
import jax.numpy as jnp
from .solver import FiniteDifferenceSolver
from .utils import interpolate_fn, add_time_fn
import equinox as eqx


class SigKernel(eqx.Module):
  static_kernel_kind: str = eqx.field(static=True, default_factory=lambda: "linear") 
  scales: jnp.ndarray = eqx.field(default_factory=lambda: jnp.array([1e0]))
  s0: float = 0.
  t0: float = 0.
  S: float = 1.
  T: float = 1.
  refinement_factor: float = 1.
  add_time: bool = False
  interpolation: str = eqx.field(static=True, default_factory=lambda: "linear")

  @eqx.filter_jit
  def kernel_matrix(self, X: jnp.ndarray, Y: jnp.ndarray, directions: jnp.ndarray = jnp.array([])) -> jnp.ndarray:

    # interpolate on new grid
    X = interpolate_fn(X, t_min=self.s0, t_max=self.S, refinement_factor=self.refinement_factor, kind=self.interpolation)
    Y = interpolate_fn(Y, t_min=self.t0, t_max=self.T, refinement_factor=self.refinement_factor, kind=self.interpolation)
    if directions.shape != (0,):
      interpolate_fn_vmap = jax.vmap(lambda Z: interpolate_fn(Z, t_min=self.s0, t_max=self.S,
                                                              refinement_factor=self.refinement_factor,
                                                              kind=self.interpolation), in_axes=0, out_axes=0)
      directions = interpolate_fn_vmap(directions)

    # add time channel (optionally)
    if self.add_time:
      X = add_time_fn(X, t_min=self.s0, t_max=self.S)
      Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)
      if directions.shape != (0,):
        add_time_fn_vmap = jax.vmap(lambda Z: add_time_fn(Z, t_min=self.s0, t_max=self.S), in_axes=0, out_axes=0)
        directions = add_time_fn_vmap(directions)

    # def body_fun(i, accum):
    #   s = self.scales[i]
    #   result = self.pde_solver.solve(s * X, Y, s * directions)
    #   return accum + result
    # initial_value = self.pde_solver.solve(self.scales[0] * X, Y, self.scales[0] * directions)
    # return jax.lax.fori_loop(1, len(self.scales), body_fun, initial_value) / len(self.scales)
    return jnp.mean(jax.vmap(lambda s: FiniteDifferenceSolver(static_kernel_kind=self.static_kernel_kind, scale=s).solve(X, Y, directions), in_axes=0, out_axes=0)(self.scales), axis=0)

