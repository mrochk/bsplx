import jax, jax.numpy as jnp
from functools import partial
from jaxtyping import jaxtyped, Array, Float
from beartype import beartype

def _safediv(num: Array, den: Array) -> Array:
    ok = (den != 0)
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)

def _clip_to_domain(x: Array, knots: Array) -> Array:
    x = jnp.where(x < knots[0], knots[0], x)
    return jnp.where(x > knots[-1], knots[-1], x)

def _basis0(x: Array, knots: Array) -> Array:
    lo, hi = knots[:-1], knots[1:]
    inside = (lo <= x) & (x < hi)
    at_end = (x >= knots[-1]) & (lo < hi) & (hi >= knots[-1])
    return (inside | at_end).astype(jnp.result_type(x, knots))

def _elevate(x: Array, knots: Array, b: Array, k: int) -> Array:
    '''one cox-de boor step'''
    K = knots.shape[0]
    t_i, t_ik = knots[:K - k - 1], knots[k:K - 1]
    t_i1, t_ik1 = knots[1:K - k], knots[k + 1:]
    left = _safediv(x - t_i, t_ik - t_i) * b[:-1]
    right = _safediv(t_ik1 - x, t_ik1 - t_i1) * b[1:]
    return left + right

@partial(jax.jit, static_argnames='d')
@jaxtyped(typechecker=beartype)
def repeat_knots(knots: Float[Array, 'k'], d: int) -> Float[Array, 'K']:
    left = jnp.repeat(knots[0], d)
    right = jnp.repeat(knots[-1], d)
    return jnp.concatenate([left, knots, right])

@partial(jax.jit, static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    x = _clip_to_domain(x, knots)
    b = _basis0(x, knots)
    for k in range(1, d + 1): b = _elevate(x, knots, b, k)
    return b

@partial(jax.jit, static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    dtype = jnp.result_type(x, knots)
    if d == 0: return jnp.zeros(knots.shape[0] - 1, dtype=dtype)

    outside = (x < knots[0]) | (x > knots[-1])
    xc = _clip_to_domain(x, knots)
    b = _basis0(xc, knots)
    for k in range(1, d): b = _elevate(xc, knots, b, k)

    K = knots.shape[0]
    la = knots[d:K - 1] - knots[:K - d - 1]
    rb = knots[d + 1:] - knots[1:K - d]
    dd = jnp.asarray(d, dtype)
    return jnp.where(outside, 0.0, dd * (_safediv(b[:-1], la) - _safediv(b[1:], rb)))

@partial(jax.jit, static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    return jax.vmap(partial(design_matrix_row, knots=knots, d=d))(x)

@partial(jax.jit, static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    return jax.vmap(partial(design_dmatrix_row, knots=knots, d=d))(x)
