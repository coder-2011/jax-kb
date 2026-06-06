- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.custom_linear_solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.custom_linear_solve.rst "Download source file")
-  .pdf

# jax.lax.custom_linear_solve

## Contents

- [`custom_linear_solve()`](#jax.lax.custom_linear_solve)

# jax.lax.custom_linear_solve[\#](#jax-lax-custom-linear-solve "Link to this heading")

jax.lax.custom_linear_solve(*matvec*, *b*, *solve*, *transpose_solve=None*, *symmetric=False*, *has_aux=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/solves.py#L209-L323)[\#](#jax.lax.custom_linear_solve "Link to this definition")  
Perform a matrix-free linear solve with implicitly defined gradients.

This function allows for overriding or defining gradients for a linear solve directly via implicit differentiation at the solution, rather than by differentiating *through* the solve operation. This can sometimes be much faster or more numerically stable, or differentiating through the solve operation may not even be implemented (e.g., if `solve` uses `lax.while_loop`).

Required invariant:

    x = solve(matvec, b)  # solve the linear equation
    assert matvec(x) == b  # not checked

Parameters:  
- **matvec** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")) – linear function to invert. Must be differentiable.

- **b** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – constant right handle side of the equation. May be any nested structure of arrays.

- **solve** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]*) – higher level function that solves for solution to the linear equation, i.e., `solve(matvec,`` ``x)`` ``==`` ``x` for all `x` of the same form as `b`. This function need not be differentiable.

- **transpose_solve** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* *None*) – higher level function for solving the transpose linear equation, i.e., `transpose_solve(vecmat,`` ``x)`` ``==`` ``x`, where `vecmat` is the transpose of the linear map `matvec` (computed automatically with autodiff). Required for backwards mode automatic differentiation, unless `symmetric=True`, in which case `solve` provides the default value.

- **symmetric** – bool indicating if it is safe to assume the linear map corresponds to a symmetric matrix, i.e., `matvec`` ``==`` ``vecmat`.

- **has_aux** – bool indicating whether the `solve` and `transpose_solve` functions return auxiliary data like solver diagnostics as a second argument.

Returns:  
Result of `solve(matvec,`` ``b)`, with gradients defined assuming that the  
solution `x` satisfies the linear equation `matvec(x)`` ``==`` ``b`.

[](jax.lax.stop_gradient.html "previous page")

previous

jax.lax.stop_gradient

[](jax.lax.custom_root.html "next page")

next

jax.lax.custom_root

Contents

- [`custom_linear_solve()`](#jax.lax.custom_linear_solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
