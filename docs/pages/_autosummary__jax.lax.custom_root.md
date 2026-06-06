- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.custom_root

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.custom_root.rst "Download source file")
-  .pdf

# jax.lax.custom_root

## Contents

- [`custom_root()`](#jax.lax.custom_root)

# jax.lax.custom_root[\#](#jax-lax-custom-root "Link to this heading")

jax.lax.custom_root(*f*, *initial_guess*, *solve*, *tangent_solve*, *has_aux=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/solves.py#L49-L130)[\#](#jax.lax.custom_root "Link to this definition")  
Differentiably solve for the roots of a function.

This is a low-level routine, mostly intended for internal use in JAX. Gradients of custom_root() are defined with respect to closed-over variables from the provided function `f` via the implicit function theorem: [https://en.wikipedia.org/wiki/Implicit_function_theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")) – function for which to find a root. Should accept a single argument, return a tree of arrays with the same structure as its input.

- **initial_guess** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – initial guess for a zero of f.

- **solve** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]*) –

  function to solve for the roots of f. Should take two positional arguments, f and initial_guess, and return a solution with the same structure as initial_guess such that func(solution) = 0. In other words, the following is assumed to be true (but not checked):

      solution = solve(f, initial_guess)
      error = f(solution)
      assert all(error == 0)

- **tangent_solve** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*,* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]*) –

  function to solve the tangent system. Should take two positional arguments, a linear function `g` (the function `f` linearized at its root) and a tree of array(s) `y` with the same structure as initial_guess, and return a solution `x` such that `g(x)=y`:

  - For scalar `y`, use `lambda`` ``g,`` ``y:`` ``y`` ``/`` ``g(1.0)`.

  - For vector `y`, you could use a linear solve with the Jacobian, if dimensionality of `y` is not too large: `lambda`` ``g,`` ``y:`` ``np.linalg.solve(jacobian(g)(y),`` ``y)`.

- **has_aux** – bool indicating whether the `solve` function returns auxiliary data like solver diagnostics as a second argument.

Returns:  
The result of calling solve(f, initial_guess) with gradients defined via implicit differentiation assuming `f(solve(f,`` ``initial_guess))`` ``==`` ``0`.

[](jax.lax.custom_linear_solve.html "previous page")

previous

jax.lax.custom_linear_solve

[](jax.lax.all_gather.html "next page")

next

jax.lax.all_gather

Contents

- [`custom_root()`](#jax.lax.custom_root)

By The JAX authors

© Copyright 2024, The JAX Authors.\
