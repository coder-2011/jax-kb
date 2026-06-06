- [](../index.html)
- [API Reference](../jax.html)
- jax.linearize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.linearize.rst "Download source file")
-  .pdf

# jax.linearize

## Contents

- [`linearize()`](#jax.linearize)

# jax.linearize[\#](#jax-linearize "Link to this heading")

jax.linearize(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*, *\*primals*, *has_aux: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L1425-L1502)[\#](#jax.linearize "Link to this definition")\
jax.linearize(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*, *\*primals*, *has_aux: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"), Any\]  
Produces a linear approximation to `fun` using [`jvp()`](jax.jvp.html#jax.jvp "jax.jvp") and partial eval.

Parameters:  
- **fun** – Function to be differentiated. Its arguments should be arrays, scalars, or standard Python containers of arrays or scalars. It should return an array, scalar, or standard python container of arrays or scalars.

- **primals** – The primal values at which the Jacobian of `fun` should be evaluated. Should be a tuple of arrays, scalar, or standard Python container thereof. The length of the tuple is equal to the number of positional parameters of `fun`.

- **has_aux** – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be linearized, and the second is auxiliary data. Default False.

Returns:  
If `has_aux` is `False`, returns a pair where the first element is the value of `f(*primals)` and the second element is a function that evaluates the (forward-mode) Jacobian-vector product of `fun` evaluated at `primals` without re-doing the linearization work. If `has_aux` is `True`, returns a `(primals_out,`` ``lin_fn,`` ``aux)` tuple where `aux` is the auxiliary data returned by `fun`.

In terms of values computed, [`linearize()`](#jax.linearize "jax.linearize") behaves much like a curried [`jvp()`](jax.jvp.html#jax.jvp "jax.jvp"), where these two code blocks compute the same values:

    y, out_tangent = jax.jvp(f, (x,), (in_tangent,))

    y, f_jvp = jax.linearize(f, x)
    out_tangent = f_jvp(in_tangent)

However, the difference is that [`linearize()`](#jax.linearize "jax.linearize") uses partial evaluation so that the function `f` is not re-linearized on calls to `f_jvp`. In general that means the memory usage scales with the size of the computation, much like in reverse-mode. (Indeed, [`linearize()`](#jax.linearize "jax.linearize") has a similar signature to [`vjp()`](jax.vjp.html#jax.vjp "jax.vjp")!)

This function is mainly useful if you want to apply `f_jvp` multiple times, i.e. to evaluate a pushforward for many different input tangent vectors at the same linearization point. Moreover if all the input tangent vectors are known at once, it can be more efficient to vectorize using [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap"), as in:

    pushfwd = partial(jvp, f, (x,))
    y, out_tangents = vmap(pushfwd, out_axes=(None, 0))((in_tangents,))

By using [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") and [`jvp()`](jax.jvp.html#jax.jvp "jax.jvp") together like this we avoid the stored-linearization memory cost that scales with the depth of the computation, which is incurred by both [`linearize()`](#jax.linearize "jax.linearize") and [`vjp()`](jax.vjp.html#jax.vjp "jax.vjp").

Here’s a more complete example of using [`linearize()`](#jax.linearize "jax.linearize"):

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x): return 3. * jnp.sin(x) + jnp.cos(x / 2.)
    ...
    >>> jax.jvp(f, (2.,), (3.,))
    (Array(3.2681944, dtype=float32, weak_type=True), Array(-5.007528, dtype=float32, weak_type=True))
    >>> y, f_jvp = jax.linearize(f, 2.)
    >>> print(y)
    3.2681944
    >>> print(f_jvp(3.))
    -5.007528
    >>> print(f_jvp(4.))
    -6.676704

[](jax.jvp.html "previous page")

previous

jax.jvp

[](jax.linear_transpose.html "next page")

next

jax.linear_transpose

Contents

- [`linearize()`](#jax.linearize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
