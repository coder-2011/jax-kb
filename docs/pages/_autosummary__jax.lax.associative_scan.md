- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.associative_scan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.associative_scan.rst "Download source file")
-  .pdf

# jax.lax.associative_scan

## Contents

- [`associative_scan()`](#jax.lax.associative_scan)

# jax.lax.associative_scan[\#](#jax-lax-associative-scan "Link to this heading")

jax.lax.associative_scan(*fn*, *elems*, *reverse=False*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py#L2748-L2889)[\#](#jax.lax.associative_scan "Link to this definition")  
Performs a scan with an associative binary operation, in parallel.

For an introduction to associative scans, see [\[BLE1990\]](#ble1990).

Parameters:  
- **fn** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) –

  A Python callable implementing an associative binary operation with signature `r`` ``=`` ``fn(a,`` ``b)`. Function fn must be associative, i.e., it must satisfy the equation `fn(a,`` ``fn(b,`` ``c))`` ``==`` ``fn(fn(a,`` ``b),`` ``c)`.

  The inputs and result are (possibly nested Python tree structures of) array(s) matching `elems`. Each array has a dimension in place of the `axis` dimension. fn should be applied elementwise over the `axis` dimension (for example, by using [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap") over the elementwise function.)

  The result `r` has the same shape (and structure) as the two inputs `a` and `b`.

- **elems** – A (possibly nested Python tree structure of) array(s), each with an `axis` dimension of size `num_elems`.

- **reverse** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A boolean stating if the scan should be reversed with respect to the `axis` dimension.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – an integer identifying the axis over which the scan should occur.

Returns:  
A (possibly nested Python tree structure of) array(s) of the same shape and structure as `elems`, in which the `k`’th element of `axis` is the result of recursively applying `fn` to combine the first `k` elements of `elems` along `axis`. For example, given `elems`` ``=`` ``[a,`` ``b,`` ``c,`` ``...]`, the result would be `[a,`` ``fn(a,`` ``b),`` ``fn(fn(a,`` ``b),`` ``c),`` ``...]`.

If `elems`` ``=`` ``[...,`` ``x,`` ``y,`` ``z]` and `reverse` is true, the result is `[...,`` ``f(f(z,`` ``y),`` ``x),`` ``f(z,`` ``y),`` ``z]`.

Example 1: partial sums of an array of numbers:

    >>> lax.associative_scan(jnp.add, jnp.arange(0, 4))
    Array([0, 1, 3, 6], dtype=int32)

Example 2: partial products of an array of matrices

    >>> mats = jax.random.uniform(jax.random.key(0), (4, 2, 2))
    >>> partial_prods = lax.associative_scan(jnp.matmul, mats)
    >>> partial_prods.shape
    (4, 2, 2)

Example 3: reversed partial sums of an array of numbers

    >>> lax.associative_scan(jnp.add, jnp.arange(0, 4), reverse=True)
    Array([6, 6, 5, 3], dtype=int32)

\[[BLE1990](#id1)\]
Blelloch, Guy E. 1990. “Prefix Sums and Their Applications.”, Technical Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon University.

[](jax.lax.zeta.html "previous page")

previous

jax.lax.zeta

[](jax.lax.cond.html "next page")

next

jax.lax.cond

Contents

- [`associative_scan()`](#jax.lax.associative_scan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
