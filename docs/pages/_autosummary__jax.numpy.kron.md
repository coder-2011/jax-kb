- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.kron

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.kron.rst "Download source file")
-  .pdf

# jax.numpy.kron

## Contents

- [`kron()`](#jax.numpy.kron)

# jax.numpy.kron[\#](#jax-numpy-kron "Link to this heading")

jax.numpy.kron(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8042-L8086)[\#](#jax.numpy.kron "Link to this definition")  
Compute the Kronecker product of two input arrays.

JAX implementation of [`numpy.kron()`](https://numpy.org/doc/stable/reference/generated/numpy.kron.html#numpy.kron "(in NumPy v2.4)").

The Kronecker product is an operation on two matrices of arbitrary size that produces a block matrix. Each element of the first matrix `a` is multiplied by the entire second matrix `b`. If `a` has shape (m, n) and `b` has shape (p, q), the resulting matrix will have shape (m \* p, n \* q).

Parameters:  
- **a** (*ArrayLike*) – first input array with any shape.

- **b** (*ArrayLike*) – second input array with any shape.

Returns:  
A new array representing the Kronecker product of the inputs `a` and `b`. The shape of the output is the element-wise product of the input shapes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.outer()`](jax.numpy.outer.html#jax.numpy.outer "jax.numpy.outer"): compute the outer product of two arrays.

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> b = jnp.array([[5, 6],
    ...                [7, 8]])
    >>> jnp.kron(a, b)
    Array([[ 5,  6, 10, 12],
           [ 7,  8, 14, 16],
           [15, 18, 20, 24],
           [21, 24, 28, 32]], dtype=int32)

[](jax.numpy.kaiser.html "previous page")

previous

jax.numpy.kaiser

[](jax.numpy.lcm.html "next page")

next

jax.numpy.lcm

Contents

- [`kron()`](#jax.numpy.kron)

By The JAX authors

© Copyright 2024, The JAX Authors.\
