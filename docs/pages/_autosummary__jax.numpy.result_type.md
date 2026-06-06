- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.result_type

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.result_type.rst "Download source file")
-  .pdf

# jax.numpy.result_type

## Contents

- [`result_type()`](#jax.numpy.result_type)

# jax.numpy.result_type[\#](#jax-numpy-result-type "Link to this heading")

jax.numpy.result_type(*\*args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L472-L514)[\#](#jax.numpy.result_type "Link to this definition")  
Return the result of applying JAX promotion rules to the inputs.

JAX implementation of [`numpy.result_type()`](https://numpy.org/doc/stable/reference/generated/numpy.result_type.html#numpy.result_type "(in NumPy v2.4)").

JAX’s dtype promotion behavior is described in [Type promotion semantics](../type_promotion.html#type-promotion).

Parameters:  
**args** (*Any*) – one or more arrays or dtype-like objects.

Returns:  
A [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") instance representing the result of type promotion for the inputs.

Return type:  
DType

Examples

Inputs can be dtype specifiers:

    >>> jnp.result_type('int32', 'float32')
    dtype('float32')
    >>> jnp.result_type(np.uint16, np.dtype('int32'))
    dtype('int32')

Inputs may also be scalars or arrays:

    >>> jnp.result_type(1.0, jnp.bfloat16(2))
    dtype(bfloat16)
    >>> jnp.result_type(jnp.arange(4), jnp.zeros(4))
    dtype('float32')

Be aware that the result type will be canonicalized based on the state of the `jax_enable_x64` configuration flag, meaning that 64-bit types may be downcast to 32-bit:

    >>> jnp.result_type('float64')  
    dtype('float32')

For details on 64-bit values, refer to [Sharp bits - double precision](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision):

[](jax.numpy.resize.html "previous page")

previous

jax.numpy.resize

[](jax.numpy.right_shift.html "next page")

next

jax.numpy.right_shift

Contents

- [`result_type()`](#jax.numpy.result_type)

By The JAX authors

© Copyright 2024, The JAX Authors.\
