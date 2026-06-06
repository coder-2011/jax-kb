- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.freeze

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.freeze.rst "Download source file")
-  .pdf

# jax.ref.freeze

## Contents

- [`freeze()`](#jax.ref.freeze)

# jax.ref.freeze[\#](#jax-ref-freeze "Link to this heading")

jax.ref.freeze(*ref*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L2900-L2925)[\#](#jax.ref.freeze "Link to this definition")  
Invalidate a given reference and return its final value.

For more information about mutable array references, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
**ref** ([*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) – A [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") object.

Returns:  
A [`jax.Array`](jax.Array.html#jax.Array "jax.Array") containing the contents of `ref`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[3] = 100
    >>> ref
    Ref([  0,   1,   2, 100,   4], dtype=int32)

    >>> jax.ref.freeze(ref)
    Array([  0,   1,   2, 100,   4], dtype=int32)

[](jax.ref.Ref.html "previous page")

previous

jax.ref.Ref

[](jax.ref.get.html "next page")

next

jax.ref.get

Contents

- [`freeze()`](#jax.ref.freeze)

By The JAX authors

© Copyright 2024, The JAX Authors.\
