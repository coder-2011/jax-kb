- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.get

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.get.rst "Download source file")
-  .pdf

# jax.ref.get

## Contents

- [`get()`](#jax.ref.get)

# jax.ref.get[\#](#jax-ref-get "Link to this heading")

jax.ref.get(*ref*, *idx=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py#L119-L162)[\#](#jax.ref.get "Link to this definition")  
Read a value from an Ref.

This is equivalent to `ref[idx]` for a NumPy-style indexer `idx`. For more on mutable array refs, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
- **ref** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref") *\|* *TransformedRef*) – a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") object.

- **idx** (*Indexer* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Indexer,* *...\]* *\|* *None*) – a NumPy-style indexer

Returns:  
A [`jax.Array`](jax.Array.html#jax.Array "jax.Array") object (note, not a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) containing the indexed elements of the mutable reference.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| HijaxType

Examples

    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.get(ref, slice(1, 3))
    Array([1, 2], dtype=int32)

Equivalent operation via indexing syntax:

    >>> ref[1:3]
    Array([1, 2], dtype=int32)

Use `...` to extract the full buffer:

    >>> ref[...]
    Array([0, 1, 2, 3, 4], dtype=int32)

[](jax.ref.freeze.html "previous page")

previous

jax.ref.freeze

[](jax.ref.new_ref.html "next page")

next

jax.ref.new_ref

Contents

- [`get()`](#jax.ref.get)

By The JAX authors

© Copyright 2024, The JAX Authors.\
