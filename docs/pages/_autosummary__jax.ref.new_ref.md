- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.new_ref

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.new_ref.rst "Download source file")
-  .pdf

# jax.ref.new_ref

## Contents

- [`new_ref()`](#jax.ref.new_ref)

# jax.ref.new_ref[\#](#jax-ref-new-ref "Link to this heading")

jax.ref.new_ref(*init_val*, *\**, *memory_space=None*, *kind=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ref.py#L19-L40)[\#](#jax.ref.new_ref "Link to this definition")  
Create a mutable array reference with initial value `init_val`.

For more discussion, see the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
- **init_val** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – A [`jax.Array`](jax.Array.html#jax.Array "jax.Array") representing the initial state of the buffer.

- **memory_space** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – An optional memory space attribute for the Ref.

- **kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – An optional string indicating the mutation semantics under rematerialization. Currently only supports `'no_grad_no_remat'` or `None`.

Returns:  
A [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") containing a reference to a mutable buffer.

Return type:  
[*Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref")

[](jax.ref.get.html "previous page")

previous

jax.ref.get

[](jax.ref.set.html "next page")

next

jax.ref.set

Contents

- [`new_ref()`](#jax.ref.new_ref)

By The JAX authors

© Copyright 2024, The JAX Authors.\
