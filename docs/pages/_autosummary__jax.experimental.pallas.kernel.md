- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.kernel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.kernel.rst "Download source file")
-  .pdf

# jax.experimental.pallas.kernel

## Contents

- [`kernel()`](#jax.experimental.pallas.kernel)

# jax.experimental.pallas.kernel[\#](#jax-experimental-pallas-kernel "Link to this heading")

jax.experimental.pallas.kernel(*body=\<not-specified\>*, *out_type=()*, *\**, *mesh*, *scratch_types=()*, *compiler_params=None*, *interpret=False*, *cost_estimate=None*, *debug=False*, *name=None*, *metadata=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/helpers.py#L156-L278)[\#](#jax.experimental.pallas.kernel "Link to this definition")  
Entry point for creating a Pallas kernel.

This is a convenience wrapper around `mpmd_map` for executing a kernel over a mesh.

If `body` is provided, this function behaves as a decorator:

    def kernel_body(in_ref, out_ref):
      ...
    kernel = pl.kernel(kernel_body, out_type=...)

If `body` is omitted, this function behaves as a decorator factory and will return a decorator that can be used to annotate a kernel body:

    @pl.kernel(mesh=..., out_type=...)
    def kernel(in_ref, out_ref):
      ...

For MPMD kernels, you can pass parallel lists of bodies and meshes:

    my_kernel = pl.kernel(
        body=[vector_fn, scalar_fn],
        mesh=[v_mesh, s_mesh],
        out_type=...
    )

JAX `Ref` objects can be closed over by the kernel body or passed in as arguments. Any such `Ref` will be treated as if it is read-from and written-to and will be aliased in and out of the kernel.

    @pl.kernel(mesh=...)
    def kernel(in_ref, out_ref):
      ...
    x_ref = jax.new_ref(...)
    y_ref = jax.new_ref(...)
    kernel(x_ref, y_ref)  # Can now mutate x_ref and y_ref

Parameters:  
- **body** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable") *\|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\]* *\|* *NotSpecified*) – The body of the kernel. If provided, this function behaves as a decorator, and if omitted, this function behaves as a decorator factory. Can also be a sequence of callables to be paired with a sequence of meshes.

- **out_type** ([*object*](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)") *\|* *None*) – The type of the output. Should be a PyTree of `jax.ShapeDtypeStruct` or JAX types.

- **mesh** (*Mesh* *\|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[Mesh\]*) – The mesh to run the kernel on. Must be a sequence of meshes if `body` is a sequence of callables.

- **scratch_types** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[ScratchShape* *\|* *ScratchShapeTree* *\|* *None\]* *\|* [*Mapping*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *ScratchShape* *\|* *ScratchShapeTree\]*) – The shapes of the scratch arrays.

- **compiler_params** (*CompilerParams* *\|* *None*) – The compiler parameters to pass to the backend.

- **interpret** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – Whether to run the function in interpret mode.

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether or not to out helpful debugging information.

- **cost_estimate** (*CostEstimate* *\|* *None*) – The cost estimate of the function.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The (optional) name of the kernel.

- **metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – Optional dictionary of information about the kernel that will be serialized as JSON in the HLO. Can be used for debugging and analysis.

Returns:  
If `body` is provided, returns a function that runs the kernel. It should take any number of input operands and returns an output with the same PyTree structure as out_type. If `body` is omitted, returns a decorator that can be used to annotate a kernel body.

[](jax.experimental.pallas.core_map.html "previous page")

previous

jax.experimental.pallas.core_map

[](jax.experimental.pallas.pallas_call.html "next page")

next

jax.experimental.pallas.pallas_call

Contents

- [`kernel()`](#jax.experimental.pallas.kernel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
