- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.emit_pipeline

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.emit_pipeline.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.emit_pipeline

## Contents

- [`emit_pipeline()`](#jax.experimental.pallas.tpu.emit_pipeline)

# jax.experimental.pallas.tpu.emit_pipeline[\#](#jax-experimental-pallas-tpu-emit-pipeline "Link to this heading")

jax.experimental.pallas.tpu.emit_pipeline(*body*, *\**, *grid*, *in_specs=()*, *out_specs=()*, *tiling=None*, *core_axis=None*, *core_axis_name=None*, *dimension_semantics=None*, *trace_scopes=True*, *no_pipelining=False*, *\_explicit_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/pipeline.py#L1567-L1779)[\#](#jax.experimental.pallas.tpu.emit_pipeline "Link to this definition")  
Creates a function to emit a manual pallas pipeline.

This has the same semantics as pallas_call but is meant to be called inside pallas_call for nesting grids. This is useful when you need to have separate windowing strategies for communication and computation.

Parameters:  
- **body** – pallas kernel to set up pipeline for.

- **grid** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array")*,* *...\]*) – a pallas grid definition.

- **in_specs** – input pallas block specs

- **out_specs** – output pallas block specs

- **tiling** (*Tiling* *\|* *None*) – optional tiling to assume for the refs.

- **core_axis** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional int or tuple of int, indicates whether or not to partition the grid along the core axis.

- **core_axis_name** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional str or tuple of str, indicates whether or not to partition the grid along the core axis.

- **dimension_semantics** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*GridDimensionSemantics*](jax.experimental.pallas.tpu.GridDimensionSemantics.html#jax.experimental.pallas.tpu.GridDimensionSemantics "jax.experimental.pallas.tpu.GridDimensionSemantics")*,* *...\]* *\|* *None*) – optional tuple of GridDimensionSemantics (e.g. PARALLEL or ARBITRARY).

- **trace_scopes** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – optional bool, indicates whether to annotate each region in the pipeline using named_scope.

- **no_pipelining** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, turns off pipelining and all copies will be made synchronous. This is useful for debugging multiple-buffering related bugs.

- **\_explicit_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, the body will receive the iteration indices as its first argument. This parameter is meant for internal use only.

[](jax.experimental.pallas.tpu.BufferedRefBase.html "previous page")

previous

jax.experimental.pallas.tpu.BufferedRefBase

[](jax.experimental.pallas.tpu.emit_pipeline_with_allocations.html "next page")

next

jax.experimental.pallas.tpu.emit_pipeline_with_allocations

Contents

- [`emit_pipeline()`](#jax.experimental.pallas.tpu.emit_pipeline)

By The JAX authors

© Copyright 2024, The JAX Authors.\
