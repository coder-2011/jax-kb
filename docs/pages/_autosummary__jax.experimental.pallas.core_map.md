- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.core_map

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.core_map.rst "Download source file")
-  .pdf

# jax.experimental.pallas.core_map

## Contents

- [`core_map()`](#jax.experimental.pallas.core_map)

# jax.experimental.pallas.core_map[\#](#jax-experimental-pallas-core-map "Link to this heading")

jax.experimental.pallas.core_map(*mesh*, *\**, *compiler_params=None*, *interpret=False*, *debug=False*, *cost_estimate=None*, *name=None*, *metadata=None*, *scratch_shapes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py#L1494-L1573)[\#](#jax.experimental.pallas.core_map "Link to this definition")  
Runs a function on a mesh, mapping it over the devices in the mesh.

The function should be stateful in that it takes in no inputs and returns no outputs but can mutate closed-over Refs, for example.

Parameters:  
- **mesh** – The mesh to run the function on.

- **compiler_params** (*Any* *\|* *None*) – The compiler parameters to pass to the backend.

- **interpret** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to run the function in interpret mode.

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether or not to out helpful debugging information.

- **cost_estimate** (*CostEstimate* *\|* *None*) – The cost estimate of the function.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The (optional) name of the kernel.

- **metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – Optional dictionary of information about the kernel that will be serialized as JSON in the HLO. Can be used for debugging and analysis.

- **scratch_shapes** (*ScratchShapeTree*) – The scratch arrays for the kernel. Supports both sequence and dict format. The space will be core-local unless the memory space type is specified to be shared (e.g., VMEM_SHARED).

[](jax.experimental.pallas.Slice.html "previous page")

previous

jax.experimental.pallas.Slice

[](jax.experimental.pallas.kernel.html "next page")

next

jax.experimental.pallas.kernel

Contents

- [`core_map()`](#jax.experimental.pallas.core_map)

By The JAX authors

© Copyright 2024, The JAX Authors.\
