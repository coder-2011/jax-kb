- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.export

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.export.rst "Download source file")
-  .pdf

# jax.export.export

## Contents

- [`export()`](#jax.export.export)

# jax.export.export[\#](#jax-export-export "Link to this heading")

jax.export.export(*fun_jit*, *\**, *platforms=None*, *disabled_checks=()*, *\_override_lowering_rules=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L563-L619)[\#](#jax.export.export "Link to this definition")  
Exports a JAX function for persistent serialization.

Parameters:  
- **fun_jit** ([*stages.Wrapped*](../jax.stages.html#jax.stages.Wrapped "jax.stages.Wrapped")) – the function to export. Should be the result of [`jax.jit()`](jax.jit.html#jax.jit "jax.jit").

- **platforms** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – Optional sequence containing a subset of ‘tpu’, ‘cpu’, ‘cuda’, ‘rocm’. If more than one platform is specified, then the exported code takes an argument specifying the platform. If None, then use the default JAX backend. The calling convention for multiple platforms is explained at [https://docs.jax.dev/en/latest/export/export.html#module-calling-convention](https://docs.jax.dev/en/latest/export/export.html#module-calling-convention).

- **\_override_lowering_rules** (*Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Any,* *Any\]\]* *\|* *None*) – an optional sequence of custom lowering rules for some JAX primitives. Each element of the sequence is a pair of a JAX primitive and a lowering function. Defining lowering rules is an advanced feature using JAX internal APIs, which are subject to change. Furthermore, the responsibility for the stability of the MLIR emitted through these custom lowering rules, rests with the user of these rules.

- **disabled_checks** (*Sequence\[*[*DisabledSafetyCheck*](../jax.export.html#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck")*\]*) – the safety checks to disable. See documentation for of [`jax.export.DisabledSafetyCheck`](../jax.export.html#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck").

Returns:  
a function that takes args and kwargs pytrees of [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct"), or values with `.shape` and `.dtype` attributes, and returns an [`Exported`](../jax.export.html#jax.export.Exported "jax.export.Exported").

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [Exported](../jax.export.html#jax.export.Exported "jax.export.Exported")\]

Usage:

    >>> from jax import export
    >>> exported: export.Exported = export.export(jnp.sin)(
    ...     np.arange(4, dtype=np.float32))
    >>>
    >>> # You can inspect the Exported object
    >>> exported.in_avals
    (ShapedArray(float32[4]),)
    >>> blob: bytearray = exported.serialize()
    >>>
    >>> # The serialized bytes are safe to use in a separate process
    >>> rehydrated: export.Exported = export.deserialize(blob)
    >>> rehydrated.fun_name
    'sin'
    >>> rehydrated.call(np.array([.1, .2, .3, .4], dtype=np.float32))
    Array([0.09983342, 0.19866933, 0.29552022, 0.38941833], dtype=float32)

[](../jax.export.html "previous page")

previous

`jax.export` module

[](jax.export.deserialize.html "next page")

next

jax.export.deserialize

Contents

- [`export()`](#jax.export.export)

By The JAX authors

© Copyright 2024, The JAX Authors.\
