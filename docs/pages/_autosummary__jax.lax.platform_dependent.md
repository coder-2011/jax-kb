- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.platform_dependent

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.platform_dependent.rst "Download source file")
-  .pdf

# jax.lax.platform_dependent

## Contents

- [`platform_dependent()`](#jax.lax.platform_dependent)

# jax.lax.platform_dependent[\#](#jax-lax-platform-dependent "Link to this heading")

jax.lax.platform_dependent(*\*args*, *default=None*, *\*\*per_platform*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/conditionals.py#L1098-L1175)[\#](#jax.lax.platform_dependent "Link to this definition")  
Stages out platform-specific code.

In JAX the actual platform on which a computation is run is determined very late, e.g., based on where the data is located. When using AOT lowering or serialization, the computation may be compiled and executed on a different machine, or even on a platform that is not available at lowering time. This means that it is not safe to write platform-dependent code using Python conditionals, e.g., based on the current default JAX platform. Instead, one can use `platform_dependent`:

Usage:

    def cpu_code(*args): ...
    def tpu_code(*args): ...
    def other_platforms_code(*args): ...
    res = platform_dependent(*args, cpu=cpu_code, tpu=tpu_code,
                             default=other_platforms_code)

When the staged out code is executed on a CPU, this is equivalent to `cpu_code(*args)`, on a TPU is equivalent to `tpu_code(*args)` and on any other platform to `other_platforms_code(*args)`. Unlike a Python conditional, all alternatives are traced and staged out to Jaxpr. This is similar to, and is implemented in terms of, [`switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch"), from which it inherits the behavior under transformations.

Unlike a [`switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch") the choice of what gets executed is made earlier: in most cases during lowering when the lowering platform is known; in the rare case of multi-platform lowering and serialization, the StableHLO code will contain a conditional on the actual platform. This conditional is resolved just in time prior to compilation when the compilation platform is known. This means that the compiler actually never sees a conditional.

Parameters:  
- **\*args** (*Any*) – JAX arrays passed to each of the branches. May be PyTrees.

- **\*\*per_platform** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *\_T\]*) – branches to use for different platforms. The branches are JAX callables invoked with `*args`. The keywords are platform names, e.g., ‘cpu’, ‘tpu’, ‘cuda’, ‘rocm’.

- **default** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *\_T\]* *\|* *None*) – optional default branch to use for a platform not mentioned in `per_platform`. If there is no `default` there will be an error when the code is lowered for a platform not mentioned in `per_platform`.

Returns:  
The value `per_platform[execution_platform](*args)`.

[](jax.lax.pad.html "previous page")

previous

jax.lax.pad

[](jax.lax.polygamma.html "next page")

next

jax.lax.polygamma

Contents

- [`platform_dependent()`](#jax.lax.platform_dependent)

By The JAX authors

© Copyright 2024, The JAX Authors.\
