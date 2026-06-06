- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.CompilerParams

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.CompilerParams.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.CompilerParams

## Contents

- [`CompilerParams`](#jax.experimental.pallas.triton.CompilerParams)
  - [`CompilerParams.num_warps`](#jax.experimental.pallas.triton.CompilerParams.num_warps)
  - [`CompilerParams.num_stages`](#jax.experimental.pallas.triton.CompilerParams.num_stages)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.triton.CompilerParams.__init__)

# jax.experimental.pallas.triton.CompilerParams[\#](#jax-experimental-pallas-triton-compilerparams "Link to this heading")

*class* jax.experimental.pallas.triton.CompilerParams(*num_warps=None*, *num_stages=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/core.py#L21-L33)[\#](#jax.experimental.pallas.triton.CompilerParams "Link to this definition")  
Compiler parameters for Triton.

Parameters:  
- **num_warps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **num_stages** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

num_warps[\#](#jax.experimental.pallas.triton.CompilerParams.num_warps "Link to this definition")  
The number of warps to use for the kernel. Each warp consists of 32 threads.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

num_stages[\#](#jax.experimental.pallas.triton.CompilerParams.num_stages "Link to this definition")  
The number of stages the compiler should use for software pipelining loops.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

\_\_init\_\_(*num_warps=None*, *num_stages=None*)[\#](#jax.experimental.pallas.triton.CompilerParams.__init__ "Link to this definition")  
Parameters:  
- **num_warps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **num_stages** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.triton.CompilerParams.__init__ "jax.experimental.pallas.triton.CompilerParams.__init__")(\[num_warps, num_stages\]) |  |

Attributes

|  |  |
|----|----|
| [`num_stages`](#jax.experimental.pallas.triton.CompilerParams.num_stages "jax.experimental.pallas.triton.CompilerParams.num_stages") |  |
| [`num_warps`](#jax.experimental.pallas.triton.CompilerParams.num_warps "jax.experimental.pallas.triton.CompilerParams.num_warps") |  |

[](../jax.experimental.pallas.triton.html "previous page")

previous

`jax.experimental.pallas.triton` module

[](jax.experimental.pallas.triton.atomic_and.html "next page")

next

jax.experimental.pallas.triton.atomic_and

Contents

- [`CompilerParams`](#jax.experimental.pallas.triton.CompilerParams)
  - [`CompilerParams.num_warps`](#jax.experimental.pallas.triton.CompilerParams.num_warps)
  - [`CompilerParams.num_stages`](#jax.experimental.pallas.triton.CompilerParams.num_stages)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.triton.CompilerParams.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
