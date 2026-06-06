- [](../index.html)
- [API Reference](../jax.html)
- jax.value_and_grad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.value_and_grad.rst "Download source file")
-  .pdf

# jax.value_and_grad

## Contents

- [`value_and_grad()`](#jax.value_and_grad)

# jax.value_and_grad[\#](#jax-value-and-grad "Link to this heading")

jax.value_and_grad(*fun*, *argnums=0*, *has_aux=False*, *holomorphic=False*, *allow_int=False*, *reduce_axes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L481-L548)[\#](#jax.value_and_grad "Link to this definition")  
Create a function that evaluates both `fun` and the gradient of `fun`.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function to be differentiated. Its arguments at positions specified by `argnums` should be arrays, scalars, or standard Python containers. It should return a scalar (which includes arrays with shape `()` but not arrays with shape `(1,)` etc.)

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default 0).

- **has_aux** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. Default False.

- **holomorphic** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` is promised to be holomorphic. If True, inputs and outputs must be complex. Default False.

- **allow_int** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Whether to allow differentiating with respect to integer valued inputs. The gradient of an integer input will have a trivial vector-space dtype (float0). Default False.

- **reduce_axes** (*Sequence\[AxisName\]*)

Returns:  
A function with the same arguments as `fun` that evaluates both `fun` and the gradient of `fun` and returns them as a pair (a two-element tuple). If `argnums` is an integer then the gradient has the same shape and type as the positional argument indicated by that integer. If argnums is a sequence of integers, the gradient is a tuple of values with the same shapes and types as the corresponding arguments. If `has_aux` is True then a tuple of ((value, auxiliary_data), gradient) is returned.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, Any\]\]

[](jax.grad.html "previous page")

previous

jax.grad

[](jax.jacobian.html "next page")

next

jax.jacobian

Contents

- [`value_and_grad()`](#jax.value_and_grad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
