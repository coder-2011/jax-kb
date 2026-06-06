- [](../index.html)
- [API Reference](../jax.html)
- jax.jacrev

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.jacrev.rst "Download source file")
-  .pdf

# jax.jacrev

## Contents

- [`jacrev()`](#jax.jacrev)

# jax.jacrev[\#](#jax-jacrev "Link to this heading")

jax.jacrev(*fun*, *argnums=0*, *has_aux=False*, *holomorphic=False*, *allow_int=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L762-L820)[\#](#jax.jacrev "Link to this definition")  
Jacobian of `fun` evaluated row-by-row using reverse-mode AD.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function whose Jacobian is to be computed.

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default `0`).

- **has_aux** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. Default False.

- **holomorphic** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` is promised to be holomorphic. Default False.

- **allow_int** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Whether to allow differentiating with respect to integer valued inputs. The gradient of an integer input will have a trivial vector-space dtype (float0). Default False.

Returns:  
A function with the same arguments as `fun`, that evaluates the Jacobian of `fun` using reverse-mode automatic differentiation. If `has_aux` is True then a pair of (jacobian, auxiliary_data) is returned.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x):
    ...   return jnp.asarray(
    ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
    ...
    >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
    [[ 1.       0.       0.     ]
     [ 0.       0.       5.     ]
     [ 0.      16.      -2.     ]
     [ 1.6209   0.       0.84147]]

[](jax.jacfwd.html "previous page")

previous

jax.jacfwd

[](jax.hessian.html "next page")

next

jax.hessian

Contents

- [`jacrev()`](#jax.jacrev)

By The JAX authors

© Copyright 2024, The JAX Authors.\
