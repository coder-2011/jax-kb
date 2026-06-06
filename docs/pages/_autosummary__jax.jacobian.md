- [](../index.html)
- [API Reference](../jax.html)
- jax.jacobian

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.jacobian.rst "Download source file")
-  .pdf

# jax.jacobian

## Contents

- [`jacobian()`](#jax.jacobian)

# jax.jacobian[\#](#jax-jacobian "Link to this heading")

jax.jacobian(*fun*, *argnums=0*, *has_aux=False*, *holomorphic=False*, *allow_int=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L822-L826)[\#](#jax.jacobian "Link to this definition")  
Alias of [`jax.jacrev()`](jax.jacrev.html#jax.jacrev "jax.jacrev").

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **has_aux** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **holomorphic** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **allow_int** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

[](jax.value_and_grad.html "previous page")

previous

jax.value_and_grad

[](jax.jacfwd.html "next page")

next

jax.jacfwd

Contents

- [`jacobian()`](#jax.jacobian)

By The JAX authors

© Copyright 2024, The JAX Authors.\
