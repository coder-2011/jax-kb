- [](../index.html)
- [API Reference](../jax.html)
- [`jax.test_util` module](../jax.test_util.html)
- jax.test_util.check_vjp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.test_util.check_vjp.rst "Download source file")
-  .pdf

# jax.test_util.check_vjp

## Contents

- [`check_vjp()`](#jax.test_util.check_vjp)

# jax.test_util.check_vjp[\#](#jax-test-util-check-vjp "Link to this heading")

jax.test_util.check_vjp(*f*, *f_vjp*, *args*, *atol=None*, *rtol=None*, *eps=0.0001*, *err_msg=''*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L288-L324)[\#](#jax.test_util.check_vjp "Link to this definition")  
Check a VJP from automatic differentiation against finite differences.

Gradients are only checked in a single randomly chosen direction, which ensures that the finite difference calculation does not become prohibitively expensive even for large input/output spaces.

Parameters:  
- **f** – function to check at `f(*args)`.

- **f_vjp** – function that calculates `jax.vjp` applied to `f`. Typically this should be `functools.partial(jax.vjp,`` ``f)`.

- **args** – tuple of argument values.

- **atol** – absolute tolerance for gradient equality.

- **rtol** – relative tolerance for gradient equality.

- **eps** – step size used for finite differences.

- **err_msg** – additional error message to include if checks fail.

Raises:  
[**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.14)") – if gradients do not match.

[](jax.test_util.check_jvp.html "previous page")

previous

jax.test_util.check_jvp

[](../jax.tree.html "next page")

next

`jax.tree` module

Contents

- [`check_vjp()`](#jax.test_util.check_vjp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
