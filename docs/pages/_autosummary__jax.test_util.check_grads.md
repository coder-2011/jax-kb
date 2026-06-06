- [](../index.html)
- [API Reference](../jax.html)
- [`jax.test_util` module](../jax.test_util.html)
- jax.test_util.check_grads

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.test_util.check_grads.rst "Download source file")
-  .pdf

# jax.test_util.check_grads

## Contents

- [`check_grads()`](#jax.test_util.check_grads)

# jax.test_util.check_grads[\#](#jax-test-util-check-grads "Link to this heading")

jax.test_util.check_grads(*f*, *args*, *order*, *modes=('fwd', 'rev')*, *atol=None*, *rtol=None*, *eps=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L326-L375)[\#](#jax.test_util.check_grads "Link to this definition")  
Check gradients from automatic differentiation against finite differences.

Gradients are only checked in a single randomly chosen direction, which ensures that the finite difference calculation does not become prohibitively expensive even for large input/output spaces.

Parameters:  
- **f** – function to check at `f(*args)`.

- **args** – tuple of argument values.

- **order** – forward and backwards gradients up to this order are checked.

- **modes** – lists of gradient modes to check (‘fwd’ and/or ‘rev’).

- **atol** – absolute tolerance for gradient equality.

- **rtol** – relative tolerance for gradient equality.

- **eps** – step size used for finite differences.

Raises:  
[**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.14)") – if gradients do not match.

[](../jax.test_util.html "previous page")

previous

`jax.test_util` module

[](jax.test_util.check_jvp.html "next page")

next

jax.test_util.check_jvp

Contents

- [`check_grads()`](#jax.test_util.check_grads)

By The JAX authors

© Copyright 2024, The JAX Authors.\
