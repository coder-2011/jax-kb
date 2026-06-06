- [](../index.html)
- [API Reference](../jax.html)
- [`jax.test_util` module](../jax.test_util.html)
- jax.test_util.check_jvp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.test_util.check_jvp.rst "Download source file")
-  .pdf

# jax.test_util.check_jvp

## Contents

- [`check_jvp()`](#jax.test_util.check_jvp)

# jax.test_util.check_jvp[\#](#jax-test-util-check-jvp "Link to this heading")

jax.test_util.check_jvp(*f*, *f_jvp*, *args*, *atol=None*, *rtol=None*, *eps=0.0001*, *err_msg=''*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/public_test_util.py#L250-L286)[\#](#jax.test_util.check_jvp "Link to this definition")  
Check a JVP from automatic differentiation against finite differences.

Gradients are only checked in a single randomly chosen direction, which ensures that the finite difference calculation does not become prohibitively expensive even for large input/output spaces.

Parameters:  
- **f** – function to check at `f(*args)`.

- **f_jvp** – function that calculates `jax.jvp` applied to `f`. Typically this should be `functools.partial(jax.jvp,`` ``f)`.

- **args** – tuple of argument values.

- **atol** – absolute tolerance for gradient equality.

- **rtol** – relative tolerance for gradient equality.

- **eps** – step size used for finite differences.

- **err_msg** – additional error message to include if checks fail.

Raises:  
[**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError "(in Python v3.14)") – if gradients do not match.

[](jax.test_util.check_grads.html "previous page")

previous

jax.test_util.check_grads

[](jax.test_util.check_vjp.html "next page")

next

jax.test_util.check_vjp

Contents

- [`check_jvp()`](#jax.test_util.check_jvp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
