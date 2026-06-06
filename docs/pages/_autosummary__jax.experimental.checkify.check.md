- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.checkify` module](../jax.experimental.checkify.html)
- jax.experimental.checkify.check

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.checkify.check.rst "Download source file")
-  .pdf

# jax.experimental.checkify.check

## Contents

- [`check()`](#jax.experimental.checkify.check)

# jax.experimental.checkify.check[\#](#jax-experimental-checkify-check "Link to this heading")

jax.experimental.checkify.check(*pred*, *msg*, *\*fmt_args*, *debug=False*, *\*\*fmt_kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/checkify.py#L1224-L1265)[\#](#jax.experimental.checkify.check "Link to this definition")  
Check a predicate, add an error with msg if predicate is False.

This is an effectful operation, and can’t be staged (jitted/scanned/…). Before staging a function with checks, [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify") it!

Parameters:  
- **pred** (*Bool*) – if False, a FailedCheckError error is added.

- **msg** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – error message if error is added. Can be a format string.

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to turn on debugging mode. If True, check will be removed during execution. If False, the check must be functionalized using checkify.checkify.

- **fmt_args** – Positional and keyword formatting arguments for msg, eg.: `check(..,`` ``"check`` ``failed`` ``on`` ``values`` ``{}`` ``and`` ``{named_arg}",`` ``x,`` ``named_arg=y)` Note that these arguments can be traced values allowing you to add run-time values to the error message. Note that tracking these run-time arrays will increase your memory usage, even if no error happens.

- **fmt_kwargs** – Positional and keyword formatting arguments for msg, eg.: `check(..,`` ``"check`` ``failed`` ``on`` ``values`` ``{}`` ``and`` ``{named_arg}",`` ``x,`` ``named_arg=y)` Note that these arguments can be traced values allowing you to add run-time values to the error message. Note that tracking these run-time arrays will increase your memory usage, even if no error happens.

Return type:  
None

For example:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jax.experimental import checkify
    >>> def f(x):
    ...   checkify.check(x>0, "{x} needs to be positive!", x=x)
    ...   return 1/x
    >>> checked_f = checkify.checkify(f)
    >>> err, out = jax.jit(checked_f)(-3.)
    >>> err.throw()  
    Traceback (most recent call last):
      ...
    jax._src.checkify.JaxRuntimeError: -3. needs to be positive!

[](jax.experimental.checkify.checkify.html "previous page")

previous

jax.experimental.checkify.checkify

[](jax.experimental.checkify.check_error.html "next page")

next

jax.experimental.checkify.check_error

Contents

- [`check()`](#jax.experimental.checkify.check)

By The JAX authors

© Copyright 2024, The JAX Authors.\
