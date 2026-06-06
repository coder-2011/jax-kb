- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.checkify` module](../jax.experimental.checkify.html)
- jax.experimental.checkify.check_error

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.checkify.check_error.rst "Download source file")
-  .pdf

# jax.experimental.checkify.check_error

## Contents

- [`check_error()`](#jax.experimental.checkify.check_error)

# jax.experimental.checkify.check_error[\#](#jax-experimental-checkify-check-error "Link to this heading")

jax.experimental.checkify.check_error(*error*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/checkify.py#L1330-L1395)[\#](#jax.experimental.checkify.check_error "Link to this definition")  
Raise an Exception if `error` represents a failure. Functionalized by [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify").

The semantics of this function are equivalent to:

    >>> def check_error(err: Error) -> None:
    ...   err.throw()  # can raise ValueError

But unlike that implementation, `check_error` can be functionalized using the [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify") transformation.

This function is similar to [`check()`](jax.experimental.checkify.check.html#jax.experimental.checkify.check "jax.experimental.checkify.check") but with a different signature: whereas [`check()`](jax.experimental.checkify.check.html#jax.experimental.checkify.check "jax.experimental.checkify.check") takes as arguments a boolean predicate and a new error message string, this function takes an `Error` value as argument. Both [`check()`](jax.experimental.checkify.check.html#jax.experimental.checkify.check "jax.experimental.checkify.check") and this function raise a Python Exception on failure (a side-effect), and thus cannot be staged out by [`jit()`](jax.jit.html#jax.jit "jax.jit"), [`pmap()`](jax.pmap.html#jax.pmap "jax.pmap"), [`scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan"), etc. Both also can be functionalized by using [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify").

But unlike [`check()`](jax.experimental.checkify.check.html#jax.experimental.checkify.check "jax.experimental.checkify.check"), this function is like a direct inverse of [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify"): whereas [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify") takes as input a function which can raise a Python Exception and produces a new function without that effect but which produces an `Error` value as output, this `check_error` function can accept an `Error` value as input and can produce the side-effect of raising an Exception. That is, while [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify") goes from functionalizable Exception effect to error value, this `check_error` goes from error value to functionalizable Exception effect.

`check_error` is useful when you want to turn checks represented by an `Error` value (produced by functionalizing `checks` via [`checkify()`](jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify")) back into Python Exceptions.

Parameters:  
**error** ([*Error*](jax.experimental.checkify.Error.html#jax.experimental.checkify.Error "jax.experimental.checkify.Error")) – Error to check.

Return type:  
None

For example, you might want to functionalize part of your program through checkify, stage out your functionalized code through [`jit()`](jax.jit.html#jax.jit "jax.jit"), then re-inject your error value outside of the [`jit()`](jax.jit.html#jax.jit "jax.jit"):

    >>> import jax
    >>> from jax.experimental import checkify
    >>> def f(x):
    ...   checkify.check(x>0, "must be positive!")
    ...   return x
    >>> def with_inner_jit(x):
    ...   checked_f = checkify.checkify(f)
    ...   # a checkified function can be jitted
    ...   error, out = jax.jit(checked_f)(x)
    ...   checkify.check_error(error)
    ...   return out
    >>> _ = with_inner_jit(1)  # no failed check
    >>> with_inner_jit(-1)  
    Traceback (most recent call last):
      ...
    jax._src.JaxRuntimeError: must be positive!
    >>> # can re-checkify
    >>> error, _ = checkify.checkify(with_inner_jit)(-1)

[](jax.experimental.checkify.check.html "previous page")

previous

jax.experimental.checkify.check

[](jax.experimental.checkify.Error.html "next page")

next

jax.experimental.checkify.Error

Contents

- [`check_error()`](#jax.experimental.checkify.check_error)

By The JAX authors

© Copyright 2024, The JAX Authors.\
