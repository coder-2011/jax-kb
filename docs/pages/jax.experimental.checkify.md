- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.checkify` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.checkify.rst "Download source file")
-  .pdf

# jax.experimental.checkify module

## Contents

- [API](#api)

# `jax.experimental.checkify` module[\#](#module-jax.experimental.checkify "Link to this heading")

## API[\#](#api "Link to this heading")

|  |  |
|----|----|
| [`checkify`](_autosummary/jax.experimental.checkify.checkify.html#jax.experimental.checkify.checkify "jax.experimental.checkify.checkify")(f\[, errors\]) | Functionalize check calls in fun, and optionally add run-time error checks. |
| [`check`](_autosummary/jax.experimental.checkify.check.html#jax.experimental.checkify.check "jax.experimental.checkify.check")(pred, msg, \*fmt_args\[, debug\]) | Check a predicate, add an error with msg if predicate is False. |
| [`check_error`](_autosummary/jax.experimental.checkify.check_error.html#jax.experimental.checkify.check_error "jax.experimental.checkify.check_error")(error) | Raise an Exception if `error` represents a failure. |
| [`Error`](_autosummary/jax.experimental.checkify.Error.html#jax.experimental.checkify.Error "jax.experimental.checkify.Error")(\_pred, \_code, \_metadata, \_payload) |  |
| [`JaxRuntimeError`](_autosummary/jax.experimental.checkify.JaxRuntimeError.html#jax.experimental.checkify.JaxRuntimeError "jax.experimental.checkify.JaxRuntimeError") |  |
| [`user_checks`](_autosummary/jax.experimental.checkify.user_checks.html#jax.experimental.checkify.user_checks "jax.experimental.checkify.user_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`nan_checks`](_autosummary/jax.experimental.checkify.nan_checks.html#jax.experimental.checkify.nan_checks "jax.experimental.checkify.nan_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`index_checks`](_autosummary/jax.experimental.checkify.index_checks.html#jax.experimental.checkify.index_checks "jax.experimental.checkify.index_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`div_checks`](_autosummary/jax.experimental.checkify.div_checks.html#jax.experimental.checkify.div_checks "jax.experimental.checkify.div_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`float_checks`](_autosummary/jax.experimental.checkify.float_checks.html#jax.experimental.checkify.float_checks "jax.experimental.checkify.float_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`automatic_checks`](_autosummary/jax.experimental.checkify.automatic_checks.html#jax.experimental.checkify.automatic_checks "jax.experimental.checkify.automatic_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |
| [`all_checks`](_autosummary/jax.experimental.checkify.all_checks.html#jax.experimental.checkify.all_checks "jax.experimental.checkify.all_checks") | frozenset() -\> empty frozenset object frozenset(iterable) -\> frozenset object |

[](jax.experimental.html "previous page")

previous

`jax.experimental` module

[](_autosummary/jax.experimental.checkify.checkify.html "next page")

next

jax.experimental.checkify.checkify

Contents

- [API](#api)

By The JAX authors

© Copyright 2024, The JAX Authors.\
