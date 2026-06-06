- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.checkify` module](../jax.experimental.checkify.html)
- jax.experimental.checkify.Error

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.checkify.Error.rst "Download source file")
-  .pdf

# jax.experimental.checkify.Error

## Contents

- [`Error`](#jax.experimental.checkify.Error)
  - [`Error.__init__()`](#jax.experimental.checkify.Error.__init__)

# jax.experimental.checkify.Error[\#](#jax-experimental-checkify-error "Link to this heading")

*class* jax.experimental.checkify.Error(*\_pred: 'dict\[ErrorEffect, Bool\]'*, *\_code: 'dict\[ErrorEffect, Int\]'*, *\_metadata: 'dict\[Int, PyTreeDef\]'*, *\_payload: 'dict\[ErrorEffect, Payload\]'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/checkify.py#L214-L312)[\#](#jax.experimental.checkify.Error "Link to this definition")  
Parameters:  
- **\_pred** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Bool\]*)

- **\_code** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Int\]*)

- **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[Int,* *PyTreeDef\]*)

- **\_payload** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Payload\]*)

\_\_init\_\_(*\_pred*, *\_code*, *\_metadata*, *\_payload*)[\#](#jax.experimental.checkify.Error.__init__ "Link to this definition")  
Parameters:  
- **\_pred** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Bool\]*)

- **\_code** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Int\]*)

- **\_metadata** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[Int,* *PyTreeDef\]*)

- **\_payload** ([*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*\[ErrorEffect,* *Payload\]*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.checkify.Error.__init__ "jax.experimental.checkify.Error.__init__")(\_pred, \_code, \_metadata, \_payload) |  |
| `get`() | Returns error message if error happened, None if no error happened. |
| `get_exception`() | Returns Python exception if error happened, None if no error happened. |
| `throw`() |  |
| `tree_flatten`() |  |
| `tree_unflatten`(metadata, data) |  |

[](jax.experimental.checkify.check_error.html "previous page")

previous

jax.experimental.checkify.check_error

[](jax.experimental.checkify.JaxRuntimeError.html "next page")

next

jax.experimental.checkify.JaxRuntimeError

Contents

- [`Error`](#jax.experimental.checkify.Error)
  - [`Error.__init__()`](#jax.experimental.checkify.Error.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
