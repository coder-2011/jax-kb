- [](../index.html)
- [API Reference](../jax.html)
- jax.closure_convert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.closure_convert.rst "Download source file")
-  .pdf

# jax.closure_convert

## Contents

- [`closure_convert()`](#jax.closure_convert)

# jax.closure_convert[\#](#jax-closure-convert "Link to this heading")

jax.closure_convert(*fun*, *\*example_args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L1263-L1332)[\#](#jax.closure_convert "Link to this definition")  
Closure conversion utility, for use with higher-order custom derivatives.

To define custom derivatives such as with `jax.custom_vjp(f)`, the target function `f` must take, as formal arguments, all values involved in differentiation. If `f` is a higher-order function, in that it accepts as an argument a Python function `g`, then values stored away in `g`’s closure will not be visible to the custom derivative rules, and attempts at AD involving these values will fail. One way around this is to convert the closure by extracting these values, and to pass them as explicit formal arguments across the custom derivative boundary. This utility carries out that conversion. More precisely, it closure-converts the function `fun` specialized to the types of the arguments given in `example_args`.

When we refer here to “values in the closure” of `fun`, we do not mean the values that are captured by Python directly when `fun` is defined (e.g. the Python objects in `fun.__closure__`, if the attribute exists). Rather, we mean values encountered during the execution of `fun` on `example_args` that determine its output. This may include, for instance, arrays captured transitively in Python closures, i.e. in the Python closure of functions called by `fun`, the closures of the functions that they call, and so forth.

The function `fun` must be a pure function.

Example usage:

    def minimize(objective_fn, x0):
      converted_fn, aux_args = closure_convert(objective_fn, x0)
      return _minimize(converted_fn, x0, *aux_args)

    @partial(custom_vjp, nondiff_argnums=(0,))
    def _minimize(objective_fn, x0, *args):
      z = objective_fn(x0, *args)
      # ... find minimizer x_opt ...
      return x_opt

    def fwd(objective_fn, x0, *args):
      y = _minimize(objective_fn, x0, *args)
      return y, (y, args)

    def rev(objective_fn, res, g):
      y, args = res
      y_bar = g
      # ... custom reverse-mode AD ...
      return x0_bar, *args_bars

    _minimize.defvjp(fwd, rev)

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Python callable to be converted. Must be a pure function.

- **example_args** – Arrays, scalars, or (nested) standard Python containers (tuples, lists, dicts, namedtuples, i.e., pytrees) thereof, used to determine the types of the formal arguments to `fun`. This type-specialized form of `fun` is the function that will be closure converted.

Returns:  
A pair comprising (i) a Python callable, accepting the same arguments as `fun` followed by arguments corresponding to the values hoisted from its closure, and (ii) a list of values hoisted from the closure.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"), [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[Any\]\]

[](jax.custom_gradient.html "previous page")

previous

jax.custom_gradient

[](jax.checkpoint.html "next page")

next

jax.checkpoint

Contents

- [`closure_convert()`](#jax.closure_convert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
