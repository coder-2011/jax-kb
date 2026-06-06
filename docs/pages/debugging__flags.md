- [](../index.html)
- [Resources and Advanced Guides](../advanced_guides.html)
- [Debugging runtime values](index.html)
- JAX debugging flags

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](../_sources/debugging/flags.md "Download source file")
-  .pdf

# JAX debugging flags

## Contents

- [`jax_debug_nans` configuration option and context manager](#jax-debug-nans-configuration-option-and-context-manager)
  - [Usage](#usage)
  - [Example(s)](#example-s)
    - [Strengths and limitations of `jax_debug_nans`](#strengths-and-limitations-of-jax-debug-nans)
      - [Strengths](#strengths)
      - [Limitations](#limitations)
- [`jax_debug_infs` configuration option and context manager](#jax-debug-infs-configuration-option-and-context-manager)
- [`jax_disable_jit` configuration option and context manager](#jax-disable-jit-configuration-option-and-context-manager)
  - [Usage](#id1)
  - [Examples](#examples)
    - [Strengths and limitations of `jax_disable_jit`](#strengths-and-limitations-of-jax-disable-jit)
      - [Strengths](#id2)
      - [Limitations](#id3)

# JAX debugging flags[\#](#jax-debugging-flags "Link to this heading")

JAX offers flags and context managers that enable catching errors more easily.

## `jax_debug_nans` configuration option and context manager[\#](#jax-debug-nans-configuration-option-and-context-manager "Link to this heading")

**Summary:** Enable the `jax_debug_nans` flag to automatically detect when NaNs are produced in `jax.jit`-compiled code.

`jax_debug_nans` is a JAX flag that when enabled, will cause computations to error-out immediately on production of a NaN. Switching this option on adds a NaN check to every floating point type value produced by XLA. That means values are pulled back to the host and checked as ndarrays for every primitive operation not under an `@jax.jit`.

For code under an `@jax.jit`, the output of every `@jax.jit` function is checked and if a NaN is present it will re-run the function in de-optimized op-by-op mode, effectively removing one level of `@jax.jit` at a time.

There could be tricky situations that arise, like NaNs that only occur under a `@jax.jit` but don’t get produced in de-optimized mode. In that case you’ll see a warning message print out but your code will continue to execute.

If the NaNs are being produced in the backward pass of a gradient evaluation, when an exception is raised several frames up in the stack trace you will be in the backward_pass function, which is essentially a simple jaxpr interpreter that walks the sequence of primitive operations in reverse.

### Usage[\#](#usage "Link to this heading")

If you want to trace where NaNs are occurring in your functions or gradients, you can turn on the NaN-checker by doing one of:

- running your code inside the `jax.debug_nans` context manager, using `with`` ``jax.debug_nans(True):`;

- setting the `JAX_DEBUG_NANS=True` environment variable;

- adding `jax.config.update("jax_debug_nans",`` ``True)` near the top of your main file;

- adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_debug_nans=True`;

### Example(s)[\#](#example-s "Link to this heading")

    import jax
    import jax.numpy as jnp
    import traceback
    jax.config.update("jax_debug_nans", True)

    def f(x):
      w = 3 * jnp.square(x)
      return jnp.log(-w)

    # The stack trace is very long so only print a couple lines.
    try:
      f(5.)
    except FloatingPointError as e:
      print(traceback.format_exc(limit=2))

    Invalid nan value encountered in the output of a jax.jit function. Calling the de-optimized version.
    Traceback (most recent call last):
      File "/tmp/ipykernel_1738/1479925735.py", line 12, in <module>
        f(5.)
      File "/tmp/ipykernel_1738/1479925735.py", line 8, in f
        return jnp.log(-w)
               ^^^^^^^^^^^
    FloatingPointError: invalid value (nan) encountered in log

The NaN generated was caught. By running `%debug`, we can get a post-mortem debugger. This also works with functions under `@jax.jit`, as the example below shows.

    jax.jit(f)(5.)

    Invalid nan value encountered in the output of a jax.jit function. Calling the de-optimized version.
    Invalid nan value encountered in the output of a jax.jit function. Calling the de-optimized version.

    ---------------------------------------------------------------------------
    FloatingPointError                        Traceback (most recent call last)
    Cell In[2], line 1
    ----> 1 jax.jit(f)(5.)

        [... skipping hidden 5 frame]

    Cell In[1], line 8, in f(x)
          6 def f(x):
          7   w = 3 * jnp.square(x)
    ----> 8   return jnp.log(-w)

        [... skipping hidden 5 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/numpy/ufuncs.py:491, in log(x)
        456 @export
        457 @jit(inline=True)
        458 def log(x: ArrayLike, /) -> Array:
        459   """Calculate element-wise natural logarithm of the input.
        460 
        461   JAX implementation of :obj:`numpy.log`.
       (...)    489     Array(True, dtype=bool)
        490   """
    --> 491   out = lax.log(*promote_args_inexact('log', x))
        492   jnp_error._set_error_if_nan(out)
        493   return out

        [... skipping hidden 7 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/pjit.py:173, in _run_python_pjit(p, args_flat, fun, args, kwargs)
        171 except api_util.InternalFloatingPointError as e:
        172   if getattr(fun, '_apply_primitive', False):
    --> 173     raise FloatingPointError(
        174         f"invalid value ({e.ty}) encountered in {fun.__qualname__}") from None
        175   api_util.maybe_recursive_nan_check(e, fun, args, kwargs)  # should always raise.
        176   raise RuntimeError("Internal error") from e  # fall-back error to be safe.

    FloatingPointError: invalid value (nan) encountered in log

When this code sees a NaN in the output of an `@jax.jit` function, it calls into the de-optimized code, so we still get a clear stack trace. And we can run a post-mortem debugger with `%debug` to inspect all the values to figure out the error.

The `jax.debug_nans` context manager can be used to activate/deactivate NaN debugging. Since we activated it above with `jax.config.update`, let’s deactivate it:

    with jax.debug_nans(False):
      print(jax.jit(f)(5.))

    nan

#### Strengths and limitations of `jax_debug_nans`[\#](#strengths-and-limitations-of-jax-debug-nans "Link to this heading")

##### Strengths[\#](#strengths "Link to this heading")

- Easy to apply

- Precisely detects where NaNs were produced

- Throws a standard Python exception and is compatible with PDB postmortem

##### Limitations[\#](#limitations "Link to this heading")

- Re-running functions eagerly can be slow. You shouldn’t have the NaN-checker on if you’re not debugging, as it can introduce lots of device-host round-trips and performance regressions.

- Errors on false positives (e.g. intentionally created NaNs)

## `jax_debug_infs` configuration option and context manager[\#](#jax-debug-infs-configuration-option-and-context-manager "Link to this heading")

`jax_debug_infs` works similarly to `jax_debug_nans`. `jax_debug_infs` often needs to be combined with `jax_disable_jit`, since Infs might not cascade to the output like NaNs. Alternatively, `jax.experimental.checkify` may be used to find Infs in intermediates.

Full documentation of `jax_debug_infs` is forthcoming.

## `jax_disable_jit` configuration option and context manager[\#](#jax-disable-jit-configuration-option-and-context-manager "Link to this heading")

**Summary:** Enable the `jax_disable_jit` flag to disable JIT-compilation, enabling use of traditional Python debugging tools like `print` and `pdb`

`jax_disable_jit` is a JAX flag that when enabled, disables JIT-compilation throughout JAX (including in control flow functions like `jax.lax.cond` and `jax.lax.scan`).

### Usage[\#](#id1 "Link to this heading")

You can disable JIT-compilation by:

- setting the `JAX_DISABLE_JIT=True` environment variable;

- adding `jax.config.update("jax_disable_jit",`` ``True)` near the top of your main file;

- adding `jax.config.parse_flags_with_absl()` to your main file, then set the option using a command-line flag like `--jax_disable_jit=True`;

### Examples[\#](#examples "Link to this heading")

    import jax
    jax.config.update("jax_disable_jit", True)

    def f(x):
      y = jnp.log(x)
      if jnp.isnan(y):
        breakpoint()
      return y
    jax.jit(f)(-2.)  # ==> Enters PDB breakpoint!

#### Strengths and limitations of `jax_disable_jit`[\#](#strengths-and-limitations-of-jax-disable-jit "Link to this heading")

##### Strengths[\#](#id2 "Link to this heading")

- Easy to apply

- Enables use of Python’s built-in `breakpoint` and `print`

- Throws standard Python exceptions and is compatible with PDB postmortem

##### Limitations[\#](#id3 "Link to this heading")

- Running functions without JIT-compilation can be slow

[](index.html "previous page")

previous

Debugging runtime values

[](print_breakpoint.html "next page")

next

Compiled prints and breakpoints

Contents

- [`jax_debug_nans` configuration option and context manager](#jax-debug-nans-configuration-option-and-context-manager)
  - [Usage](#usage)
  - [Example(s)](#example-s)
    - [Strengths and limitations of `jax_debug_nans`](#strengths-and-limitations-of-jax-debug-nans)
      - [Strengths](#strengths)
      - [Limitations](#limitations)
- [`jax_debug_infs` configuration option and context manager](#jax-debug-infs-configuration-option-and-context-manager)
- [`jax_disable_jit` configuration option and context manager](#jax-disable-jit-configuration-option-and-context-manager)
  - [Usage](#id1)
  - [Examples](#examples)
    - [Strengths and limitations of `jax_disable_jit`](#strengths-and-limitations-of-jax-disable-jit)
      - [Strengths](#id2)
      - [Limitations](#id3)

By The JAX authors

© Copyright 2024, The JAX Authors.\
