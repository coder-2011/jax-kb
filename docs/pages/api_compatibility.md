- [](index.html)
- [Notes](notes.html)
- API compatibility

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/api_compatibility.md "Download source file")
-  .pdf

# API compatibility

## Contents

- [JAX Versioning](#jax-versioning)
- [What is covered?](#what-is-covered)
- [What is not covered?](#what-is-not-covered)
  - [Explicitly private APIs](#explicitly-private-apis)
  - [jaxlib](#jaxlib)
  - [Legacy internal APIs](#legacy-internal-apis)
  - [Experimental and example libraries](#experimental-and-example-libraries)
  - [JAX extend](#jax-extend)
- [Numerics and randomness](#numerics-and-randomness)

# API compatibility[\#](#api-compatibility "Link to this heading")

JAX is constantly evolving, and we want to be able to make improvements to its APIs. That said, we want to minimize churn for the JAX user community, and we try to make breaking changes rarely.

## JAX Versioning[\#](#jax-versioning "Link to this heading")

JAX uses [Effort-based versioning](https://jacobtomlinson.dev/effver/) (see [JEP 25516: Effort-based versioning for JAX](jep/25516-effver.html#jep-effver)), and is currently in the Zero version phase. This means that for version `0.X.Y`, incrementing `Y` will introduce minor breaking changes, and incrementing `X` will introduce major breaking changes.

For any breaking change, JAX currently follows a 3 month deprecation policy. When an incompatible change is made to an API, we will make our best effort to obey the following procedure:

- the change will be announced in `CHANGELOG.md` and in the doc string for the deprecated API, and the old API will issue a `DeprecationWarning`.

- three months after the `jax` release that deprecated an API, we may remove the deprecated API at any time. Note that three months is a *lower* bound, and is intentionally chosen to be faster than that of many more mature projects. In practice, deprecations may take considerably longer, particularly if there are many users of a feature. If a three month deprecation period becomes problematic, please raise this with us.

We reserve the right to change this policy at any time.

## What is covered?[\#](#what-is-covered "Link to this heading")

Only public JAX APIs are covered, which includes the following modules:

- `jax`

- `jax.dlpack`

- `jax.image`

- `jax.lax`

- `jax.nn`

- `jax.numpy`

- `jax.ops`

- `jax.profiler`

- `jax.random` (see [details below](#numerics-and-randomness))

- `jax.scipy`

- `jax.tree`

- `jax.tree_util`

- `jax.test_util`

Not everything in these modules is intended to be public, and over time, we are working to separate public and private APIs. Public APIs are documented in the JAX documentation. Additionally, our goal is that all non-public APIs should have names prefixed with underscores, although we do not entirely comply with this yet.

## What is not covered?[\#](#what-is-not-covered "Link to this heading")

### Explicitly private APIs[\#](#explicitly-private-apis "Link to this heading")

Any API or import path prefixed with an underscore is explicitly private, and may change without warning between JAX releases. We are working to move all private APIs into `jax._src` to make these expectations more clear.

### jaxlib[\#](#jaxlib "Link to this heading")

Any import path in the `jaxlib` package is considered private, and may change without warning between releases. Some APIs defined in `jaxlib` have public aliases in the `jax` package.

### Legacy internal APIs[\#](#legacy-internal-apis "Link to this heading")

In addition, there are several legacy modules that currently expose some private APIs without an underscore, including:

- `jax.core`

- `jax.interpreters`

- `jax.lib`

- `jax.util`

We are actively working on deprecating these modules and the APIs they contain. In most cases, such deprecations will follow the 3 month deprecation period, but this may not always be possible. If you use any such APIs, please expect them to be deprecated soon, and seek alternatives.

### Experimental and example libraries[\#](#experimental-and-example-libraries "Link to this heading")

The following modules include code for experimental or demonstration purposes, and API may change between releases without warning:

- `jax.experimental`

- `jax.example_libraries`

We understand that some users depend on `jax.experimental`, and so in most cases we follow the 3 month deprecation period for changes, but this may not always be possible.

### JAX extend[\#](#jax-extend "Link to this heading")

The [`jax.extend`](jax.extend.html#module-jax.extend "jax.extend") module includes semi-public JAX internal APIs that are meant for use by downstream projects, but do not have the same stability guarantees of the main JAX package. If you have code that uses `jax.extend`, we would strongly recommend CI tests against JAX’s nightly releases, so as to catch potential changes before they are released.

For details on `jax.extend`, see the [`jax.extend` module documentation](https://docs.jax.dev/en/latest/jax.extend.html), or the design document, [jax.extend: a module for extensions](jep/15856-jex.html#jax-extend-jep).

## Numerics and randomness[\#](#numerics-and-randomness "Link to this heading")

The *exact* values of numerical operations are not guaranteed to be stable across JAX releases. In fact, exact numerics are not necessarily stable at a given JAX version, across accelerator platforms, within or without `jax.jit`, and more.

For a fixed PRNG key input, the outputs of pseudorandom functions in `jax.random` may vary across JAX versions. The compatibility policy applies only to the output *distribution*. For example, the expression `jax.random.gumbel(jax.random.key(72))` may return a different value across JAX releases, but `jax.random.gumbel` will remain a pseudorandom generator for the Gumbel distribution.

We try to make such changes to pseudorandom values infrequently. When they happen, the changes are announced in the changelog, but do not follow a deprecation cycle. In some situations, JAX might expose a transient configuration flag that reverts the new behavior, to help users diagnose and update affected code. Such flags will last a deprecation window’s amount of time.

[](notes.html "previous page")

previous

Notes

[](deprecation.html "next page")

next

Python and NumPy version support policy

Contents

- [JAX Versioning](#jax-versioning)
- [What is covered?](#what-is-covered)
- [What is not covered?](#what-is-not-covered)
  - [Explicitly private APIs](#explicitly-private-apis)
  - [jaxlib](#jaxlib)
  - [Legacy internal APIs](#legacy-internal-apis)
  - [Experimental and example libraries](#experimental-and-example-libraries)
  - [JAX extend](#jax-extend)
- [Numerics and randomness](#numerics-and-randomness)

By The JAX authors

© Copyright 2024, The JAX Authors.\
