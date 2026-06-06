- [](index.html)
- [API Reference](jax.html)
- `jax.extend` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.extend.rst "Download source file")
-  .pdf

# jax.extend module

## Contents

- [API policy](#api-policy)
- [Modules](#modules)

# `jax.extend` module[\#](#module-jax.extend "Link to this heading")

Modules for JAX extensions.

The [`jax.extend`](#module-jax.extend "jax.extend") module provides modules for access to JAX internal machinery. See [JEP \#15856](https://docs.jax.dev/en/latest/jep/15856-jex.html).

This module is not the only means by which JAX aims to be extensible. For example, the main JAX API offers mechanisms for [customizing derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html), [registering custom pytree definitions](https://docs.jax.dev/en/latest/custom_pytrees.html#pytrees-custom-pytree-nodes), and more.

## API policy[\#](#api-policy "Link to this heading")

Unlike the [public API](https://docs.jax.dev/en/latest/api_compatibility.html), this module offers **no compatibility guarantee** across releases. Breaking changes will be announced via the [JAX project changelog](https://docs.jax.dev/en/latest/changelog.html).

## Modules[\#](#modules "Link to this heading")

- [`jax.extend.backend` module](jax.extend.backend.html)
- [`jax.extend.core` module](jax.extend.core.html)
- [`jax.extend.linear_util` module](jax.extend.linear_util.html)
- [`jax.extend.lowering` module](jax.extend.lowering.html)
- [`jax.extend.mlir` module](jax.extend.mlir.html)
- [`jax.extend.pallas` module](jax.extend.pallas.html)
- [`jax.extend.random` module](jax.extend.random.html)
- [`jax.extend.xla` module](jax.extend.xla.html)

[](_autosummary/jax.export.SymbolicScope.html "previous page")

previous

jax.export.SymbolicScope

[](jax.extend.backend.html "next page")

next

`jax.extend.backend` module

Contents

- [API policy](#api-policy)
- [Modules](#modules)

By The JAX authors

© Copyright 2024, The JAX Authors.\
