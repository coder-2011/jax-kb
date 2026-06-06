- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.compilation_cache` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.compilation_cache.rst "Download source file")
-  .pdf

# jax.experimental.compilation_cache module

## Contents

- [API](#api)
  - [`set_cache_dir()`](#jax.experimental.compilation_cache.compilation_cache.set_cache_dir)
  - [`reset_cache()`](#jax.experimental.compilation_cache.compilation_cache.reset_cache)

# `jax.experimental.compilation_cache` module[\#](#jax-experimental-compilation-cache-module "Link to this heading")

JAX disk compilation cache.

## API[\#](#api "Link to this heading")

jax.experimental.compilation_cache.compilation_cache.set_cache_dir(*path*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/compilation_cache.py#L191-L209)[\#](#jax.experimental.compilation_cache.compilation_cache.set_cache_dir "Link to this definition")  
Sets the persistent compilation cache directory.

After calling this, jit-compiled functions are saved to path, so they do not need be recompiled if the process is restarted or otherwise run again. This also tells Jax where to look for compiled functions before compiling.

For more information, see the [persistent compilation cache guide](persistent_compilation_cache.html#persistent-compilation-cache).

Warning

The compilation cache is considered trusted. Do not share a compilation cache with users you do not trust. For example, if you put the compilation cache in a directory to which others may write, those users can trigger your JAX process to run arbitrary code. Sharing a compilation cache is equivalent to allowing anyone who can write to the cache directory to run code on your machine.

Return type:  
None

&nbsp;

jax.experimental.compilation_cache.compilation_cache.reset_cache()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/compilation_cache.py#L430-L446)[\#](#jax.experimental.compilation_cache.compilation_cache.reset_cache "Link to this definition")  
Get back to pristine, uninitialized state.

For more information, see the [persistent compilation cache guide](persistent_compilation_cache.html#persistent-compilation-cache).

Return type:  
None

[](_autosummary/jax.experimental.checkify.all_checks.html "previous page")

previous

jax.experimental.checkify.all_checks

[](jax.experimental.custom_partitioning.html "next page")

next

`jax.experimental.custom_partitioning` module

Contents

- [API](#api)
  - [`set_cache_dir()`](#jax.experimental.compilation_cache.compilation_cache.set_cache_dir)
  - [`reset_cache()`](#jax.experimental.compilation_cache.compilation_cache.reset_cache)

By The JAX authors

© Copyright 2024, The JAX Authors.\
