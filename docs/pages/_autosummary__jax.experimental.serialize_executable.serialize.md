- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.serialize_executable` module](../jax.experimental.serialize_executable.html)
- jax.experimental.serialize_executable.serialize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.serialize_executable.serialize.rst "Download source file")
-  .pdf

# jax.experimental.serialize_executable.serialize

## Contents

- [`serialize()`](#jax.experimental.serialize_executable.serialize)

# jax.experimental.serialize_executable.serialize[\#](#jax-experimental-serialize-executable-serialize "Link to this heading")

jax.experimental.serialize_executable.serialize(*compiled*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/serialize_executable.py#L27-L47)[\#](#jax.experimental.serialize_executable.serialize "Link to this definition")  
Serializes a compiled binary.

Because pytrees are not serializable, they are returned so that the user can handle them properly.

Parameters:  
**compiled** ([*jax.stages.Compiled*](../jax.stages.html#jax.stages.Compiled "jax.stages.Compiled"))

[](../jax.experimental.serialize_executable.html "previous page")

previous

`jax.experimental.serialize_executable` module

[](jax.experimental.serialize_executable.deserialize_and_load.html "next page")

next

jax.experimental.serialize_executable.deserialize_and_load

Contents

- [`serialize()`](#jax.experimental.serialize_executable.serialize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
