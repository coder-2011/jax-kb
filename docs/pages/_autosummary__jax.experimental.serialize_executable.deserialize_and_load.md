- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.serialize_executable` module](../jax.experimental.serialize_executable.html)
- jax.experimental.serialize_executable.deserialize_and_load

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.serialize_executable.deserialize_and_load.rst "Download source file")
-  .pdf

# jax.experimental.serialize_executable.deserialize_and_load

## Contents

- [`deserialize_and_load()`](#jax.experimental.serialize_executable.deserialize_and_load)

# jax.experimental.serialize_executable.deserialize_and_load[\#](#jax-experimental-serialize-executable-deserialize-and-load "Link to this heading")

jax.experimental.serialize_executable.deserialize_and_load(*serialized*, *in_tree*, *out_tree*, *backend=None*, *execution_devices=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/serialize_executable.py#L49-L87)[\#](#jax.experimental.serialize_executable.deserialize_and_load "Link to this definition")  
Constructs a [`jax.stages.Compiled`](../jax.stages.html#jax.stages.Compiled "jax.stages.Compiled") from a serialized executable.

Warning

It is not safe to call this API with untrusted inputs. Do not do this. Calling this API loads a serialized executable. Even loading such an executable may run arbitrary code on your machine. It is not safe to pass untrusted data here and likely never will be.

Parameters:  
- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xc.Client* *\|* *None*)

- **execution_devices** (*Sequence\[xc.Device\]* *\|* *None*)

[](jax.experimental.serialize_executable.serialize.html "previous page")

previous

jax.experimental.serialize_executable.serialize

[](../jax.experimental.sparse.html "next page")

next

`jax.experimental.sparse` module

Contents

- [`deserialize_and_load()`](#jax.experimental.serialize_executable.deserialize_and_load)

By The JAX authors

© Copyright 2024, The JAX Authors.\
