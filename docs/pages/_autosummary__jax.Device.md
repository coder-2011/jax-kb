- [](../index.html)
- [API Reference](../jax.html)
- jax.Device

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Device.rst "Download source file")
-  .pdf

# jax.Device

## Contents

- [`Device`](#jax.Device)
  - [`Device.__init__()`](#jax.Device.__init__)

# jax.Device[\#](#jax-device "Link to this heading")

*class* jax.Device[\#](#jax.Device "Link to this definition")  
A descriptor of an available device.

Subclasses are used to represent specific types of devices, e.g. CPUs, GPUs. Subclasses may have additional properties specific to that device type.

\_\_init\_\_(*\*args*, *\*\*kwargs*)[\#](#jax.Device.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.Device.__init__ "jax.Device.__init__")(\*args, \*\*kwargs) |  |

Attributes

|  |  |
|----|----|
| `addressable_memories` | Returns all the memories that a device can address. |
| `client` | (self) -\> jax::PyClient |
| `default_memory` | Returns the default memory of a device. |
| `device_kind` | (self) -\> str |
| `get_stream_for_external_ready_events` |  |
| `host_id` | Deprecated; please use process_index |
| `id` | Integer ID of this device. |
| `live_buffers` |  |
| `local_hardware_id` | Opaque hardware ID, e.g., the CUDA device number. |
| `memory` |  |
| `memory_stats` | Returns memory statistics for this device keyed by name. |
| `platform` | (self) -\> str |
| `poison_execution` |  |
| `process_index` | Integer index of this device's process. |
| `task_id` | Deprecated; please use process_index |

[](jax.experimental.io_callback.html "previous page")

previous

jax.experimental.io_callback

[](jax.print_environment_info.html "next page")

next

jax.print_environment_info

Contents

- [`Device`](#jax.Device)
  - [`Device.__init__()`](#jax.Device.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
