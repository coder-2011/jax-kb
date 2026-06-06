- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.spatial.transform.Slerp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.spatial.transform.Slerp.rst "Download source file")
-  .pdf

# jax.scipy.spatial.transform.Slerp

## Contents

- [`Slerp`](#jax.scipy.spatial.transform.Slerp)
  - [`Slerp.__init__()`](#jax.scipy.spatial.transform.Slerp.__init__)

# jax.scipy.spatial.transform.Slerp[\#](#jax-scipy-spatial-transform-slerp "Link to this heading")

*class* jax.scipy.spatial.transform.Slerp(*times*, *timedelta*, *rotations*, *rotvecs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py#L209-L278)[\#](#jax.scipy.spatial.transform.Slerp "Link to this definition")  
Spherical Linear Interpolation of Rotations.

JAX implementation of [`scipy.spatial.transform.Slerp`](https://scipy.github.io/devdocs/reference/generated/scipy.spatial.transform.Slerp.html#scipy.spatial.transform.Slerp "(in SciPy v1.19.0.dev)").

Examples

Create a Slerp instance from a series of rotations:

    >>> import math
    >>> from jax.scipy.spatial.transform import Rotation, Slerp
    >>> rots = jnp.array([[90, 0, 0],
    ...                   [0, 45, 0],
    ...                   [0, 0, -30]])
    >>> key_rotations = Rotation.from_euler('zxy', rots, degrees=True)
    >>> key_times = [0, 1, 2]
    >>> slerp = Slerp.init(key_times, key_rotations)
    >>> times = [0, 0.5, 1, 1.5, 2]
    >>> interp_rots = slerp(times)
    >>> interp_rots.as_euler('zxy')
    Array([[ 1.5707963e+00,  0.0000000e+00,  0.0000000e+00],
           [ 8.5309029e-01,  3.8711953e-01,  1.7768645e-01],
           [-2.3841858e-07,  7.8539824e-01,  0.0000000e+00],
           [-5.6668043e-02,  3.9213133e-01, -2.8347540e-01],
           [ 0.0000000e+00,  0.0000000e+00, -5.2359891e-01]], dtype=float32)

Parameters:  
- **times** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **timedelta** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **rotations** ([*Rotation*](jax.scipy.spatial.transform.Rotation.html#jax.scipy.spatial.transform.Rotation "jax.scipy.spatial.transform.Rotation"))

- **rotvecs** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

\_\_init\_\_()[\#](#jax.scipy.spatial.transform.Slerp.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.scipy.spatial.transform.Slerp.__init__ "jax.scipy.spatial.transform.Slerp.__init__")() |  |
| `count`(value, /) | Return number of occurrences of value. |
| `index`(value\[, start, stop\]) | Return first index of value. |
| `init`(times, rotations) |  |

Attributes

|             |                          |
|-------------|--------------------------|
| `rotations` | Alias for field number 2 |
| `rotvecs`   | Alias for field number 3 |
| `timedelta` | Alias for field number 1 |
| `times`     | Alias for field number 0 |

[](jax.scipy.spatial.transform.Rotation.html "previous page")

previous

jax.scipy.spatial.transform.Rotation

[](jax.scipy.sparse.linalg.bicgstab.html "next page")

next

jax.scipy.sparse.linalg.bicgstab

Contents

- [`Slerp`](#jax.scipy.spatial.transform.Slerp)
  - [`Slerp.__init__()`](#jax.scipy.spatial.transform.Slerp.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
