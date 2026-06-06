- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.spatial.transform.Rotation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.spatial.transform.Rotation.rst "Download source file")
-  .pdf

# jax.scipy.spatial.transform.Rotation

## Contents

- [`Rotation`](#jax.scipy.spatial.transform.Rotation)
  - [`Rotation.__init__()`](#jax.scipy.spatial.transform.Rotation.__init__)

# jax.scipy.spatial.transform.Rotation[\#](#jax-scipy-spatial-transform-rotation "Link to this heading")

*class* jax.scipy.spatial.transform.Rotation(*quat*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py#L30-L207)[\#](#jax.scipy.spatial.transform.Rotation "Link to this definition")  
Rotation in 3 dimensions.

JAX implementation of [`scipy.spatial.transform.Rotation`](https://scipy.github.io/devdocs/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation "(in SciPy v1.19.0.dev)").

Examples

Construct an object describing a 90 degree rotation about the z-axis:

    >>> from jax.scipy.spatial.transform import Rotation
    >>> r = Rotation.from_euler('z', 90, degrees=True)

Convert to a rotation vector:

    >>> r.as_rotvec()
    Array([0.       , 0.       , 1.5707964], dtype=float32)

Convert to rotation matrix:

    >>> r.as_matrix()
    Array([[ 0.        , -0.99999994,  0.        ],
           [ 0.99999994,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.99999994]], dtype=float32)

Compose with another rotation:

    >>> r2 = Rotation.from_euler('x', 90, degrees=True)
    >>> r3 = r * r2
    >>> r3.as_matrix()
    Array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]], dtype=float32)

See the scipy [`Rotation`](https://scipy.github.io/devdocs/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation "(in SciPy v1.19.0.dev)") documentation for further examples of manipulating Rotation objects.

Parameters:  
**quat** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

\_\_init\_\_()[\#](#jax.scipy.spatial.transform.Rotation.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.scipy.spatial.transform.Rotation.__init__ "jax.scipy.spatial.transform.Rotation.__init__")() |  |
| `apply`(vectors\[, inverse\]) | Apply this rotation to one or more vectors. |
| `as_euler`(seq\[, degrees\]) | Represent as Euler angles. |
| `as_matrix`() | Represent as rotation matrix. |
| `as_mrp`() | Represent as Modified Rodrigues Parameters (MRPs). |
| `as_quat`(\[canonical, scalar_first\]) | Represent as quaternions. |
| `as_rotvec`(\[degrees\]) | Represent as rotation vectors. |
| `concatenate`(rotations) | Concatenate a sequence of Rotation objects. |
| `count`(value, /) | Return number of occurrences of value. |
| `from_euler`(seq, angles\[, degrees\]) | Initialize from Euler angles. |
| `from_matrix`(matrix) | Initialize from rotation matrix. |
| `from_mrp`(mrp) | Initialize from Modified Rodrigues Parameters (MRPs). |
| `from_quat`(quat) | Initialize from quaternions. |
| `from_rotvec`(rotvec\[, degrees\]) | Initialize from rotation vectors. |
| `identity`(\[num, dtype\]) | Get identity rotation(s). |
| `index`(value\[, start, stop\]) | Return first index of value. |
| `inv`() | Invert this rotation. |
| `magnitude`() | Get the magnitude(s) of the rotation(s). |
| `mean`(\[weights\]) | Get the mean of the rotations. |
| `random`(random_key\[, num\]) | Generate uniformly distributed rotations. |

Attributes

|          |                                                     |
|----------|-----------------------------------------------------|
| `quat`   | Alias for field number 0                            |
| `single` | Whether this instance represents a single rotation. |

[](jax.scipy.signal.welch.html "previous page")

previous

jax.scipy.signal.welch

[](jax.scipy.spatial.transform.Slerp.html "next page")

next

jax.scipy.spatial.transform.Slerp

Contents

- [`Rotation`](#jax.scipy.spatial.transform.Rotation)
  - [`Rotation.__init__()`](#jax.scipy.spatial.transform.Rotation.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
