- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.save

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.save.rst "Download source file")
-  .pdf

# jax.numpy.save

## Contents

- [`save()`](#jax.numpy.save)

# jax.numpy.save[\#](#jax-numpy-save "Link to this heading")

jax.numpy.save(*file*, *arr*, *allow_pickle=True*)[\#](#jax.numpy.save "Link to this definition")  
Save an array to a binary file in NumPy `.npy` format.

Parameters:  
- **file** (*file,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*, or* [*pathlib.Path*](https://docs.python.org/3/library/pathlib.html#pathlib.Path "(in Python v3.14)")) – File or filename to which the data is saved. If file is a file-object, then the filename is unchanged. If file is a string or Path, a `.npy` extension will be appended to the filename if it does not already have one.

- **arr** (*array_like*) – Array data to be saved.

- **allow_pickle** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* *optional*) – Allow saving object arrays using Python pickles. Reasons for disallowing pickles include security (loading pickled data can execute arbitrary code) and portability (pickled objects may not be loadable on different Python installations, for example if the stored objects require libraries that are not available, and not all pickled data is compatible between different versions of Python). Default: True

See also

[`savez`](jax.numpy.savez.html#jax.numpy.savez "jax.numpy.savez")  
Save several arrays into a `.npz` archive

`savetxt`, [`load`](jax.numpy.load.html#jax.numpy.load "jax.numpy.load")

Notes

For a description of the `.npy` format, see [`numpy.lib.format`](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format "(in NumPy v2.4)").

Any data saved to the file is appended to the end of the file.

Examples

    >>> import numpy as np

    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    >>> with open('test.npy', 'wb') as f:
    ...     np.save(f, np.array([1, 2]))
    ...     np.save(f, np.array([1, 3]))
    >>> with open('test.npy', 'rb') as f:
    ...     a = np.load(f)
    ...     b = np.load(f)
    >>> print(a, b)
    # [1 2] [1 3]

[](jax.numpy.s_.html "previous page")

previous

jax.numpy.s\_

[](jax.numpy.savez.html "next page")

next

jax.numpy.savez

Contents

- [`save()`](#jax.numpy.save)

By The JAX authors

© Copyright 2024, The JAX Authors.\
