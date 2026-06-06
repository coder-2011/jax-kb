- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- Foreign function interface (FFI)

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/ffi.md "Download source file")
-  .pdf

# Foreign function interface (FFI)

## Contents

- [A simple example](#a-simple-example)
- [Backend code](#backend-code)
  - [C++ interface](#c-interface)
  - [Building and registering an FFI handler](#building-and-registering-an-ffi-handler)
- [Frontend code](#frontend-code)
  - [Batching with `vmap`](#batching-with-vmap)
  - [Differentiation](#differentiation)
- [FFI calls on a GPU](#ffi-calls-on-a-gpu)
  - [Supporting multiple platforms](#supporting-multiple-platforms)
- [Sharding](#sharding)
  - [Using `shard_map`](#using-shard-map)
  - [Using `custom`` ``partitioning`](#using-custom-partitioning)
- [Advanced topics](#advanced-topics)

# Foreign function interface (FFI)[\#](#foreign-function-interface-ffi "Link to this heading")

*This tutorial requires JAX v0.4.31 or newer.*

While a wide range of numerical operations can be easily and efficiently implemented using JAX’s built in `jax.numpy` and `jax.lax` interfaces, it can sometimes be useful to explicitly call out to external compiled libraries via a “foreign function interface” (FFI). This can be particularly useful when particular operations have been previously implemented in an optimized C or CUDA library, and it would be non-trivial to reimplement these computations directly using JAX, but it can also be useful for optimizing runtime or memory performance of JAX programs. That being said, the FFI should typically be considered a last resort option because the XLA compiler that sits in the backend, or the Pallas kernel language, which provides lower level control, typically produce performant code with a lower development and maintenance cost.

One point that should be taken into account when considering use of the FFI is that *JAX doesn’t automatically know how to differentiate through foreign functions*. This means that if you want to use JAX’s autodifferentiation capabilities alongside a foreign function, you’ll also need to provide an implementation of the relevant differentiation rules. We will discuss some possible approaches below, but it is important to call this limitation out right from the start!

JAX’s FFI support is provided in two parts:

1.  A header-only C++ library from XLA which is packaged as part of JAX as of v0.4.29 or available from the [openxla/xla](https://github.com/openxla/xla) project, and

2.  A Python front end, available in the `jax.ffi` submodule.

In this tutorial we demonstrate the use of both of these components using a simple example, and then go on to discuss some lower-level extensions for more complicated use cases. We start by presenting the FFI on CPU, and discuss generalizations to GPU or multi-device environments below.

The end-to-end code for this example and some other more advanced use cases can be found in the JAX FFI examples project on GitHub at [`examples/ffi` in the JAX repository](https://github.com/jax-ml/jax/tree/main/examples/ffi).

Because we demonstrate how FFI calls can be sharded at the end of this tutorial, let’s first set up our environment to be treated by JAX as having multiple CPUs:

    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

## A simple example[\#](#a-simple-example "Link to this heading")

To demonstrate the use of the FFI interface, we will implement a simple “root-mean-square (RMS)” normalization function. RMS normalization takes an array \\x\\ with shape \\(N,)\\ and returns

\\ y_n = \frac{x_n}{\sqrt{\frac{1}{N}\sum\_{n=1}^N {x_n}^2 + \epsilon}} \\

where \\\epsilon\\ is a tuning parameter used for numerical stability.

This is a somewhat silly example, because it can be easily implemented using JAX as follows:

    import jax
    import jax.numpy as jnp


    def rms_norm_ref(x, eps=1e-5):
      scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
      return x / scale

But, it’s just non-trivial enough to be useful for demonstrating some key details of the FFI, while still being straightforward to understand. We will use this reference implementation to test our FFI version below.

## Backend code[\#](#backend-code "Link to this heading")

To begin with, we need an implementation of RMS normalization in C++ that we will expose using the FFI. This isn’t meant to be particularly performant, but you could imagine that if you had some new better implementation of RMS normalization in a C++ library, it might have an interface like the following. So, here’s a simple implementation of RMS normalization in C++:

    #include <cmath>
    #include <cstdint>

    float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y) {
      float sm = 0.0f;
      for (int64_t n = 0; n < size; ++n) {
        sm += x[n] * x[n];
      }
      float scale = 1.0f / std::sqrt(sm / float(size) + eps);
      for (int64_t n = 0; n < size; ++n) {
        y[n] = x[n] * scale;
      }
      return scale;
    }

and, for our example, this is the function that we want to expose to JAX via the FFI.

### C++ interface[\#](#c-interface "Link to this heading")

To expose our library function to JAX and XLA, we need to write a thin wrapper using the APIs provided by the header-only library in the [`xla/ffi/api`](https://github.com/openxla/xla/tree/main/xla/ffi/api) directory of the [XLA project](https://github.com/openxla/xla). For more information about this interface, take a look at [the XLA custom call documentation](https://openxla.org/xla/custom_call). The full source listing can be downloaded [here](https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/rms_norm.cc), but the key implementation details are reproduced here:

    #include <functional>
    #include <numeric>
    #include <utility>

    #include "xla/ffi/api/c_api.h"
    #include "xla/ffi/api/ffi.h"

    namespace ffi = xla::ffi;

    // A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
    // In this example, we treat all leading dimensions as batch dimensions, so this
    // function returns the total number of elements in the buffer, and the size of
    // the last dimension.
    template <ffi::DataType T>
    std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
      auto dims = buffer.dimensions();
      if (dims.size() == 0) {
        return std::make_pair(0, 0);
      }
      return std::make_pair(buffer.element_count(), dims.back());
    }

    // A wrapper function providing the interface between the XLA FFI call and our
    // library function `ComputeRmsNorm` above. This function handles the batch
    // dimensions by calling `ComputeRmsNorm` within a loop.
    ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                           ffi::ResultBuffer<ffi::F32> y) {
      auto [totalSize, lastDim] = GetDims(x);
      if (lastDim == 0) {
        return ffi::Error::InvalidArgument("RmsNorm input must be an array");
      }
      for (int64_t n = 0; n < totalSize; n += lastDim) {
        ComputeRmsNorm(eps, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
      }
      return ffi::Error::Success();
    }

    // Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
    // this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
    // macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
    XLA_FFI_DEFINE_HANDLER_SYMBOL(
        RmsNorm, RmsNormImpl,
        ffi::Ffi::Bind()
            .Attr<float>("eps")
            .Arg<ffi::Buffer<ffi::F32>>()  // x
            .Ret<ffi::Buffer<ffi::F32>>()  // y
    );

Starting at the bottom, we’re using the XLA-provided macro `XLA_FFI_DEFINE_HANDLER_SYMBOL` to generate some boilerplate which will expand into a function called `RmsNorm` with the appropriate signature. But, the important stuff here is all in the call to `ffi::Ffi::Bind()`, where we define the input and output types, and the types of any parameters.

Then, in `RmsNormImpl`, we accept `ffi::Buffer` arguments which include information about the buffer shape, and pointers to the underlying data. In this implementation, we treat all leading dimensions of the buffer as batch dimensions, and perform RMS normalization over the last axis. `GetDims` is a helper function providing support for this batching behavior. We discuss this batching behavior in more detail [below](#ffi-call-vmap), but the general idea is that it can be useful to transparently handle batching in the left-most dimensions of the input arguments. In this case, we treat all but the last axis as batch dimensions, but other foreign functions may require a different number of non-batch dimensions.

### Building and registering an FFI handler[\#](#building-and-registering-an-ffi-handler "Link to this heading")

Now that we have our minimal FFI wrapper implemented, we need to expose this function (`RmsNorm`) to Python. In this tutorial, we compile `RmsNorm` into a shared library and load it using [ctypes](https://docs.python.org/3/library/ctypes.html), but another common pattern is to use [nanobind](https://nanobind.readthedocs.io/) or [pybind11](https://pybind11.readthedocs.io/) as discussed below.

To compile the shared library, we’re using CMake here, but you should be able to use your favorite build system without too much trouble.

    !cmake -DCMAKE_BUILD_TYPE=Release -B ffi/_build ffi
    !cmake --build ffi/_build
    !cmake --install ffi/_build

Show code cell output

Hide code cell output

    -- The CXX compiler identification is GNU 13.2.0
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /usr/bin/c++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Found Python: /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/bin/python3.12 (found suitable version "3.12.12", minimum required is "3.8") found components: Interpreter Development.Module
    -- XLA include directory: /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include
    -- Configuring done (4.5s)
    -- Generating done (0.0s)
    -- Build files have been written to: /home/docs/checkouts/readthedocs.org/user_builds/jax/checkouts/latest/docs/ffi/_build
    [ 50%] Building CXX object CMakeFiles/rms_norm.dir/rms_norm.cc.o
    In file included from /home/docs/checkouts/readthedocs.org/user_builds/jax/checkouts/latest/docs/ffi/rms_norm.cc:24:
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/ffi.h:772:68: warning: ‘always_inline’ function might not be inlinable []8;;https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wattributes-Wattributes]8;;]
      772 | _ATTRIBUTE_ALWAYS_INLINE std::optional<Buffer<dtype, rank>> DecodeBuffer(
          |                                                             ^~~~~~~~~~~~
    In file included from /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/ffi.h:49:
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h: In function ‘std::ostream& operator<<(std::ostream&, XLA_FFI_ExecutionStage)’:
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h:193:1: warning: control reaches end of non-void function []8;;https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wreturn-type-Wreturn-type]8;;]
      193 | }
          | ^
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h: In function ‘std::ostream& operator<<(std::ostream&, XLA_FFI_AttrType)’:
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h:179:1: warning: control reaches end of non-void function []8;;https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wreturn-type-Wreturn-type]8;;]
      179 | }
          | ^
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h: In function ‘std::ostream& operator<<(std::ostream&, XLA_FFI_DataType)’:
    /home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/api.h:166:1: warning: control reaches end of non-void function []8;;https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wreturn-type-Wreturn-type]8;;]
      166 | }
          | ^
    [100%] Linking CXX shared library librms_norm.so
    [100%] Built target rms_norm
    -- Install configuration: "Release"
    -- Installing: /home/docs/checkouts/readthedocs.org/user_builds/jax/checkouts/latest/docs/ffi/librms_norm.so

With this compiled library in hand, we now need to register this handler with XLA via the [`register_ffi_target()`](_autosummary/jax.ffi.register_ffi_target.html#jax.ffi.register_ffi_target "jax.ffi.register_ffi_target") function. This function expects our handler (a function pointer to the C++ function `RmsNorm`) to be wrapped in a [`PyCapsule`](https://docs.python.org/3/c-api/capsule.html). JAX provides a helper function [`pycapsule()`](_autosummary/jax.ffi.pycapsule.html#jax.ffi.pycapsule "jax.ffi.pycapsule") to help with this:

    import ctypes
    from pathlib import Path

    path = next(Path("ffi").glob("librms_norm*"))
    rms_norm_lib = ctypes.cdll.LoadLibrary(path)
    jax.ffi.register_ffi_target(
        "rms_norm", jax.ffi.pycapsule(rms_norm_lib.RmsNorm), platform="cpu")

Tip

If you’re familiar with the legacy “custom call” API, it’s worth noting that you can also use [`register_ffi_target()`](_autosummary/jax.ffi.register_ffi_target.html#jax.ffi.register_ffi_target "jax.ffi.register_ffi_target") to register a custom call target by manually specifying the keyword argument `api_version=0`. The default `api_version` for [`register_ffi_target()`](_autosummary/jax.ffi.register_ffi_target.html#jax.ffi.register_ffi_target "jax.ffi.register_ffi_target") is `1`, the new “typed” FFI API that we’re using here.

**An alternative approach**: A common alternative pattern for exposing handlers to Python is to use [nanobind](https://nanobind.readthedocs.io/) or [pybind11](https://pybind11.readthedocs.io/) to define a tiny Python extension which can be imported. For our example here, the nanobind code would be:

    #include <type_traits>

    #include "nanobind/nanobind.h"
    #include "xla/ffi/api/c_api.h"

    namespace nb = nanobind;

    template <typename T>
    nb::capsule EncapsulateFfiCall(T *fn) {
      // This check is optional, but it can be helpful for avoiding invalid handlers.
      static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                    "Encapsulated function must be and XLA FFI handler");
      return nb::capsule(reinterpret_cast<void *>(fn));
    }

    NB_MODULE(rms_norm, m) {
      m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
    }

Then, in Python we can register this handler using:

    # Assuming that we compiled a nanobind extension called `rms_norm`:
    import rms_norm as rms_norm_lib

    jax.ffi.register_ffi_target("rms_norm", rms_norm_lib.rms_norm(), platform="cpu")

## Frontend code[\#](#frontend-code "Link to this heading")

Now that we have registered our FFI handler, it is straightforward to call our C++ library from JAX using the [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") function:

    import numpy as np


    def rms_norm(x, eps=1e-5):
      # We only implemented the `float32` version of this function, so we start by
      # checking the dtype. This check isn't strictly necessary because type
      # checking is also performed by the FFI when decoding input and output
      # buffers, but it can be useful to check types in Python to raise more
      # informative errors.
      if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by rms_norm")

      call = jax.ffi.ffi_call(
        # The target name must be the same string as we used to register the target
        # above in `register_custom_call_target`
        "rms_norm",

        # In this case, the output of our FFI function is just a single array with
        # the same shape and dtype as the input. We discuss a case with a more
        # interesting output type below.
        jax.ShapeDtypeStruct.like(x),

        # The `vmap_method` parameter controls this function's behavior under `vmap`
        # as discussed below.
        vmap_method="broadcast_all",
      )

      # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
      # the attribute `eps`. Our FFI function expects this to have the C++ `float`
      # type (which corresponds to numpy's `float32` type), and it must be a
      # static parameter (i.e. not a JAX array).
      return call(x, eps=np.float32(eps))


    # Test that this gives the same result as our reference implementation
    x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
    np.testing.assert_allclose(rms_norm(x), rms_norm_ref(x), rtol=1e-5)

This code cell includes a lot of inline comments which should explain most of what is happening here, but there are a few points that are worth explicitly highlighting. Most of the heavy lifting here is done by the [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") function, which tells JAX how to call the foreign function for a particular set of inputs. It’s important to note that the first argument to [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") must be a string that matches the target name that we used when calling `register_custom_call_target` above.

Any attributes (defined using `Attr` in the C++ wrapper above) should be passed as keyword arguments to [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call"). Note that we explicitly cast `eps` to `np.float32` because our FFI library expects a C `float`, and we can’t use `jax.numpy` here, because these parameters must be static arguments.

The `vmap_method` argument to [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") defines how this FFI call interacts with [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") as described next.

Tip

If you are familiar with the earlier “custom call” interface, you might be surprised that we’re not passing the problem dimensions as parameters (batch size, etc.) to [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call"). In this earlier API, the backend had no mechanism for receiving metadata about the input arrays, but since the FFI includes dimension information with the `Buffer` objects, we no longer need to compute this using Python when lowering. One major perk of this change is [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") can support some simple [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") semantics out of the box, as discussed below.

### Batching with `vmap`[\#](#batching-with-vmap "Link to this heading")

[`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") supports some simple [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") semantics out of the box using the `vmap_method` parameter. The docs for [`pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") provide more details about the `vmap_method` parameter, and the same behavior applies to [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call").

The simplest `vmap_method` is `"sequential"`. In this case, when `vmap`ped, an `ffi_call` will be rewritten as a [`scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan") with the `ffi_call` in the body. This implementation is general purpose, but it doesn’t parallelize very well. Many FFI calls provide more efficient batching behavior and, in some simple cases, the `"expand_dims"` or `"broadcast_all"` methods can be used to expose a better implementation.

In this case, since we only have one input argument, `"expand_dims"` and `"broadcast_all"` actually have the same behavior. The specific assumption required to use these methods is that the foreign function knows how to handle batch dimensions. Another way of saying this is that the result of calling `ffi_call` on the batched inputs is assumed to be equal to stacking the repeated application of `ffi_call` to each element in the batched input, roughly:

    ffi_call(xs) == jnp.stack([ffi_call(x) for x in xs])

Tip

Note that things get a bit more complicated when we have multiple input arguments. For simplicity, we will use the `"broadcast_all"` throughout this tutorial, which guarantees that all inputs will be broadcasted to have the same batch dimensions, but it would also be possible to implement a foreign function to handle the `"expand_dims"` method. The documentation for [`pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") includes some examples of this

Our implementation of `rms_norm` has the appropriate semantics, and it supports `vmap` with `vmap_method="broadcast_all"` out of the box:

    np.testing.assert_allclose(jax.vmap(rms_norm)(x), jax.vmap(rms_norm_ref)(x), rtol=1e-5)

We can inspect the [jaxpr](jaxpr.html#jax-internals-jaxpr) of the [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") of `rms_norm` to confirm that it isn’t being rewritten using [`scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan"):

    jax.make_jaxpr(jax.vmap(rms_norm))(x)

    { lambda ; a:f32[8,4]. let
        b:f32[8,4] = ffi_call[
          attributes=(('eps', np.float32(1e-05)),)
          custom_call_api_version=4
          has_side_effect=False
          input_layouts=((1, 0),)
          input_output_aliases=()
          legacy_backend_config=None
          output_layouts=((1, 0),)
          result_avals=(ShapedArray(float32[8,4]),)
          target_name=rms_norm
          vmap_method=broadcast_all
        ] a
      in (b,) }

Using `vmap_method="sequential"`, `vmap`ping a `ffi_call` will fall back on a [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan") with the `ffi_call` in the body:

    def rms_norm_sequential(x, eps=1e-5):
      return jax.ffi.ffi_call(
        "rms_norm",
        jax.ShapeDtypeStruct.like(x),
        vmap_method="sequential",
      )(x, eps=np.float32(eps))


    jax.make_jaxpr(jax.vmap(rms_norm_sequential))(x)

    { lambda ; a:f32[8,4]. let
        b:f32[8,4] = scan[
          jaxpr={ lambda ; c:f32[4]. let
              d:f32[4] = ffi_call[
                attributes=(('eps', np.float32(1e-05)),)
                custom_call_api_version=4
                has_side_effect=False
                input_layouts=((0,),)
                input_output_aliases=()
                legacy_backend_config=None
                output_layouts=((0,),)
                result_avals=(ShapedArray(float32[4]),)
                target_name=rms_norm
                vmap_method=sequential
              ] c
            in (d,) }
          length=8
          num_carry=0
          num_consts=0
          reverse=False
          unroll=1
        ] a
      in (b,) }

If your foreign function provides an efficient batching rule that isn’t supported by this simple `vmap_method` parameter, it might also be possible to define more flexible custom `vmap` rules using the experimental `custom_vmap` interface, but it’s worth also opening an issue describing your use case on [the JAX issue tracker](https://github.com/jax-ml/jax/issues).

### Differentiation[\#](#differentiation "Link to this heading")

Unlike with batching, [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") doesn’t provide any default support for automatic differentiation (AD) of foreign functions. As far as JAX is concerned, the foreign function is a black box that can’t be inspected to determine the appropriate behavior when differentiated. Therefore, it is the [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") user’s responsibility to define a custom derivative rule.

More details about custom derivative rules can be found in the [custom derivatives tutorial](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html), but the most common pattern used for implementing differentiation for foreign functions is to define a [`custom_vjp()`](_autosummary/jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp") which itself calls a foreign function. In this case, we actually define two new FFI calls:

1.  `rms_norm_fwd` returns two outputs: (a) the “primal” result, and (b) the “residuals” which are used in the backwards pass.

2.  `rms_norm_bwd` takes the residuals and the output co-tangents, and returns the input co-tangents.

We won’t get into the details of the RMS normalization backwards pass, but take a look at the [C++ source code](https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/rms_norm.cc) to see how these functions are implemented on the back end. The main point to emphasize here is that the “residual” computed has a different shape than the primal output, therefore, in the [`ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") to `res_norm_fwd`, the output type has two elements with different shapes.

This custom derivative rule can be wired in as follows:

    jax.ffi.register_ffi_target(
      "rms_norm_fwd", jax.ffi.pycapsule(rms_norm_lib.RmsNormFwd), platform="cpu"
    )
    jax.ffi.register_ffi_target(
      "rms_norm_bwd", jax.ffi.pycapsule(rms_norm_lib.RmsNormBwd), platform="cpu"
    )


    def rms_norm_fwd(x, eps=1e-5):
      y, res = jax.ffi.ffi_call(
        "rms_norm_fwd",
        (
          jax.ShapeDtypeStruct.like(x),
          jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),
        ),
        vmap_method="broadcast_all",
      )(x, eps=np.float32(eps))
      return y, (res, x)


    def rms_norm_bwd(eps, res, ct):
      del eps
      res, x = res
      assert res.shape == ct.shape[:-1]
      assert x.shape == ct.shape
      return (
        jax.ffi.ffi_call(
          "rms_norm_bwd",
          jax.ShapeDtypeStruct.like(ct),
          vmap_method="broadcast_all",
        )(res, x, ct),
      )


    rms_norm = jax.custom_vjp(rms_norm, nondiff_argnums=(1,))
    rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)

    # Check that this gives the right answer when compared to the reference version
    ct_y = jnp.ones_like(x)
    np.testing.assert_allclose(
      jax.vjp(rms_norm, x)[1](ct_y), jax.vjp(rms_norm_ref, x)[1](ct_y), rtol=1e-5
    )

At this point, we can use our new `rms_norm` function transparently for many JAX applications, and it will transform appropriately under the standard JAX function transformations like [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") and [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"). One thing that this example doesn’t support is forward-mode AD ([`jax.jvp()`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp"), for example) since [`custom_vjp()`](_autosummary/jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp") is restricted to reverse-mode. JAX doesn’t currently expose a public API for simultaneously customizing both forward-mode and reverse-mode AD, but such an API is on the roadmap, so please [open an issue](https://github.com/jax-ml/jax/issues) describing you use case if you hit this limitation in practice.

One other JAX feature that this example doesn’t support is higher-order AD. It would be possible to work around this by wrapping the `res_norm_bwd` function above in a [`jax.custom_jvp()`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp") or [`jax.custom_vjp()`](_autosummary/jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp") decorator, but we won’t go into the details of that advanced use case here.

## FFI calls on a GPU[\#](#ffi-calls-on-a-gpu "Link to this heading")

So far, we have been interfacing only with foreign functions running on the CPU, but JAX’s FFI also supports calls to GPU code. Since this documentation page is automatically generated on a machine without access to a GPU, we can’t execute any GPU-specific examples here, but we will go over the key points.

When defining our FFI wrapper for CPU, the function signature that we used was:

    ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                           ffi::ResultBuffer<ffi::F32> y)

To update this to interface with a CUDA kernel, this signature becomes:

    ffi::Error RmsNormImpl(cudaStream_t stream, float eps,
                           ffi::Buffer<ffi::F32> x,
                           ffi::ResultBuffer<ffi::F32> y)

And the handler definition is updated to include a `Ctx` in its binding:

    XLA_FFI_DEFINE_HANDLER(
        RmsNorm, RmsNormImpl,
        ffi::Ffi::Bind()
            .Ctx<ffi::PlatformStream<cudaStream_t>>()
            .Attr<float>("eps")
            .Arg<ffi::Buffer<ffi::F32>>()  // x
            .Ret<ffi::Buffer<ffi::F32>>()  // y
    );

Then, the `RmsNormImpl` can use the CUDA stream to launch CUDA kernels.

On the front end, the registration code would be updated to specify the appropriate platform:

    jax.ffi.register_ffi_target(
      "rms_norm_cuda", rms_norm_lib_cuda.rms_norm(), platform="CUDA"
    )

### Supporting multiple platforms[\#](#supporting-multiple-platforms "Link to this heading")

To support running our `rms_norm` function on both GPU and CPU, we can combine our implementation above with the [`jax.lax.platform_dependent()`](_autosummary/jax.lax.platform_dependent.html#jax.lax.platform_dependent "jax.lax.platform_dependent") function:

    def rms_norm_cross_platform(x, eps=1e-5):
      assert x.dtype == jnp.float32
      out_type = jax.ShapeDtypeStruct.like(x)

      def impl(target_name):
        return lambda x: jax.ffi.ffi_call(
          target_name,
          out_type,
          vmap_method="broadcast_all",
        )(x, eps=np.float32(eps))

      return jax.lax.platform_dependent(x, cpu=impl("rms_norm"), cuda=impl("rms_norm_cuda"))


    np.testing.assert_allclose(rms_norm_cross_platform(x), rms_norm_ref(x), rtol=1e-5)

This version of the function will call the appropriate FFI target depending on the runtime platform.

As an aside, it may be interesting to note that while the jaxpr and lowered HLO both contain a reference to both FFI targets:

    jax.make_jaxpr(rms_norm_cross_platform)(x)

    { lambda ; a:f32[8,4]. let
        b:i32[] = platform_index[platforms=(('cpu',), ('cuda',))] 
        c:f32[8,4] = cond[
          branches=(
            { lambda ; d:f32[8,4]. let
                e:f32[8,4] = ffi_call[
                  attributes=(('eps', np.float32(1e-05)),)
                  custom_call_api_version=4
                  has_side_effect=False
                  input_layouts=((1, 0),)
                  input_output_aliases=()
                  legacy_backend_config=None
                  output_layouts=((1, 0),)
                  result_avals=(ShapedArray(float32[8,4]),)
                  target_name=rms_norm
                  vmap_method=broadcast_all
                ] d
              in (e,) }
            { lambda ; f:f32[8,4]. let
                g:f32[8,4] = ffi_call[
                  attributes=(('eps', np.float32(1e-05)),)
                  custom_call_api_version=4
                  has_side_effect=False
                  input_layouts=((1, 0),)
                  input_output_aliases=()
                  legacy_backend_config=None
                  output_layouts=((1, 0),)
                  result_avals=(ShapedArray(float32[8,4]),)
                  target_name=rms_norm_cuda
                  vmap_method=broadcast_all
                ] f
              in (g,) }
          )
          branches_platforms=(('cpu',), ('cuda',))
        ] b a
      in (c,) }

    print(jax.jit(rms_norm_cross_platform).lower(x).as_text().strip())

    module @jit_rms_norm_cross_platform attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
      func.func public @main(%arg0: tensor<8x4xf32>) -> (tensor<8x4xf32> {jax.result_info = "result"}) {
        %c = stablehlo.constant dense<0> : tensor<i32>
        %0 = "stablehlo.case"(%c) ({
          %c_0 = stablehlo.constant dense<0> : tensor<i32>
          stablehlo.return %c_0 : tensor<i32>
        }, {
          %c_0 = stablehlo.constant dense<0> : tensor<i32>
          stablehlo.return %c_0 : tensor<i32>
        }) : (tensor<i32>) -> tensor<i32>
        %1 = "stablehlo.case"(%0) ({
          %2 = stablehlo.custom_call @rms_norm(%arg0) {backend_config = "", mhlo.backend_config = {eps = 9.99999974E-6 : f32}, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<8x4xf32>) -> tensor<8x4xf32>
          stablehlo.return %2 : tensor<8x4xf32>
        }) : (tensor<i32>) -> tensor<8x4xf32>
        return %1 : tensor<8x4xf32>
      }
    }

by the time the function is compiled, the appropriate FFI has been selected:

    print(jax.jit(rms_norm_cross_platform).lower(x).as_text(dialect="hlo").strip())

    HloModule jit_rms_norm_cross_platform, entry_computation_layout={(f32[8,4]{1,0})->f32[8,4]{1,0}}

    ENTRY main.1 {
      x.1 = f32[8,4]{1,0} parameter(0)
      ROOT ffi_call.1 = f32[8,4]{1,0} custom-call(x.1), custom_call_target="rms_norm", operand_layout_constraints={f32[8,4]{1,0}}, api_version=API_VERSION_TYPED_FFI, backend_config={eps = 9.99999974E-6 : f32}
    }

and there will be no runtime overhead to using [`jax.lax.platform_dependent()`](_autosummary/jax.lax.platform_dependent.html#jax.lax.platform_dependent "jax.lax.platform_dependent"), and the compiled program won’t include any references to unavailable FFI targets.

## Sharding[\#](#sharding "Link to this heading")

Most large scale users of JAX use its APIs for distributed computation across multiple devices. As discussed in [Distributed arrays and automatic parallelization](parallel.html#parallel), parallelism in JAX is controlled by sharding data across devices, and most JAX operations can be used within any of the supported parallel programming paradigms (from automatic to fully manual). But, the story is a little bit more complicated for FFI calls. Since the internals of an FFI call are opaque to both JAX and XLA, FFI calls won’t typically show optimal (or even good) performance when the data are sharded.

Before getting into the FFI details, let’s consider the behavior of our pure-JAX reference implementation of RMS normalization (the `rms_norm_ref` function defined at the top of this document) with a sharded input. As discussed above, our implementation treats all leading axes of the input as *batch* dimensions, and the normalization is performed along the last axis. This means that if the data are sharded along any batch dimensions, but replicated on the last dimension, no communication is required. This can be seen by sharding our 2-dimensional test data from above along its first dimension and checking the compiled HLO for operations like `all-gather`, `all-reduce`, etc.:

    from jax.sharding import PartitionSpec as P

    assert len(jax.devices()) == 4  # Set using the XLA_FLAGS environment variable
    mesh = jax.make_mesh((4,), ("x",))

    batch_shd = jax.NamedSharding(mesh, P("x", None))
    x_batch_shd = jax.device_put(x, batch_shd)
    hlo_batch = jax.jit(rms_norm_ref, out_shardings=batch_shd).lower(x_batch_shd).compile().as_text()
    assert "all-" not in hlo_batch

However, if the data are sharded along the last axis, communication (in this case an `all-reduce`) is required to compute the sum in the normalization:

    data_shd = jax.NamedSharding(mesh, P(None, "x"))
    x_data_shd = jax.device_put(x, data_shd)
    hlo_data = jax.jit(rms_norm_ref, out_shardings=data_shd).lower(x_data_shd).compile().as_text()
    assert "all-reduce" in hlo_data

Now, if we try to naively use our FFI version of the same model, it runs fine and gets the right answer:

    output = jax.jit(rms_norm, out_shardings=batch_shd)(x_batch_shd)
    np.testing.assert_allclose(output, rms_norm_ref(x), rtol=1e-5)

But, if you look at the compiled HLO (omitting a helper functions for clarity), you’ll see that

1.  the data are first fully replicated onto each device via an `all-gather` operation,

2.  the FFI call is executed on the full dataset on each device, and

3.  the output is sliced to discard the unused portions.

    hlo = jax.jit(rms_norm, out_shardings=batch_shd).lower(x_batch_shd).compile().as_text().strip()
    print(hlo.split("\n\n")[-1])

    ENTRY %main.0_spmd (param: f32[2,4]) -> f32[2,4] {
      %partition-id = u32[] partition-id(), metadata={op_name="jit(rms_norm)/ffi_call" stack_frame_id=20}
      %param = f32[2,4]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}, metadata={op_name="x"}
      %all-gather = f32[8,4]{1,0} all-gather(%param), channel_id=1, replica_groups=mesh['axis_0'=4,'axis_1'=1] {'axis_0'}, dimensions={0}, use_global_device_ids=true, metadata={op_name="jit(rms_norm)/ffi_call" stack_frame_id=20}
      %ffi_call.0 = f32[8,4]{1,0} custom-call(%all-gather), custom_call_target="rms_norm", operand_layout_constraints={f32[8,4]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(rms_norm)/ffi_call" stack_frame_id=20}, backend_config={eps = 9.99999974E-6 : f32}
      ROOT %multiply_dynamic-slice_fusion = f32[2,4]{1,0} fusion(%ffi_call.0, %partition-id), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(rms_norm)/ffi_call" stack_frame_id=20}
    }

This clearly (to us!) isn’t the optimal partitioning of this function, but it’s the best that JAX/XLA can do with the information given.

To generate better partitioning logic, we can use [`shard_map()`](_autosummary/jax.shard_map.html#jax.shard_map "jax.shard_map") or [`custom_partitioning()`](jax.experimental.custom_partitioning.html#jax.experimental.custom_partitioning.custom_partitioning "jax.experimental.custom_partitioning.custom_partitioning"), and we discuss both options here. That being said, it’s not straightforward to generate *optimal* partitioning for all inputs, because sometimes this would require algorithmic changes. Specifically, let’s add support for “batch partitioning”, which handles the case where the data are sharded on batch dimensions, but sharding on the last dimension will always require in re-sharding.

### Using `shard_map`[\#](#using-shard-map "Link to this heading")

If you are using manual sharding control via [`shard_map()`](_autosummary/jax.shard_map.html#jax.shard_map "jax.shard_map"), any FFI calls in your program should already partition appropriately:

    from functools import partial

    @partial(jax.shard_map, mesh=mesh, in_specs=P("x", None), out_specs=P("x", None))
    def rms_norm_shmap(x):
      return rms_norm(x)

    np.testing.assert_allclose(rms_norm_shmap(x_batch_shd), rms_norm_ref(x), rtol=1e-5)
    print(jax.jit(rms_norm_shmap, out_shardings=batch_shd).lower(x_batch_shd).compile().as_text().strip())

    HloModule jit_rms_norm_shmap, is_scheduled=true, entry_computation_layout={(f32[2,4]{1,0})->f32[2,4]{1,0}}, num_partitions=4

    FileNames
    1 "<frozen runpy>"
    2 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel_launcher.py"
    3 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/traitlets/config/application.py"
    4 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/kernelapp.py"
    5 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/tornado/platform/asyncio.py"
    6 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/kernelbase.py"
    7 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/ipkernel.py"
    8 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/zmqshell.py"
    9 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/IPython/core/interactiveshell.py"
    10 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/IPython/core/async_helpers.py"
    11 "/tmp/ipykernel_1811/559451466.py"
    12 "/tmp/ipykernel_1811/1097505059.py"

    FunctionNames
    1 "_run_module_as_main"
    2 "_run_code"
    3 "<module>"
    4 "Application.launch_instance"
    5 "IPKernelApp.start"
    6 "BaseAsyncIOLoop.start"
    7 "Kernel.shell_main"
    8 "Kernel.dispatch_shell"
    9 "IPythonKernel.execute_request"
    10 "Kernel.execute_request"
    11 "IPythonKernel.do_execute"
    12 "ZMQInteractiveShell.run_cell"
    13 "InteractiveShell.run_cell"
    14 "InteractiveShell._run_cell"
    15 "_pseudo_sync_runner"
    16 "InteractiveShell.run_cell_async"
    17 "InteractiveShell.run_ast_nodes"
    18 "InteractiveShell.run_code"
    19 "rms_norm_shmap"
    20 "rms_norm"

    FileLocations
    1 {file_name_id=1 function_name_id=1 line=198 end_line=198 column=11 end_column=11}
    2 {file_name_id=1 function_name_id=2 line=88 end_line=88 column=4 end_column=4}
    3 {file_name_id=2 function_name_id=3 line=18 end_line=18 column=4 end_column=4}
    4 {file_name_id=3 function_name_id=4 line=1082 end_line=1082 column=8 end_column=8}
    5 {file_name_id=4 function_name_id=5 line=758 end_line=758 column=16 end_column=16}
    6 {file_name_id=5 function_name_id=6 line=211 end_line=211 column=8 end_column=8}
    7 {file_name_id=6 function_name_id=7 line=621 end_line=621 column=12 end_column=12}
    8 {file_name_id=6 function_name_id=8 line=478 end_line=478 column=20 end_column=20}
    9 {file_name_id=7 function_name_id=9 line=372 end_line=372 column=8 end_column=8}
    10 {file_name_id=6 function_name_id=10 line=834 end_line=834 column=28 end_column=28}
    11 {file_name_id=7 function_name_id=11 line=464 end_line=464 column=26 end_column=26}
    12 {file_name_id=8 function_name_id=12 line=663 end_line=663 column=15 end_column=15}
    13 {file_name_id=9 function_name_id=13 line=3170 end_line=3170 column=25 end_column=25}
    14 {file_name_id=9 function_name_id=14 line=3225 end_line=3225 column=21 end_column=21}
    15 {file_name_id=10 function_name_id=15 line=128 end_line=128 column=8 end_column=8}
    16 {file_name_id=9 function_name_id=16 line=3447 end_line=3447 column=29 end_column=29}
    17 {file_name_id=9 function_name_id=17 line=3688 end_line=3688 column=19 end_column=19}
    18 {file_name_id=9 function_name_id=18 line=3748 end_line=3748 column=20 end_column=20}
    19 {file_name_id=11 function_name_id=3 line=8 end_line=8 column=6 end_column=6}
    20 {file_name_id=11 function_name_id=19 line=5 end_line=5 column=9 end_column=9}
    21 {file_name_id=12 function_name_id=20 line=32 end_line=32 column=9 end_column=9}

    StackFrames
    1 {file_location_id=1 parent_frame_id=1}
    2 {file_location_id=2 parent_frame_id=2}
    3 {file_location_id=3 parent_frame_id=3}
    4 {file_location_id=4 parent_frame_id=4}
    5 {file_location_id=5 parent_frame_id=5}
    6 {file_location_id=6 parent_frame_id=6}
    7 {file_location_id=7 parent_frame_id=7}
    8 {file_location_id=8 parent_frame_id=8}
    9 {file_location_id=9 parent_frame_id=9}
    10 {file_location_id=10 parent_frame_id=10}
    11 {file_location_id=11 parent_frame_id=11}
    12 {file_location_id=12 parent_frame_id=12}
    13 {file_location_id=13 parent_frame_id=13}
    14 {file_location_id=14 parent_frame_id=14}
    15 {file_location_id=15 parent_frame_id=15}
    16 {file_location_id=16 parent_frame_id=16}
    17 {file_location_id=17 parent_frame_id=17}
    18 {file_location_id=18 parent_frame_id=18}
    19 {file_location_id=19 parent_frame_id=19}
    20 {file_location_id=20 parent_frame_id=1}
    21 {file_location_id=21 parent_frame_id=21}
    22 {file_location_id=20 parent_frame_id=20}
    23 {file_location_id=21 parent_frame_id=23}


    ENTRY %main.0_spmd (param.1: f32[2,4]) -> f32[2,4] {
      %param.1 = f32[2,4]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}, metadata={op_name="x"}
      ROOT %ffi_call.3 = f32[2,4]{1,0} custom-call(%param.1), custom_call_target="rms_norm", operand_layout_constraints={f32[2,4]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(rms_norm_shmap)/shard_map/ffi_call" stack_frame_id=23}, backend_config={eps = 9.99999974E-6 : f32}
    }

As you can see in this program, if the input and output shardings match the `shard_map` specs, no communication is required and the FFI call is executed on the appropriately sharded subset of the data.

You can also use inputs and outputs with shardings that don’t match the `shard_map` specs, but (unrelated to the FFI) this will require re-sharding, as seen by the `all-to-all` operations in the compiled HLO:

    x_data_shd = jax.device_put(x_data_shd, jax.NamedSharding(mesh, jax.P('x', None)))
    hlo_data_shmap = jax.jit(rms_norm_shmap, out_shardings=data_shd).lower(x_data_shd).compile().as_text()
    assert "all-to-all" in hlo_data_shmap

### Using `custom`` ``partitioning`[\#](#using-custom-partitioning "Link to this heading")

If you can’t use [`shard_map()`](_autosummary/jax.shard_map.html#jax.shard_map "jax.shard_map"), an alternative approach is to use [`custom_partitioning()`](jax.experimental.custom_partitioning.html#jax.experimental.custom_partitioning.custom_partitioning "jax.experimental.custom_partitioning.custom_partitioning"), which supports automatic parallelization via [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"). [`custom_partitioning()`](jax.experimental.custom_partitioning.html#jax.experimental.custom_partitioning.custom_partitioning "jax.experimental.custom_partitioning.custom_partitioning") works by adding Python callbacks into the XLA compiler’s partitioning pass, which allows very flexible logic, but also comes with some rough edges. We won’t go into too much detail on the caveats here, but the main issues that you should be aware of are:

1.  `custom_partitioning` can cause unexpected cache misses when used with the JAX’s [Persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html). This can be mitigated using the `jax_remove_custom_partitioning_ptr_from_cache_key` configuration flag, but that isn’t always appropriate either.

2.  Debugging `custom_partitioning` logic can be tedious because Python errors don’t always get propagated, instead causing your Python process to exit. That being said, any exceptions will show up in the process logs, so you should be able to track them down there.

All that being said, here’s how we can wrap our FFI implementation of `rms_norm` using [`custom_partitioning()`](jax.experimental.custom_partitioning.html#jax.experimental.custom_partitioning.custom_partitioning "jax.experimental.custom_partitioning.custom_partitioning"):

    from jax.experimental.custom_partitioning import custom_partitioning

    @partial(custom_partitioning, static_argnums=(1,))
    def rms_norm_partitioned(x, eps=1e-5):
      return rms_norm(x, eps=eps)

    def replicate_sharding_on_last_dim(mesh, sharding, target_info):
      # Our implementation supports trivial sharding on any batch dimensions, but the data
      # must be replicated on the last (non-batch) dimension.
      rank = len(target_info.shape)
      num_batch_dims = min(len(sharding.spec), rank - 1)

      # The Nones here indicate which dimensions should be replicated.
      names = tuple(sharding.spec[:num_batch_dims]) + (None,) * (rank - num_batch_dims)
      return jax.NamedSharding(mesh, P(*names))

    def rms_norm_infer_sharding_from_operands(eps, mesh, args_info, result_info):
      del eps  # unused
      arg_info, = args_info
      result_sharding = replicate_sharding_on_last_dim(mesh, arg_info.sharding, result_info)

      # In this case, we only have a single output, but the return value from this function
      # must have the same pytree structure as the output from the underlying function
      # (`rms_norm` in this case).
      return result_sharding

    def rms_norm_partition(eps, mesh, args_info, result_info):
      arg_info, = args_info
      arg_sharding = replicate_sharding_on_last_dim(mesh, arg_info.sharding, arg_info)
      result_sharding = replicate_sharding_on_last_dim(mesh, arg_info.sharding, result_info)

      # This is the function that computes the partitioned model on the appropriate subset
      # of the data.
      def partitioned_rms_norm(x):
        return rms_norm(x, eps=eps)

      # Note that the third element of our returned tuple must be the shardings for the
      # _outputs_ and its pytree structure must match the output of `rms_norm`. Similarly,
      # the fourth element must have the same pytree structure as the _inputs_ to
      # `rms_norm`. In this case, there is only one input, but it must be returned within
      # a `tuple` anyways.
      return mesh, partitioned_rms_norm, result_sharding, (arg_sharding,)

    rms_norm_partitioned.def_partition(
        infer_sharding_from_operands=rms_norm_infer_sharding_from_operands,
        partition=rms_norm_partition,
        sharding_rule="... i -> ... j",
    )

    output = jax.jit(rms_norm_partitioned, out_shardings=batch_shd)(x_batch_shd)
    np.testing.assert_allclose(output, rms_norm_ref(x), rtol=1e-5)
    print(jax.jit(rms_norm_partitioned, out_shardings=batch_shd).lower(x_batch_shd).compile().as_text().strip())

    HloModule jit__unnamed_function, is_scheduled=true, entry_computation_layout={(f32[2,4]{1,0})->f32[2,4]{1,0}}, num_partitions=4

    FileNames
    1 "<frozen runpy>"
    2 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel_launcher.py"
    3 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/traitlets/config/application.py"
    4 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/kernelapp.py"
    5 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/tornado/platform/asyncio.py"
    6 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/kernelbase.py"
    7 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/ipkernel.py"
    8 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel/zmqshell.py"
    9 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/IPython/core/interactiveshell.py"
    10 "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/IPython/core/async_helpers.py"
    11 "/tmp/ipykernel_1811/1116868249.py"

    FunctionNames
    1 "_run_module_as_main"
    2 "_run_code"
    3 "<module>"
    4 "Application.launch_instance"
    5 "IPKernelApp.start"
    6 "BaseAsyncIOLoop.start"
    7 "Kernel.shell_main"
    8 "Kernel.dispatch_shell"
    9 "IPythonKernel.execute_request"
    10 "Kernel.execute_request"
    11 "IPythonKernel.do_execute"
    12 "ZMQInteractiveShell.run_cell"
    13 "InteractiveShell.run_cell"
    14 "InteractiveShell._run_cell"
    15 "_pseudo_sync_runner"
    16 "InteractiveShell.run_cell_async"
    17 "InteractiveShell.run_ast_nodes"
    18 "InteractiveShell.run_code"

    FileLocations
    1 {file_name_id=1 function_name_id=1 line=198 end_line=198 column=11 end_column=11}
    2 {file_name_id=1 function_name_id=2 line=88 end_line=88 column=4 end_column=4}
    3 {file_name_id=2 function_name_id=3 line=18 end_line=18 column=4 end_column=4}
    4 {file_name_id=3 function_name_id=4 line=1082 end_line=1082 column=8 end_column=8}
    5 {file_name_id=4 function_name_id=5 line=758 end_line=758 column=16 end_column=16}
    6 {file_name_id=5 function_name_id=6 line=211 end_line=211 column=8 end_column=8}
    7 {file_name_id=6 function_name_id=7 line=621 end_line=621 column=12 end_column=12}
    8 {file_name_id=6 function_name_id=8 line=478 end_line=478 column=20 end_column=20}
    9 {file_name_id=7 function_name_id=9 line=372 end_line=372 column=8 end_column=8}
    10 {file_name_id=6 function_name_id=10 line=834 end_line=834 column=28 end_column=28}
    11 {file_name_id=7 function_name_id=11 line=464 end_line=464 column=26 end_column=26}
    12 {file_name_id=8 function_name_id=12 line=663 end_line=663 column=15 end_column=15}
    13 {file_name_id=9 function_name_id=13 line=3170 end_line=3170 column=25 end_column=25}
    14 {file_name_id=9 function_name_id=14 line=3225 end_line=3225 column=21 end_column=21}
    15 {file_name_id=10 function_name_id=15 line=128 end_line=128 column=8 end_column=8}
    16 {file_name_id=9 function_name_id=16 line=3447 end_line=3447 column=29 end_column=29}
    17 {file_name_id=9 function_name_id=17 line=3688 end_line=3688 column=19 end_column=19}
    18 {file_name_id=9 function_name_id=18 line=3748 end_line=3748 column=20 end_column=20}
    19 {file_name_id=11 function_name_id=3 line=50 end_line=50 column=9 end_column=9}

    StackFrames
    1 {file_location_id=1 parent_frame_id=1}
    2 {file_location_id=2 parent_frame_id=2}
    3 {file_location_id=3 parent_frame_id=3}
    4 {file_location_id=4 parent_frame_id=4}
    5 {file_location_id=5 parent_frame_id=5}
    6 {file_location_id=6 parent_frame_id=6}
    7 {file_location_id=7 parent_frame_id=7}
    8 {file_location_id=8 parent_frame_id=8}
    9 {file_location_id=9 parent_frame_id=9}
    10 {file_location_id=10 parent_frame_id=10}
    11 {file_location_id=11 parent_frame_id=11}
    12 {file_location_id=12 parent_frame_id=12}
    13 {file_location_id=13 parent_frame_id=13}
    14 {file_location_id=14 parent_frame_id=14}
    15 {file_location_id=15 parent_frame_id=15}
    16 {file_location_id=16 parent_frame_id=16}
    17 {file_location_id=17 parent_frame_id=17}
    18 {file_location_id=18 parent_frame_id=18}
    19 {file_location_id=19 parent_frame_id=19}


    ENTRY %main.0_spmd (param: f32[2,4]) -> f32[2,4] {
      %param = f32[2,4]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}, metadata={op_name="args[0]"}
      ROOT %ffi_call.1 = f32[2,4]{1,0} custom-call(%param), custom_call_target="rms_norm", operand_layout_constraints={f32[2,4]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(<unnamed function>)/custom_partitioning" stack_frame_id=19}, backend_config={eps = 9.99999974E-6 : f32}
    }

As you can see from the compiled program above, this `custom_partitioning` logic produces exactly the same program as the `shard_map` version above when the input is sharded on the batch dimension.

However, it’s worth noting that the behavior is *different* when the input is sharded along the data dimension. When used under `shard_map`, the data are resharded on the batch dimension, whereas with `custom_partitioning` the data are gathered onto each device.

    hlo_data_partitioned = jax.jit(rms_norm_partitioned, out_shardings=data_shd).lower(x_data_shd).compile().as_text().strip()

To also support automatic parallelization of the backwards pass, we would also need to write (similar) [`custom_partitioning()`](jax.experimental.custom_partitioning.html#jax.experimental.custom_partitioning.custom_partitioning "jax.experimental.custom_partitioning.custom_partitioning") rules for `rms_norm_fwd` and `rms_norm_bwd`, but we leave those as an exercise for the reader.

## Advanced topics[\#](#advanced-topics "Link to this heading")

This tutorial covers most of the basic steps that are required to get up and running with JAX’s FFI, but advanced use cases may require more features. We will leave these topics to future tutorials, but here are some possibly useful references:

- **Supporting multiple dtypes**: In this tutorial’s example, we restricted to only support `float32` inputs and outputs, but many use cases require supporting multiple different input types. One option to handle this is to register different FFI targets for all supported input types and then use Python to select the appropriate target for [`jax.ffi.ffi_call()`](_autosummary/jax.ffi.ffi_call.html#jax.ffi.ffi_call "jax.ffi.ffi_call") depending on the input types. But, this approach could get quickly unwieldy depending on the combinatorics of the supported cases. So it is also possible to define the C++ handler to accept `ffi::AnyBuffer` instead of `ffi::Buffer<Dtype>`. Then, the input buffer will include a `element_type()` method which can be used to define the appropriate dtype dispatching logic in the backend.

- **Stateful foreign functions**: It is also possible to use the FFI to wrap functions with associated state. There is a [low-level example included in the XLA test suite](https://github.com/openxla/xla/blob/737a7da3c5405583dc95773ac0bb11b1349fc9ea/xla/service/gpu/custom_call_test.cc#L794-L845), and a future tutorial will include more details.

[](external-callbacks.html "previous page")

previous

External callbacks

[](notebooks/cute_dsl_jax.html "next page")

next

Writing High-Performance GPU Kernels with CuTe DSL and JAX

Contents

- [A simple example](#a-simple-example)
- [Backend code](#backend-code)
  - [C++ interface](#c-interface)
  - [Building and registering an FFI handler](#building-and-registering-an-ffi-handler)
- [Frontend code](#frontend-code)
  - [Batching with `vmap`](#batching-with-vmap)
  - [Differentiation](#differentiation)
- [FFI calls on a GPU](#ffi-calls-on-a-gpu)
  - [Supporting multiple platforms](#supporting-multiple-platforms)
- [Sharding](#sharding)
  - [Using `shard_map`](#using-shard-map)
  - [Using `custom`` ``partitioning`](#using-custom-partitioning)
- [Advanced topics](#advanced-topics)

By The JAX authors

© Copyright 2024, The JAX Authors.\
