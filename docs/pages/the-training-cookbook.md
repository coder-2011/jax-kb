- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- The Training Cookbook

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/the-training-cookbook.rst "Download source file")
-  .pdf

# The Training Cookbook

## Contents

- [Device Mesh and Shardings](#device-mesh-and-shardings)
  - [Device Mesh](#device-mesh)
- [Train State Initialization](#train-state-initialization)
  - [Parameter Initialization](#parameter-initialization)
  - [Optimizer Initialization](#optimizer-initialization)
- [The Train Step (Functional Transformations)](#the-train-step-functional-transformations)
  - [Model Forward Pass](#model-forward-pass)
  - [Gradient and Optimizer Update](#gradient-and-optimizer-update)
- [The Training Loop](#the-training-loop)
  - [Efficiency via Asynchronous Dispatch](#efficiency-via-asynchronous-dispatch)
  - [Common Mistakes](#common-mistakes)
    - [Requesting device-to-host transfers](#requesting-device-to-host-transfers)
    - [Interrupting the accelerator](#interrupting-the-accelerator)
  - [Data Loading](#data-loading)
- [Achieving High Performance](#achieving-high-performance)
  - [Data Parallel](#data-parallel)
  - [Fully-Sharded Data Parallel (FSDP)](#fully-sharded-data-parallel-fsdp)
  - [Tensor Parallel](#tensor-parallel)

# The Training Cookbook[\#](#the-training-cookbook "Link to this heading")

Traditionally, machine learning codebases rely on libraries to perform much of the bookkeeping and parameter wrangling necessary for training large, complex models. While convenient, these libraries can abstract the key functionality and core APIs offered in JAX. The purpose of this cookbook, therefore, is to demonstrate best practices (or “recipes”) for writing simple yet high-performance machine learning training code directly in JAX. Following the patterns documented below will prepare your machine learning workloads to maximally leverage our compiler (XLA) for performance and tractability. Most training scripts adhere roughly to the following structure:

    def train_loop(config: Config):
      record_writer = RecordWriter()
      train_state = init_train_state(config)
      train_state = jax.tree.map(jax.ref.new_ref, train_state)
      batch = iter(get_dataset_on_device(config))
      for step in range(config.num_train_steps):
        metrics = train_step(config, train_state, next(batch))
        record_writer({"step": step} | metrics)

For each line of code above, we will explain the best practices and showcase the core technologies we have assembled to empower you to write simple, yet unbelievably performant code in JAX. The code above is a segment of a self-contained, completely functional [companion script](https://github.com/jax-ml/jax/blob/main/docs/the-training-cookbook.py) in which we initialize a [Vaswani et al. (2017)](https://arxiv.org/abs/170.03762) Transformer decoder, define the training loss for next-token prediction, and [Adam optimizer](https://arxiv.org/abs/1412.6980), in pure JAX. The code therein is suited to TPUs, CPUs, and GPUs, as well as single- and multi-host systems. For that reason, we use the terms *device* or *accelerator* to refer interchangeably to the hardware JAX is primarily performing arithmetic on—whether it be a TPU, GPU, or CPU—and *host system* to refer to operations performed exclusively using the host CPU. In this guide, there are many aspects of the JAX APIs we will gloss over for the sake of expediency. These are available for you to peruse at your leisure in our API documentation. However, there is a central JAX concept that one must confront in detail for much of what follows to cohere.

## Device Mesh and Shardings[\#](#device-mesh-and-shardings "Link to this heading")

JAX employs the [Single Program, Multiple Data (SPMD)](https://en.wikipedia.org/wiki/Single_program,_multiple_data) model of parallelism. This means we write a single program that runs on multiple devices, using annotations to specify which part of the data each device is responsible for. The two primary concepts for this are the [`jax.sharding.Mesh`](jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") and `jax.P`.

### Device Mesh[\#](#device-mesh "Link to this heading")

A [`jax.sharding.Mesh`](jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") is an arrangement of all our accelerators into a NumPy `ndarray`, together with string labels for the axes of the device array. The reason for using an array is that this allows for a very convenient annotation for how arrays should be partitioned across devices. For this introduction, we will use the notation of an ordered dictionary [^1], so that `{"x":`` ``2,`` ``"y":`` ``4}` refers to a device mesh of shape `(2,`` ``4)` with labeled axes `"x"` and `"y"`. To shard an array `param`, we decorate it with a `jax.P`, which is a tuple of `str`` ``|`` ``None` elements of the same length as the dimensions of the array. The `jax.P` specifies which axes of our array are to be sharded over which axes of devices. A more thorough account of the notation of shardings and sharded computations is available in [Distributed arrays and automatic parallelization](parallel.html#parallel). Some common sharding strategies such as data parallel, fully sharded data parallel, and basic tensor parallelism will be covered in [Achieving High Performance](#achieving-high-performance).

Example

Suppose we have a device mesh of `{"x":`` ``2,`` ``"y":`` ``4}` and an array `param` of shape `(32,`` ``64,`` ``64,`` ``128)`. If we shard this array with jax.P(None, “x”, “y”, None) \`, we end up with shards of size \`\`(32, 32, 16, 128)\` distributed across the devices. The `None` indicates that an axis should not be sharded. JAX implicitly broadcasts trailing axes, so an identical sharding can be achieved more concisely with jax.P(None, “x”, “y”). As a result, the shorthand for a fully replicated array (of any dimension) is jax.P().

Example

More advanced mesh geometries are convenient when aligned with the communication hierarchy of our devices. Host-to-host communication is typically slower than accelerator-to-accelerator communication. Suppose we have two host machines, each with eight attached GPUs. One might arrange the devices into a mesh of `{"host":`` ``2,`` ``"gpu":`` ``8}`. Then we can shard a parameter as follows:

    param = jnp.zeros((256, 192), out_sharding=jax.P("gpu", None))

The whole of `param` will be replicated twice, but within each host, it will be spread across the eight locally attached GPUs, with each GPU storing a shard of shape `(32,`` ``192)` in HBM. This is particularly useful for [Fully-Sharded Data Parallel (FSDP)](#fsdp-sharding).

## Train State Initialization[\#](#train-state-initialization "Link to this heading")

    @jax.jit
    def init_train_state(config: Config) -> dot_dict:
      train_state = dot_dict()
      train_state.params = init_param_state(config)
      train_state.opt = jax.tree.map(init_adam_state, train_state.params)
      return train_state

Before we can get started, the first thing we need to do is set up the train state. The train state encapsulates (unsurprisingly) all the *stateful* aspects of the training process. This typically includes, at a minimum, the model parameters and the optimizer state. The way we have structured this function (though you may choose to do otherwise) is to:

1.  Create a series of nested dictionaries to house the model parameters, and then

2.  [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") over those parameters to produce a similar set of nested dictionaries to house the accompanying optimizer states. (More on this [below](#optimizer-initialization).)

### Parameter Initialization[\#](#parameter-initialization "Link to this heading")

    @jax.jit
    def init_train_state(config: Config) -> dot_dict:
      train_state = dot_dict()
      train_state.params = init_param_state(config)
      train_state.opt = jax.tree.map(init_adam_state, train_state.params)
      return train_state

To initialize our parameters, we build a series of nested dictionaries that correspond to the semantic sections of the neural network. If we were using a layer-based library such as PyTorch or Flax, these might correspond to neural network layers. For this example, we could, in fact, get by with a completely flattened dictionary, but the nested approach is convenient both for working with some of the APIs in JAX and for structuring our code.

    def init_param_state(config: Config) -> dot_dict:
      root_key = jax.random.key(config.param_seed)
      key = map(ft.partial(jax.random.fold_in, root_key), it.count())
      zero_init = jax.nn.initializers.constant(0.0)
      he_init = jax.nn.initializers.he_normal(1, 1)
      dtype = config.dtype

      params = dot_dict(
        pos_embed=zero_init(next(key), (config.seq_length, config.embed_dim), dtype, config.pos_embed),
        layers=dot_dict(),
      )
      params.embedding = he_init(next(key), (config.vocab_size, config.embed_dim), dtype, config.embed)
      params.linear_in = dot_dict(
        kernel=he_init(next(key), (1, config.embed_dim), dtype, config.in_kernel),
        bias=zero_init(next(key), (config.embed_dim,), dtype, config.in_bias),
      )
      params.linear_out = dot_dict(
        kernel=he_init(next(key), (config.embed_dim, config.vocab_size), dtype, config.out_kernel),
      )
      for layer in range(config.num_layers):
        qkv_shape = (3, config.embed_dim, config.num_heads, config.head_dim)
        out_shape = (config.num_heads, config.head_dim, config.embed_dim)
        params.layers[layer] = dot_dict(
          attention=dot_dict(
            qkv=he_init(next(key), qkv_shape, dtype, config.att_qkv),
            out=he_init(next(key), out_shape, dtype, config.att_out),
          ),
          mlp=dot_dict(
            in_kernel=he_init(next(key), (config.embed_dim, config.mlp_dim), dtype, config.mlp_in),
            out_kernel=he_init(next(key), (config.mlp_dim, config.embed_dim), dtype, config.mlp_out),
          ),
        )
      return params

Our `get_param_state` function makes use of the `constant` and `he_normal` factories provided in [`jax.nn.initializers`](jax.nn.initializers.html#module-jax.nn.initializers "jax.nn.initializers"). These factories return an *initializer*, which is a function conforming to the following protocol:

    class Initializer(Protocol):
        def __call__(self, key, shape, dtype, out_sharding) -> jax.Array:
            ...

The functional flavor of JAX requires explicit handling of all stochasticity (viz. [Pseudorandom numbers](random-numbers.html#pseudorandom-numbers)), so we set up a little iterator that yields PRNG keys. Then, to build our parameters, we initialize them at their respective positions in the `params` nested dictionary, supplying the parameter shape, dtype, and sharding from the `Config` class.

Note

By specifying the shardings here, we initialize each shard of each parameter directly on the correct device in the device mesh where it needs to be, preventing the need for needless host-to-device transfers or, in the case of a model that does not fit in system memory, avoiding out-of-memory errors.

### Optimizer Initialization[\#](#optimizer-initialization "Link to this heading")

    @jax.jit
    def init_train_state(config: Config) -> dot_dict:
      train_state = dot_dict()
      train_state.params = init_param_state(config)
      train_state.opt = jax.tree.map(init_adam_state, train_state.params)
      return train_state

When it comes to setting up the optimizer state, things are a little less straightforward than when we built the model parameters. The [Adam optimizer](https://arxiv.org/abs/1412.6980) requires that, for each parameter, we keep track of three optimization states: `mu`, `nu`, and `count`. The simplest of these is `count`, which stores the number of training steps we have performed. This is just a scalar used to de-bias the Adam updates. The `mu` and `nu` states will be arrays of the same shape, dtype, and sharding as the accompanying parameter `param` [^2]

    def init_adam_state(param: jax.Array) -> dot_dict:
      adam_state = dot_dict(mu=jnp.zeros_like(param), nu=jnp.zeros_like(param), count=jnp.array(0))
      return adam_state

When we use [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"), it iterates over the items in `train_state.params`. For each parameter, it creates a corresponding Adam state, resulting in a new nested dictionary that mirrors the structure of `train_state.params`. Each leaf in this new structure contains the optimizer state for the corresponding parameter.

## The Train Step (Functional Transformations)[\#](#the-train-step-functional-transformations "Link to this heading")

    @jax.jit
    def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
      def loss_fn(params):
        logits = model_apply(config, params, batch["observed_ids"])
        labels = jax.nn.one_hot(batch["target_ids"], config.vocab_size)
        return -(labels * jax.nn.log_softmax(logits)).mean()

      params = jax.tree.map(jax.ref.get, train_state.params)
      loss, grad = jax.value_and_grad(loss_fn)(params)
      jax.tree.map(ft.partial(adam_update, config), train_state.params, grad, train_state.opt)
      metrics = {"train_loss": loss}
      return metrics

The train step is where we calculate the gradient of the model with respect to the current parameters and use the gradient, together with the optimizer, to update the parameters. To do this in JAX, we define the forward pass of the model, then we leverage JAX’s functional transformations to automatically generate the backward pass, which we use to calculate the gradients and perform the update.

### Model Forward Pass[\#](#model-forward-pass "Link to this heading")

    def model_apply(config: Config, params: dot_dict, tokens: jax.Array) -> jax.Array:
      out = params.embedding.at[tokens].get(out_sharding=config.act_seq)
      out += params.pos_embed
      del tokens

      for layer in range(config.num_layers):
        block = params.layers[layer]
        att_skip = out  # 1 billion dollars in venture capital funding please
        qkv = jnp.einsum("bsd,3dkh->bs3kh", out, block.attention.qkv, out_sharding=config.act_att)
        out = jax.nn.dot_product_attention(qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :], is_causal=True)
        out = jnp.einsum("bskh,khd->bsd", out, block.attention.out, out_sharding=config.act_seq)
        out += att_skip
        out *= jax.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

        mlp_skip = out  # machine learning circa 1986
        out = jnp.einsum("bsd,dh->bsh", out, block.mlp.in_kernel, out_sharding=config.act_hidden)
        out = jax.nn.gelu(out)
        out = jnp.einsum("bsh,hd->bsd", out, block.mlp.out_kernel, out_sharding=config.act_seq)
        out += mlp_skip
        out *= jax.lax.rsqrt(jnp.linalg.norm(out, axis=-1, keepdims=True) + 1e-6)

      logits = jnp.einsum("bsd,dl->bsl", out, params.linear_out.kernel, out_sharding=config.act_seq)
      return logits

The model’s forward pass is mostly unremarkable, aside from the `out_sharding` annotations we have supplied. These annotations declare what the result-sharding should be after the operation executes. The compiler uses these activation shardings, together with the parameter shardings we supplied when we [initialized the model](#parameter-initialization), to dynamically insert [communication collectives](https://en.wikipedia.org/wiki/Collective_operation) that ferry parameters and activations alike between devices. By choosing a good sharding strategy, we can achieve highly performant training (and inference) code. We will cover some standard strategies that serve most use cases in the section titled [Achieving High Performance](#achieving-high-performance). For a detailed discussion of the principles underpinning the design of sharding strategies, see [The Scaling Cookbook](https://jax-ml.github.io/scaling-book/).

### Gradient and Optimizer Update[\#](#gradient-and-optimizer-update "Link to this heading")

    @jax.jit
    def train_step(config: Config, train_state: dot_dict, batch: dict) -> dict:
      def loss_fn(params):
        logits = model_apply(config, params, batch["observed_ids"])
        labels = jax.nn.one_hot(batch["target_ids"], config.vocab_size)
        return -(labels * jax.nn.log_softmax(logits)).mean()

      params = jax.tree.map(jax.ref.get, train_state.params)
      loss, grad = jax.value_and_grad(loss_fn)(params)
      jax.tree.map(ft.partial(adam_update, config), train_state.params, grad, train_state.opt)
      metrics = {"train_loss": loss}
      return metrics

In order to calculate the gradient, we define the training loss. This is a function of the parameters that returns a scalar which summarizes how well our model, with the current `train_state` parameters, is explaining the data.

    loss, grad = jax.value_and_grad(loss_fn)(params)

By supplying this function to [`jax.value_and_grad()`](_autosummary/jax.value_and_grad.html#jax.value_and_grad "jax.value_and_grad"), we transform it into a function that returns both the scalar value and the gradient of `loss_fn` evaluated at `params` (the *value* and *grad*). Since we have defined our parameters in terms of a series of nested dictionaries, the gradient will also be a series of nested dictionaries, mirroring the parameters. Recall that, unlike the parameters, the optimizer states contain some extra, deeper nested dictionaries corresponding to the optimizer state per parameter. Take a moment, before reading the explanation, to ponder what the semantics of the following function call might be:

    jax.tree.map(ft.partial(adam_update, config), train_state.params, grad, train_state.opt)

Examining the call signature of the function `adam_apply` gives us a hint:

    def adam_update(config: Config, param: jax.Ref, grad: jax.Array, adam_state: dot_dict):
      adam_state.mu[...] = (1 - config.beta_1) * adam_state.mu[...] + config.beta_1 * grad
      adam_state.nu[...] = (1 - config.beta_2) * adam_state.nu[...] + config.beta_2 * grad**2
      adam_state.count[...] += 1

      mu_hat = adam_state.mu[...] / (1 - config.beta_1 ** adam_state.count[...])
      nu_hat = adam_state.nu[...] / (1 - config.beta_2 ** adam_state.count[...])
      param[...] -= config.learning_rate * mu_hat / (jnp.sqrt(nu_hat + config.eps_root) + config.eps)

Because `train_state.params` is the first argument, [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") uses its tree structure to guide the mapping process. [^3] This means that `train_state.opt` is traversed only as deep as the leaves of `train_state.params`. The optimizer state for each parameter is therefore passed in as a complete subtree, which allows us to easily access all relevant states (like `mu` and `nu`) for a given `param` inside `adam_apply`.

Tip

If we wished to use different optimization algorithms and states on different parameters in our model (or freeze some parameters), we could achieve this by modifying the body of `adam_apply` and replacing [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") with [`jax.tree_util.tree_map_with_path()`](_autosummary/jax.tree_util.tree_map_with_path.html#jax.tree_util.tree_map_with_path "jax.tree_util.tree_map_with_path"), which allows the operand function to customize its behavior depending on the parameter.

## The Training Loop[\#](#the-training-loop "Link to this heading")

    def train_loop(config: Config):
      record_writer = RecordWriter()
      train_state = init_train_state(config)
      train_state = jax.tree.map(jax.ref.new_ref, train_state)
      batch = iter(get_dataset_on_device(config))
      for step in range(config.num_train_steps):
        metrics = train_step(config, train_state, next(batch))
        record_writer({"step": step} | metrics)

During training, we have to orchestrate the flow of data between two key players: the host system and the accelerator. Ensuring smooth interplay between these systems is key to writing highly performant training code. The Python [GIL](https://en.wikipedia.org/wiki/Global_interpreter_lock) would ordinarily pose a significant obstacle here, but to work around this, the paradigm of [Asynchronous Dispatch](async_dispatch.html#async-dispatch) adopted by JAX makes this orchestration easy to accomplish. But, in order to leverage this paradigm, we need to be mindful of how our code will be executed when structuring our training step.

### Efficiency via Asynchronous Dispatch[\#](#efficiency-via-asynchronous-dispatch "Link to this heading")

One of the most important tasks performed by the host system is to fetch data and place it on the accelerators so that the accelerators are never waiting for data. The time when accelerators are waiting idle between train steps is referred to as the *step bubble*. We can leverage asynchronous dispatch to minimize the step bubble. Let’s see how this works with our training loop, discarding, for the moment, the line concerning the `record_writer`.

    for step in range(config.num_train_steps):
      metrics = train_step(config, train_state, next(batch))

When this code executes, Python will first query the range iterator, get `step` (with value `0`), then call `next(batch)`, which will take some time to retrieve the batch. Then, `train_step` gets called. So far, nothing out of the ordinary.

What happens next is interesting. Because [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit")-decorated calls are non-blocking, the call to `train_step` returns to the Python interpreter immediately. While the computation is enqueued on the accelerator, no work is actually performed yet. The Python loop continues, advancing the step counter and calling `next(batch)` for the *next* iteration. Once the second call to `train_step` is made, its inputs are now the mutated reference to `train_state` from the previous JIT call and a fresh batch of data. The runtime is clever and sees that in order to execute the second call to `train_step`, we first need to realize the `train_state` result of step `0` to perform the mutation. And so it fires off the computation for the first step, and, crucially, while this happens, `train_step`, once again, returns immediately, and the loop skips over again. Python now runs ahead until it encounters the `next(batch)` function at step 3, which proceeds to execute in Python, loading data, *while* the first train step is executing (for real this time). And just like that, we can simultaneously load data and perform math on the accelerator, without any traditional multiprocessing. [^4]

``` mermaid

        ---
displayMode: compact
---
gantt
    title Synchronous Dispatch: No Overlap
    axisFormat %

    section Host
    next(batch) :gb0, 0, 1000s
    next(batch) :gb1, after ajc0, 1000s
    next(batch) :gb2, after ajc1, 1000s

    section Accelerator

    train_step 0 :ajc0, after gb0, 2000s
    train_step 1 :ajc1, after gb1, 2000s
    
```

``` mermaid

        ---
displayMode: compact
---
gantt
    title JAX Asynchronous Dispatch: Host-Device Overlap
    axisFormat %

    section Host
    %% Task: id, name, start, duration_or_end
    next(batch) :gb0, 0, 1000s
    next(batch) :gb1, after gb0, 1000s
    next(batch) :gb2, after gb1, 1000s
    next(batch) :gb3, after jc0, 1000s
    next(batch) :gb4, after jc1, 1000s

    section Accelerator
    %% Task: id, name, start, duration_or_end
    train_step 0 :jc0, after gb1, 2000s
    train_step 1 :jc1, after jc0, 2000s
    train_step 2 :jc2, after jc1, 2000s
    
```

### Common Mistakes[\#](#common-mistakes "Link to this heading")

When writing asynchronous dispatch code in Python, there are two primary mistakes one should be wary of so as not to interrupt our careful orchestration of compute.

#### Requesting device-to-host transfers[\#](#requesting-device-to-host-transfers "Link to this heading")

Up until now, we have ignored what happens to the variable `metrics`. Indeed, if this is left dangling, nothing will happen, and we will achieve good overlap just as advertised. However, more often than not, we would like to observe telemetry from our train step, such as the current loss, gradient statistics, and so on. Suppose we were to insert code such as:

    metrics = train_step(config, train_state, next(batch))
    print({"step": step} | metrics)

Instead of the loop ticking over, `print` will incur a device-to-host transfer of whatever on-device arrays are in `metrics`. This interrupts the Python interpreter, and the code is forced to execute synchronously, producing a step bubble. The solution is slightly counterintuitive: at each step, we gather the telemetry for the *previous* step.

    class RecordWriter:
      prev_metrics = None

      def __call__(self, cur_metrics: dict):
        self.prev_metrics, log_metrics = cur_metrics, self.prev_metrics
        if log_metrics is None:
          return
        print(*it.starmap("{}: {}".format, log_metrics.items()), sep="\t")

and

    metrics = train_step(config, train_state, next(batch))

A small helper function like this is essential to achieve good overlap and make the most of the resources of our host system and our accelerator. Of course, the simple `print` statement here can be swapped out for any Python operation that requests data from the accelerator.

#### Interrupting the accelerator[\#](#interrupting-the-accelerator "Link to this heading")

The other common way in which we can waste spectacular amounts of cloud compute money is by unintentionally enqueuing math operations on the accelerator outside of the train step. Suppose we are using a cosine learning rate schedule.

    def learning_rate(count, init_value: float = 1e-4, decay_steps: int = 10_000, alpha: float = 1e-6):
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(count, decay_steps) / decay_steps))
        return init_value * (1 - alpha) * cosine_decay

A common pattern is to want to visualize the schedule alongside the other metrics we’re gathering. However, even if we use the clever `record_writer` class we defined earlier, the following code will create a bubble on the accelerator.

    metrics = train_step(config, train_state, next(batch))
    record_writer({"step": step, "learning_rate": learning_rate(step)} | metrics)

This is because we have used [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") in our calculations. When [`jax.numpy.minimum()`](_autosummary/jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum") is called, the Python integer `step` is promoted to a [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") and transferred to the accelerator (a host-to-device transfer). The calculation is now enqueued on the accelerator, outside our main `train_step`. To `print` the result, the value must be transferred back to the host (a device-to-host transfer). This round-trip forces the accelerator to synchronize with the host, and we have thrown away money by creating a performance bubble. The two ways to avoid this are to use NumPy for these calculations or to use the [`jax.default_device()`](_autosummary/jax.default_device.html#jax.default_device "jax.default_device") context manager.

    metrics = train_step(config, train_state, next(batch))
    with jax.default_device('cpu'):
      record_writer({"step": step, "learning_rate": learning_rate(step)} | metrics)

### Data Loading[\#](#data-loading "Link to this heading")

In addition to overlapping the actual loading of the data (that is, retrieving it from network storage to the host), JAX also allows us to overlap the host-to-device transfer of the data itself with the computation of the train step. The special function [`jax.device_put()`](_autosummary/jax.device_put.html#jax.device_put "jax.device_put") is carefully designed to be non-blocking, executing asynchronously, which makes it perfectly fine to use in the context of our train step. However, there is a more convenient function specifically designed for the task of loading data. In the following code, `dataset` is an ordinary Python iterator that yields a `dict` of batched data. By mapping over this iterator with [`jax.make_array_from_process_local_data()`](_autosummary/jax.make_array_from_process_local_data.html#jax.make_array_from_process_local_data "jax.make_array_from_process_local_data"), we generate a new iterator. Yielding from this new iterator will generate data placed on the device, ready for consumption by our train step. Internally, it will [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") to create [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") objects and queue them to be transferred to the device. Provided the data can be batched fast enough, on both TPUs and GPUs, these transfers will be overlapped with the train step computation.

    def get_dataset_on_device(config: Config) -> Iterator[dict[str, jax.Array]]:
      datset = get_dataset(config)
      sharding = jax.P(config.mesh_axis_names)
      return map(ft.partial(jax.make_array_from_process_local_data, sharding), datset)

## Achieving High Performance[\#](#achieving-high-performance "Link to this heading")

In this section, we will describe the three primary forms of model parallelism that are useful for training. During training, *throughput* is of paramount importance; that is, we wish to maximize the average number of operations per second. This contrasts with inference, where the goal is to minimize *latency* by ensuring all the operations happen in as little time as possible. Keeping throughput in mind as our ultimate goal for training, this section introduces the three primary strategies for sharding during training. For each strategy, we outline the JAX shardings that implement it and describe the collectives involved so that when studying program traces, you’ll have landmarks to look for to confirm that the program is behaving as expected. The sharding variables we define in the code blocks below correspond to their uses in the [initialization](#train-state-initialization) and [model forward pass](#model-forward-pass). But in the companion script these and other aspects of the training code are set conveniently using the global Config class.

    @jax.tree_util.register_static
    @dataclass(kw_only=True, frozen=True)
    class Config:
      mesh_axis_names: tuple[str, ...] = ("fsdp",)
      mesh_shape: tuple[int, ...] = (8,)
      seq_length: int = 128

      num_train_steps: int = 10**6
      host_batch_size: int = 16
      learning_rate: float = 1e-4
      beta_1: float = 0.9
      beta_2: float = 0.999
      eps: float = 1e-8
      eps_root: float = 0.0

      param_seed: int = 12738
      num_layers: int = 4
      embed_dim: int = 512
      mlp_dim: int = 512 * 4
      vocab_size: int = 2**8  # uint8 ascii encoding
      num_heads: int = 8
      head_dim: int = 128
      dtype: str = "bfloat16"

      embed: jax.P = jax.P(None, None)
      pos_embed: jax.P = jax.P(None, None)
      att_qkv: jax.P = jax.P(None, "fsdp", None, None)
      att_out: jax.P = jax.P("fsdp", None, None)
      mlp_in: jax.P = jax.P("fsdp", None)
      mlp_out: jax.P = jax.P(None, "fsdp")
      in_kernel: jax.P = jax.P(None, None)
      in_bias: jax.P = jax.P(None)
      out_kernel: jax.P = jax.P("fsdp", None)
      out_bias: jax.P = jax.P(None)

      act_ids: jax.P = jax.P("fsdp")
      act_seq: jax.P = jax.P("fsdp", None, None)
      act_att: jax.P = jax.P("fsdp", None, None, None)
      act_hidden: jax.P = jax.P("fsdp", None, None)

      def __post_init__(self):
        mesh = jax.make_mesh(self.mesh_shape, self.mesh_axis_names, len(self.mesh_shape) * (AxisType.Explicit,))
        jax.sharding.set_mesh(mesh)

### Data Parallel[\#](#data-parallel "Link to this heading")

Data parallel is the most common and easy-to-understand form of parallelism. In this scheme, each accelerator stores a complete copy of the model parameters, and we shard activations along the batch axis to split the computation of the gradients. To compute the gradients, each accelerator performs an individual forward and backward pass. Then, before the parameters are updated, XLA inserts an `AllReduce` to share the updates and keep the models in sync.

*Mesh:*

    mesh = jax.sharding.Mesh(jax.devices(), ('devices',))

*Parameter Shardings:*

    pos_embed = jax.P(None, None)
    att_qkv = jax.P(None, None, None, None)
    att_out = jax.P(None, None, None)
    mlp_in = jax.P(None, None)
    mlp_out = jax.P(None, None)
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P(None, None)
    out_bias = jax.P(None)

*Activation Shardings:*

    act_ids = jax.P("devices")
    act_seq = jax.P("devices", None, None)
    act_att = jax.P("devices", None, None, None)
    act_hidden = jax.P("devices", None, None)

### Fully-Sharded Data Parallel (FSDP)[\#](#fully-sharded-data-parallel-fsdp "Link to this heading")

The drawback of data-parallel sharding is that we have to keep multiple, full, redundant copies of the model parameters in HBM. This is a very performant strategy for small models, but since HBM is in short supply, we need to shard the model parameters as well. In the *Fully-Sharded Data Parallel (FSDP)* strategy, we shard both the activations and the model parameters. Now, as the forward pass happens, the parameters are, one-by-one, unsharded (via `AllGather`) into whole arrays before they are applied to the activations. This unsharding is brief and temporary, however, leading to a large saving in HBM. In the backward pass, each `AllGather` becomes a `ReduceScatter`. Then there is a final `ReduceScatter` at the optimizer update to synchronize gradients. Compared with Data parallelism, the total communication traffic is 50% higher, but our HBM pressure is reduced by the size of the model divided by the number of devices.

*Mesh:*

    mesh = jax.make_mesh((128*4,), ("fsdp",))

*Parameter Shardings:*

    pos_embed = jax.P(None, None)
    att_qkv = jax.P(None, "fsdp", None, None)
    att_out = jax.P("fsdp", None, None)
    mlp_in = jax.P("fsdp", None)
    mlp_out = jax.P(None, "fsdp")
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P("fsdp", None)
    out_bias = jax.P(None)

*Activation Shardings:*

    act_ids = jax.P("fsdp")
    act_seq = jax.P("fsdp", None, None)
    act_att = jax.P("fsdp", None, None, None)
    act_hidden = jax.P("fsdp", None, None)

Note

While FSDP entails a great deal more communication than data parallel, in practice we are able to overlap the communication with the compute, thereby hiding it and achieving the same throughput at a drastically improved HBM budget.

### Tensor Parallel[\#](#tensor-parallel "Link to this heading")

If our model is large enough and structured appropriately, it becomes beneficial to partition the computation within a single example across our accelerators. Using a matrix multiplication as an example, we can spread the large matrix multiplications over two or four accelerators. This entails significantly more communication, and so this strategy only works for computations with a very high arithmetic intensity, such as extremely large matrix multiplications. With multi-head self-attention, we opt to shard along the heads with a replicated sequence axis, since this offers the most natural amount of parallelism. If the MLP is large enough we can also efficiently shard the matrix multiplications.

*Mesh:*

    mesh = jax.make_mesh((128,4), ("fsdp", "tensor"))

*Parameter Shardings:*

    pos_embed = jax.P(None, "tensor")
    att_qkv = jax.P(None, "fsdp", "tensor", None)
    att_out = jax.P("fsdp", None, None)
    mlp_in = jax.P("fsdp", "tensor")
    mlp_out = jax.P("tensor", "fsdp")
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P("fsdp", None)
    out_bias = jax.P(None)

*Activation Shardings:*

    act_ids = jax.P("fsdp")
    act_seq = jax.P("fsdp", None, None)
    act_att = jax.P("fsdp", None, "tensor", None)
    act_hidden = jax.P("fsdp", None, "tensor")

\[[1](#id1)\]

Of course, all dictionaries are order-preserving in modern Python, so this is somewhat redundant.

\[[2](#id3)\]

This is accomplished by using the `zeros_like` constructor, but we could have specified the sharding manually using the `devices` argument of many of the [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") functions.

\[[3](#id4)\]

We could have achieved the same behavior equivalently by ordering `grad` first.

\[[4](#id5)\]

For the purposes of this explanation, you can think of `next(batch)` as just a sleep.

[](notebooks/colocated-python.html "previous page")

previous

Colocated Python

[](notebooks/autodiff_cookbook.html "next page")

next

The Autodiff Cookbook

Contents

- [Device Mesh and Shardings](#device-mesh-and-shardings)
  - [Device Mesh](#device-mesh)
- [Train State Initialization](#train-state-initialization)
  - [Parameter Initialization](#parameter-initialization)
  - [Optimizer Initialization](#optimizer-initialization)
- [The Train Step (Functional Transformations)](#the-train-step-functional-transformations)
  - [Model Forward Pass](#model-forward-pass)
  - [Gradient and Optimizer Update](#gradient-and-optimizer-update)
- [The Training Loop](#the-training-loop)
  - [Efficiency via Asynchronous Dispatch](#efficiency-via-asynchronous-dispatch)
  - [Common Mistakes](#common-mistakes)
    - [Requesting device-to-host transfers](#requesting-device-to-host-transfers)
    - [Interrupting the accelerator](#interrupting-the-accelerator)
  - [Data Loading](#data-loading)
- [Achieving High Performance](#achieving-high-performance)
  - [Data Parallel](#data-parallel)
  - [Fully-Sharded Data Parallel (FSDP)](#fully-sharded-data-parallel-fsdp)
  - [Tensor Parallel](#tensor-parallel)

By The JAX authors

© Copyright 2024, The JAX Authors.\

[^1]:

[^2]:

[^3]:

[^4]:
