- [](index.html)
- [API Reference](jax.html)
- [`jax.example_libraries` module](jax.example_libraries.html)
- `jax.example_libraries.optimizers` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.example_libraries.optimizers.rst "Download source file")
-  .pdf

# jax.example_libraries.optimizers module

## Contents

- [`JoinPoint`](#jax.example_libraries.optimizers.JoinPoint)
- [`Optimizer`](#jax.example_libraries.optimizers.Optimizer)
  - [`Optimizer.init_fn`](#jax.example_libraries.optimizers.Optimizer.init_fn)
  - [`Optimizer.params_fn`](#jax.example_libraries.optimizers.Optimizer.params_fn)
  - [`Optimizer.update_fn`](#jax.example_libraries.optimizers.Optimizer.update_fn)
- [`OptimizerState`](#jax.example_libraries.optimizers.OptimizerState)
  - [`OptimizerState.packed_state`](#jax.example_libraries.optimizers.OptimizerState.packed_state)
  - [`OptimizerState.subtree_defs`](#jax.example_libraries.optimizers.OptimizerState.subtree_defs)
  - [`OptimizerState.tree_def`](#jax.example_libraries.optimizers.OptimizerState.tree_def)
- [`adagrad()`](#jax.example_libraries.optimizers.adagrad)
- [`adam()`](#jax.example_libraries.optimizers.adam)
- [`adamax()`](#jax.example_libraries.optimizers.adamax)
- [`clip_grads()`](#jax.example_libraries.optimizers.clip_grads)
- [`constant()`](#jax.example_libraries.optimizers.constant)
- [`exponential_decay()`](#jax.example_libraries.optimizers.exponential_decay)
- [`inverse_time_decay()`](#jax.example_libraries.optimizers.inverse_time_decay)
- [`l2_norm()`](#jax.example_libraries.optimizers.l2_norm)
- [`make_schedule()`](#jax.example_libraries.optimizers.make_schedule)
- [`momentum()`](#jax.example_libraries.optimizers.momentum)
- [`nesterov()`](#jax.example_libraries.optimizers.nesterov)
- [`optimizer()`](#jax.example_libraries.optimizers.optimizer)
- [`pack_optimizer_state()`](#jax.example_libraries.optimizers.pack_optimizer_state)
- [`piecewise_constant()`](#jax.example_libraries.optimizers.piecewise_constant)
- [`polynomial_decay()`](#jax.example_libraries.optimizers.polynomial_decay)
- [`rmsprop()`](#jax.example_libraries.optimizers.rmsprop)
- [`rmsprop_momentum()`](#jax.example_libraries.optimizers.rmsprop_momentum)
- [`sgd()`](#jax.example_libraries.optimizers.sgd)
- [`sm3()`](#jax.example_libraries.optimizers.sm3)
- [`unpack_optimizer_state()`](#jax.example_libraries.optimizers.unpack_optimizer_state)

# `jax.example_libraries.optimizers` module[\#](#module-jax.example_libraries.optimizers "Link to this heading")

Examples of how to write optimizers with JAX.

You likely do not mean to import this module! The optimizers in this library are intended as examples only. If you are looking for a fully featured optimizer library, consider [Optax](https://github.com/google-deepmind/optax).

This module contains some convenient optimizer definitions, specifically initialization and update functions, which can be used with ndarrays or arbitrarily-nested tuple/list/dicts of ndarrays.

An optimizer is modeled as an `(init_fun,`` ``update_fun,`` ``get_params)` triple of functions, where the component functions have these signatures:

    init_fun(params)

    Args:
      params: pytree representing the initial parameters.

    Returns:
      A pytree representing the initial optimizer state, which includes the
      initial parameters and may also include auxiliary values like initial
      momentum. The optimizer state pytree structure generally differs from that
      of `params`.

    update_fun(step, grads, opt_state)

    Args:
      step: integer representing the step index.
      grads: a pytree with the same structure as `get_params(opt_state)`
        representing the gradients to be used in updating the optimizer state.
      opt_state: a pytree representing the optimizer state to be updated.

    Returns:
      A pytree with the same structure as the `opt_state` argument representing
      the updated optimizer state.

    get_params(opt_state)

    Args:
      opt_state: pytree representing an optimizer state.

    Returns:
      A pytree representing the parameters extracted from `opt_state`, such that
      the invariant `params == get_params(init_fun(params))` holds true.

Notice that an optimizer implementation has a lot of flexibility in the form of opt_state: it just has to be a pytree of JaxTypes (so that it can be passed to the JAX transforms defined in api.py) and it has to be consumable by update_fun and get_params.

Example Usage:

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params)

    def step(step, opt_state):
      value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
      opt_state = opt_update(step, grads, opt_state)
      return value, opt_state

    for i in range(num_steps):
      value, opt_state = step(i, opt_state)

*class* jax.example_libraries.optimizers.JoinPoint(*subtree*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L578-L586)[\#](#jax.example_libraries.optimizers.JoinPoint "Link to this definition")  
Bases: [`object`](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")

Marks the boundary between two joined (nested) pytrees.

&nbsp;

*class* jax.example_libraries.optimizers.Optimizer(*init_fn*, *update_fn*, *params_fn*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L133-L137)[\#](#jax.example_libraries.optimizers.Optimizer "Link to this definition")  
Bases: [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple "(in Python v3.14)")

Parameters:  
- **init_fn** (*InitFn*)

- **update_fn** (*UpdateFn*)

- **params_fn** (*ParamsFn*)

init_fn*: InitFn*[\#](#jax.example_libraries.optimizers.Optimizer.init_fn "Link to this definition")  
Alias for field number 0

params_fn*: ParamsFn*[\#](#jax.example_libraries.optimizers.Optimizer.params_fn "Link to this definition")  
Alias for field number 2

update_fn*: UpdateFn*[\#](#jax.example_libraries.optimizers.Optimizer.update_fn "Link to this definition")  
Alias for field number 1

&nbsp;

*class* jax.example_libraries.optimizers.OptimizerState(*packed_state*, *tree_def*, *subtree_defs*)[\#](#jax.example_libraries.optimizers.OptimizerState "Link to this definition")  
Bases: [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")

packed_state[\#](#jax.example_libraries.optimizers.OptimizerState.packed_state "Link to this definition")  
Alias for field number 0

subtree_defs[\#](#jax.example_libraries.optimizers.OptimizerState.subtree_defs "Link to this definition")  
Alias for field number 2

tree_def[\#](#jax.example_libraries.optimizers.OptimizerState.tree_def "Link to this definition")  
Alias for field number 1

&nbsp;

jax.example_libraries.optimizers.adagrad(*step_size*, *momentum=0.9*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L293-L328)[\#](#jax.example_libraries.optimizers.adagrad "Link to this definition")  
Construct optimizer triple for Adagrad.

Adaptive Subgradient Methods for Online Learning and Stochastic Optimization: [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

Parameters:  
- **step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **momentum** – optional, a positive scalar value for momentum

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.adam(*step_size*, *b1=0.9*, *b2=0.999*, *eps=1e-08*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L392-L426)[\#](#jax.example_libraries.optimizers.adam "Link to this definition")  
Construct optimizer triple for Adam.

Parameters:  
- **step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **b1** – optional, a positive scalar value for beta_1, the exponential decay rate for the first moment estimates (default 0.9).

- **b2** – optional, a positive scalar value for beta_2, the exponential decay rate for the second moment estimates (default 0.999).

- **eps** – optional, a positive scalar value for epsilon, a small constant for numerical stability (default 1e-8).

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.adamax(*step_size*, *b1=0.9*, *b2=0.999*, *eps=1e-08*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L428-L461)[\#](#jax.example_libraries.optimizers.adamax "Link to this definition")  
Construct optimizer triple for AdaMax (a variant of Adam based on infinity norm).

Parameters:  
- **step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **b1** – optional, a positive scalar value for beta_1, the exponential decay rate for the first moment estimates (default 0.9).

- **b2** – optional, a positive scalar value for beta_2, the exponential decay rate for the second moment estimates (default 0.999).

- **eps** – optional, a positive scalar value for epsilon, a small constant for numerical stability (default 1e-8).

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.clip_grads(*grad_tree*, *max_norm*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L569-L574)[\#](#jax.example_libraries.optimizers.clip_grads "Link to this definition")  
Clip gradients stored as a pytree of arrays to maximum norm max_norm.

&nbsp;

jax.example_libraries.optimizers.constant(*step_size*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L514-L518)[\#](#jax.example_libraries.optimizers.constant "Link to this definition")  
Return type:  
Schedule

&nbsp;

jax.example_libraries.optimizers.exponential_decay(*step_size*, *decay_steps*, *decay_rate*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L519-L523)[\#](#jax.example_libraries.optimizers.exponential_decay "Link to this definition")  

&nbsp;

jax.example_libraries.optimizers.inverse_time_decay(*step_size*, *decay_steps*, *decay_rate*, *staircase=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L524-L532)[\#](#jax.example_libraries.optimizers.inverse_time_decay "Link to this definition")  

&nbsp;

jax.example_libraries.optimizers.l2_norm(*tree*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L564-L568)[\#](#jax.example_libraries.optimizers.l2_norm "Link to this definition")  
Compute the l2 norm of a pytree of arrays. Useful for weight decay.

&nbsp;

jax.example_libraries.optimizers.make_schedule(*scalar_or_schedule*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L553-L560)[\#](#jax.example_libraries.optimizers.make_schedule "Link to this definition")  
Parameters:  
**scalar_or_schedule** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *Schedule*)

Return type:  
Schedule

&nbsp;

jax.example_libraries.optimizers.momentum(*step_size*, *mass*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L239-L264)[\#](#jax.example_libraries.optimizers.momentum "Link to this definition")  
Construct optimizer triple for SGD with momentum.

Parameters:  
- **step_size** (*Schedule*) – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **mass** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – positive scalar representing the momentum coefficient.

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.nesterov(*step_size*, *mass*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L266-L291)[\#](#jax.example_libraries.optimizers.nesterov "Link to this definition")  
Construct optimizer triple for SGD with Nesterov momentum.

Parameters:  
- **step_size** (*Schedule*) – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **mass** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – positive scalar representing the momentum coefficient.

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.optimizer(*opt_maker*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L140-L215)[\#](#jax.example_libraries.optimizers.optimizer "Link to this definition")  
Decorator to make an optimizer defined for arrays generalize to containers.

With this decorator, you can write init, update, and get_params functions that each operate only on single arrays, and convert them to corresponding functions that operate on pytrees of parameters. See the optimizers defined in optimizers.py for examples.

Parameters:  
**opt_maker** ([*Callable*](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Callable*](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Params\],* *State\],* [*Callable*](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Step,* *Updates,* *Params\],* *Params\],* [*Callable*](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[State\],* *Params\]\]\]*) –

a function that returns an `(init_fun,`` ``update_fun,`` ``get_params)` triple of functions that might only work with ndarrays, as per

    init_fun :: ndarray -> OptStatePytree ndarray
    update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
    get_params :: OptStatePytree ndarray -> ndarray

Returns:  
An `(init_fun,`` ``update_fun,`` ``get_params)` triple of functions that work on arbitrary pytrees, as per

    init_fun :: ParameterPytree ndarray -> OptimizerState
    update_fun :: OptimizerState -> OptimizerState
    get_params :: OptimizerState -> ParameterPytree ndarray

The OptimizerState pytree type used by the returned functions is isomorphic to `ParameterPytree`` ``(OptStatePytree`` ``ndarray)`, but may store the state instead as e.g. a partially-flattened data structure for performance.

Return type:  
[Callable](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [Optimizer](#jax.example_libraries.optimizers.Optimizer "jax.example_libraries.optimizers.Optimizer")\]

&nbsp;

jax.example_libraries.optimizers.pack_optimizer_state(*marked_pytree*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L604-L622)[\#](#jax.example_libraries.optimizers.pack_optimizer_state "Link to this definition")  
Converts a marked pytree to an OptimizerState.

The inverse of unpack_optimizer_state. Converts a marked pytree with the leaves of the outer pytree represented as JoinPoints back into an OptimizerState. This function is intended to be useful when deserializing optimizer states.

Parameters:  
**marked_pytree** – A pytree containing JoinPoint leaves that hold more pytrees.

Returns:  
An equivalent OptimizerState to the input argument.

&nbsp;

jax.example_libraries.optimizers.piecewise_constant(*boundaries*, *values*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L541-L552)[\#](#jax.example_libraries.optimizers.piecewise_constant "Link to this definition")  
Parameters:  
- **boundaries** (*Any*)

- **values** (*Any*)

&nbsp;

jax.example_libraries.optimizers.polynomial_decay(*step_size*, *decay_steps*, *final_step_size*, *power=1.0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L533-L540)[\#](#jax.example_libraries.optimizers.polynomial_decay "Link to this definition")  

&nbsp;

jax.example_libraries.optimizers.rmsprop(*step_size*, *gamma=0.9*, *eps=1e-08*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L330-L356)[\#](#jax.example_libraries.optimizers.rmsprop "Link to this definition")  
Construct optimizer triple for RMSProp.

Parameters:  
**step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar. gamma: Decay parameter. eps: Epsilon parameter.

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.rmsprop_momentum(*step_size*, *gamma=0.9*, *eps=1e-08*, *momentum=0.9*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L358-L390)[\#](#jax.example_libraries.optimizers.rmsprop_momentum "Link to this definition")  
Construct optimizer triple for RMSProp with momentum.

This optimizer is separate from the rmsprop optimizer because it needs to keep track of additional parameters.

Parameters:  
- **step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **gamma** – Decay parameter.

- **eps** – Epsilon parameter.

- **momentum** – Momentum parameter.

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.sgd(*step_size*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L219-L238)[\#](#jax.example_libraries.optimizers.sgd "Link to this definition")  
Construct optimizer triple for stochastic gradient descent.

Parameters:  
**step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.sm3(*step_size*, *momentum=0.9*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L463-L510)[\#](#jax.example_libraries.optimizers.sm3 "Link to this definition")  
Construct optimizer triple for SM3.

Memory-Efficient Adaptive Optimization for Large-Scale Learning. [https://arxiv.org/abs/1901.11150](https://arxiv.org/abs/1901.11150)

Parameters:  
- **step_size** – positive scalar, or a callable representing a step size schedule that maps the iteration index to a positive scalar.

- **momentum** – optional, a positive scalar value for momentum

Returns:  
An (init_fun, update_fun, get_params) triple.

&nbsp;

jax.example_libraries.optimizers.unpack_optimizer_state(*opt_state*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/example_libraries/optimizers.py#L587-L603)[\#](#jax.example_libraries.optimizers.unpack_optimizer_state "Link to this definition")  
Converts an OptimizerState to a marked pytree.

Converts an OptimizerState to a marked pytree with the leaves of the outer pytree represented as JoinPoints to avoid losing information. This function is intended to be useful when serializing optimizer states.

Parameters:  
**opt_state** – An OptimizerState

Returns:  
A pytree with JoinPoint leaves that contain a second level of pytrees.

[](jax.example_libraries.html "previous page")

previous

`jax.example_libraries` module

[](jax.example_libraries.stax.html "next page")

next

`jax.example_libraries.stax` module

Contents

- [`JoinPoint`](#jax.example_libraries.optimizers.JoinPoint)
- [`Optimizer`](#jax.example_libraries.optimizers.Optimizer)
  - [`Optimizer.init_fn`](#jax.example_libraries.optimizers.Optimizer.init_fn)
  - [`Optimizer.params_fn`](#jax.example_libraries.optimizers.Optimizer.params_fn)
  - [`Optimizer.update_fn`](#jax.example_libraries.optimizers.Optimizer.update_fn)
- [`OptimizerState`](#jax.example_libraries.optimizers.OptimizerState)
  - [`OptimizerState.packed_state`](#jax.example_libraries.optimizers.OptimizerState.packed_state)
  - [`OptimizerState.subtree_defs`](#jax.example_libraries.optimizers.OptimizerState.subtree_defs)
  - [`OptimizerState.tree_def`](#jax.example_libraries.optimizers.OptimizerState.tree_def)
- [`adagrad()`](#jax.example_libraries.optimizers.adagrad)
- [`adam()`](#jax.example_libraries.optimizers.adam)
- [`adamax()`](#jax.example_libraries.optimizers.adamax)
- [`clip_grads()`](#jax.example_libraries.optimizers.clip_grads)
- [`constant()`](#jax.example_libraries.optimizers.constant)
- [`exponential_decay()`](#jax.example_libraries.optimizers.exponential_decay)
- [`inverse_time_decay()`](#jax.example_libraries.optimizers.inverse_time_decay)
- [`l2_norm()`](#jax.example_libraries.optimizers.l2_norm)
- [`make_schedule()`](#jax.example_libraries.optimizers.make_schedule)
- [`momentum()`](#jax.example_libraries.optimizers.momentum)
- [`nesterov()`](#jax.example_libraries.optimizers.nesterov)
- [`optimizer()`](#jax.example_libraries.optimizers.optimizer)
- [`pack_optimizer_state()`](#jax.example_libraries.optimizers.pack_optimizer_state)
- [`piecewise_constant()`](#jax.example_libraries.optimizers.piecewise_constant)
- [`polynomial_decay()`](#jax.example_libraries.optimizers.polynomial_decay)
- [`rmsprop()`](#jax.example_libraries.optimizers.rmsprop)
- [`rmsprop_momentum()`](#jax.example_libraries.optimizers.rmsprop_momentum)
- [`sgd()`](#jax.example_libraries.optimizers.sgd)
- [`sm3()`](#jax.example_libraries.optimizers.sm3)
- [`unpack_optimizer_state()`](#jax.example_libraries.optimizers.unpack_optimizer_state)

By The JAX authors

© Copyright 2024, The JAX Authors.\
