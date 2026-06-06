- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- Gradient checkpointing with `jax.checkpoint` (`jax.remat`)

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/gradient-checkpointing.md "Download source file")
-  .pdf

# Gradient checkpointing with jax.checkpoint (jax.remat)

## Contents

- [Let’s think step by step](#let-s-think-step-by-step)
  - [`jax.checkpoint` fundamentals](#jax-checkpoint-fundamentals)
  - [Custom policies for what’s saveable](#custom-policies-for-what-s-saveable)
  - [Custom policies for offload](#custom-policies-for-offload)
  - [List of policies](#list-of-policies)
  - [Advanced: Recursive `jax.checkpoint`](#advanced-recursive-jax-checkpoint)
- [Practical notes](#practical-notes)

# Gradient checkpointing with `jax.checkpoint` (`jax.remat`)[\#](#gradient-checkpointing-with-jax-checkpoint-jax-remat "Link to this heading")

In this tutorial, you will learn how to control JAX automatic differentiation’s saved values using [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") (also known as `jax.remat()`), which can be particularly helpful in machine learning.

If you are new to automatic differentiation (autodiff) or need to refresh your memory, JAX has an [Automatic differentiation](automatic-differentiation.html#automatic-differentiation) tutorial and several [Advanced automatic differentiation guides](advanced_guides.html#advanced-guides).

**TL;DR** Use the [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") decorator (aliased as `jax.remat()`) with [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") to control which intermediates are saved on the forward pass versus the recomputed intermediates on the backward pass, trading off memory and FLOPs.

If you don’t use [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint"), the `jax.grad(f)(x)` forward pass stores Jacobian coefficients and other intermediates to use during the backward pass. These saved values are called *residuals*.

**Note:** Don’t miss the [Practical notes](#gradient-checkpointing-practical-notes) for a discussion about how [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") interacts with [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit").

    import jax
    import jax.numpy as jnp

    def g(W, x):
      y = jnp.dot(W, x)
      return jnp.sin(y)

    def f(W1, W2, W3, x):
      x = g(W1, x)
      x = g(W2, x)
      x = g(W3, x)
      return x

    W1 = jnp.ones((5, 4))
    W2 = jnp.ones((6, 5))
    W3 = jnp.ones((7, 6))
    x = jnp.ones(4)

    # Inspect the 'residual' values to be saved on the forward pass
    # if you were to evaluate `jax.grad(f)(W1, W2, W3, x)`
    from jax.ad_checkpoint import print_saved_residuals
    print_saved_residuals(f, W1, W2, W3, x)

    f32[5,4] from the argument W1
    f32[6,5] from the argument W2
    f32[7,6] from the argument W3
    f32[4] from the argument x
    f32[5] output of sin from /tmp/ipykernel_1933/1857807639.py:6:9 (g)
    f32[5] output of cos from /tmp/ipykernel_1933/1857807639.py:6:9 (g)
    f32[6] output of sin from /tmp/ipykernel_1933/1857807639.py:6:9 (g)
    f32[6] output of cos from /tmp/ipykernel_1933/1857807639.py:6:9 (g)
    f32[7] output of cos from /tmp/ipykernel_1933/1857807639.py:6:9 (g)

By applying [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to sub-functions, as a decorator or at specific application sites, you force JAX not to save any of that sub-function’s residuals. Instead, only the inputs of a [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")-decorated function might be saved, and any residuals consumed on the backward pass are re-computed from those inputs as needed:

    def f2(W1, W2, W3, x):
      x = jax.checkpoint(g)(W1, x)
      x = jax.checkpoint(g)(W2, x)
      x = jax.checkpoint(g)(W3, x)
      return x

    print_saved_residuals(f2, W1, W2, W3, x)

    f32[5,4] from the argument W1
    f32[6,5] from the argument W2
    f32[7,6] from the argument W3
    f32[4] from the argument x
    f32[5] output of sin from /tmp/ipykernel_1933/1857807639.py:6:9 (g)
    f32[6] output of sin from /tmp/ipykernel_1933/1857807639.py:6:9 (g)

Here, the values of two `sin` applications are saved because they are arguments in subsequent applications of the [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")-decorated `g` function, and inputs to a [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")-decorated function may be saved. But no values of `cos` applications are saved.

To control which values are saveable without having to edit the definition of the function to be differentiated, you can use a rematerialization *policy*. Here is an example that saves only the results of `dot` operations with no batch dimensions (since they are often FLOP-bound, and hence worth saving rather than recomputing):

    f3 = jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    print_saved_residuals(f3, W1, W2, W3, x)

    f32[5,4] from the argument W1
    f32[6,5] from the argument W2
    f32[7,6] from the argument W3
    f32[4] from the argument x
    f32[5] output of reduce_precision from /tmp/ipykernel_1933/1857807639.py:5:6 (g)
    f32[6] output of reduce_precision from /tmp/ipykernel_1933/1857807639.py:5:6 (g)
    f32[7] output of reduce_precision from /tmp/ipykernel_1933/1857807639.py:5:6 (g)

You can also use policies to refer to intermediate values you name using [`jax.ad_checkpoint.checkpoint_name()`](_autosummary/jax.ad_checkpoint.checkpoint_name.html#jax.ad_checkpoint.checkpoint_name "jax.ad_checkpoint.checkpoint_name"):

    from jax.ad_checkpoint import checkpoint_name

    def f4(W1, W2, W3, x):
      x = checkpoint_name(g(W1, x), name='a')
      x = checkpoint_name(g(W2, x), name='b')
      x = checkpoint_name(g(W3, x), name='c')
      return x

    f4 = jax.checkpoint(f4, policy=jax.checkpoint_policies.save_only_these_names('a'))
    print_saved_residuals(f4, W1, W2, W3, x)

    f32[5,4] from the argument W1
    f32[6,5] from the argument W2
    f32[7,6] from the argument W3
    f32[4] from the argument x
    f32[5] output of reduce_precision from /tmp/ipykernel_1933/3722338705.py:4:6 (f4)

When playing around with these toy examples, you can get a closer look at what’s going on using a custom `print_fwd_bwd` utility defined in this notebook:

    from jax.tree_util import tree_flatten, tree_unflatten

    from rich.console import Console
    from rich.table import Table
    import rich.text

    def print_fwd_bwd(f, *args, **kwargs) -> None:
      args, in_tree = tree_flatten((args, kwargs))

      def f_(*args):
        args, kwargs = tree_unflatten(in_tree, args)
        return f(*args, **kwargs)

      fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

      y, f_vjp = jax.vjp(f_, *args)
      res, in_tree = tree_flatten(f_vjp)

      def g_(*args):
        *res, y = args
        f_vjp = tree_unflatten(in_tree, res)
        return f_vjp(y)

      bwd = jax.make_jaxpr(g_)(*res, y).jaxpr

      table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
      table.add_row("[bold green]forward computation:",
                    "[bold green]backward computation:")
      table.add_row(rich.text.Text.from_ansi(str(fwd)),
                    rich.text.Text.from_ansi(str(bwd)))
      console = Console(width=240, force_jupyter=True)
      console.print(table)

    def _renderable_repr(self):
      return self.html
    rich.jupyter.JupyterRenderable._repr_html_ = _renderable_repr

    # Without using `jax.checkpoint`:
    print_fwd_bwd(f, W1, W2, W3, x)

```
                                                                                                                                                         
  forward computation:                                         backward computation:                                                                     
                                                                                                                                                         
  { lambda ; a:f32[5,4] b:f32[6,5] c:f32[7,6] d:f32[4]. let    { lambda ; a:f32[5,4] b:f32[6,5] c:f32[7,6] d:f32[4] e:f32[5] f:f32[5] g:f32[6] h:f32[6]  
      e:f32[5] = dot_general[                                      i:f32[7] j:f32[7]. let                                                                
        dimension_numbers=(([1], [0]), ([], []))                   k:f32[7] = mul j i                                                                    
        preferred_element_type=float32                             l:f32[6] = dot_general[                                                               
      ] a d                                                          dimension_numbers=(([0], [0]), ([], []))                                            
      f:f32[5] = sin e                                               preferred_element_type=float32                                                      
      g:f32[5] = cos e                                             ] k c                                                                                 
      h:f32[6] = dot_general[                                      m:f32[7,6] = dot_general[                                                             
        dimension_numbers=(([1], [0]), ([], []))                     dimension_numbers=(([], []), ([], []))                                              
        preferred_element_type=float32                               preferred_element_type=float32                                                      
      ] b f                                                        ] k h                                                                                 
      i:f32[6] = sin h                                             n:f32[6] = mul l g                                                                    
      j:f32[6] = cos h                                             o:f32[5] = dot_general[                                                               
      k:f32[7] = dot_general[                                        dimension_numbers=(([0], [0]), ([], []))                                            
        dimension_numbers=(([1], [0]), ([], []))                     preferred_element_type=float32                                                      
        preferred_element_type=float32                             ] n b                                                                                 
      ] c i                                                        p:f32[6,5] = dot_general[                                                             
      l:f32[7] = sin k                                               dimension_numbers=(([], []), ([], []))                                              
      m:f32[7] = cos k                                               preferred_element_type=float32                                                      
    in (l, a, b, c, d, g, f, j, i, m) }                            ] n f                                                                                 
                                                                   q:f32[5] = mul o e                                                                    
                                                                   r:f32[4] = dot_general[                                                               
                                                                     dimension_numbers=(([0], [0]), ([], []))                                            
                                                                     preferred_element_type=float32                                                      
                                                                   ] q a                                                                                 
                                                                   s:f32[5,4] = dot_general[                                                             
                                                                     dimension_numbers=(([], []), ([], []))                                              
                                                                     preferred_element_type=float32                                                      
                                                                   ] q d                                                                                 
                                                                 in (s, p, m, r) }                                                                       
```

    # Using `jax.checkpoint` with policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable:
    print_fwd_bwd(f3, W1, W2, W3, x)

```
                                                                                                                                                             
  forward computation:                                                   backward computation:                                                               
                                                                                                                                                             
  { lambda ; a:f32[5,4] b:f32[6,5] c:f32[7,6] d:f32[4]. let              let jaxpr = { lambda ; a:f32[5] b:f32[6] c:f32[7] d:f32[5,4] e:f32[6,5] f:f32[7,6]  
      e:f32[5] = dot_general[                                                g:f32[4] h:f32[7]. let                                                          
        dimension_numbers=(([1], [0]), ([], []))                             i:f32[5] = sin a                                                                
        preferred_element_type=float32                                       j:f32[5] = cos a                                                                
      ] a d                                                                  k:f32[6] = sin b                                                                
      f:f32[5] = reduce_precision[exponent_bits=8 mantissa_bits=23] e        l:f32[6] = cos b                                                                
      g:f32[5] = sin f                                                       m:f32[7] = cos c                                                                
      h:f32[6] = dot_general[                                                n:f32[7] = mul h m                                                              
        dimension_numbers=(([1], [0]), ([], []))                             o:f32[6] = dot_general[                                                         
        preferred_element_type=float32                                         dimension_numbers=(([0], [0]), ([], []))                                      
      ] b g                                                                    preferred_element_type=float32                                                
      i:f32[6] = reduce_precision[exponent_bits=8 mantissa_bits=23] h        ] n f                                                                           
      j:f32[6] = sin i                                                       p:f32[7,6] = dot_general[                                                       
      k:f32[7] = dot_general[                                                  dimension_numbers=(([], []), ([], []))                                        
        dimension_numbers=(([1], [0]), ([], []))                               preferred_element_type=float32                                                
        preferred_element_type=float32                                       ] n k                                                                           
      ] c j                                                                  q:f32[6] = mul o l                                                              
      l:f32[7] = reduce_precision[exponent_bits=8 mantissa_bits=23] k        r:f32[5] = dot_general[                                                         
      m:f32[7] = sin l                                                         dimension_numbers=(([0], [0]), ([], []))                                      
    in (m, a, b, c, d, f, i, l) }                                              preferred_element_type=float32                                                
                                                                             ] q e                                                                           
                                                                             s:f32[6,5] = dot_general[                                                       
                                                                               dimension_numbers=(([], []), ([], []))                                        
                                                                               preferred_element_type=float32                                                
                                                                             ] q i                                                                           
                                                                             t:f32[5] = mul r j                                                              
                                                                             u:f32[4] = dot_general[                                                         
                                                                               dimension_numbers=(([0], [0]), ([], []))                                      
                                                                               preferred_element_type=float32                                                
                                                                             ] t d                                                                           
                                                                             v:f32[5,4] = dot_general[                                                       
                                                                               dimension_numbers=(([], []), ([], []))                                        
                                                                               preferred_element_type=float32                                                
                                                                             ] t g                                                                           
                                                                           in (v, s, p, u) } in                                                              
                                                                         { lambda ; w:f32[5,4] x:f32[6,5] y:f32[7,6] z:f32[4] ba:f32[5] bb:f32[6] bc:f32[7]  
                                                                             bd:f32[7]. let                                                                  
                                                                             be:f32[5,4] bf:f32[6,5] bg:f32[7,6] bh:f32[4] = remat2[                         
                                                                               differentiated=True                                                           
                                                                               jaxpr=jaxpr                                                                   
                                                                               policy=<function dots_with_no_batch_dims_saveable at 0xXXXXXXXXXXXX>          
                                                                               prevent_cse=True                                                              
                                                                             ] ba bb bc w x y z bd                                                           
                                                                           in (be, bf, bg, bh) }                                                             
```

## Let’s think step by step[\#](#let-s-think-step-by-step "Link to this heading")

**Note:** It may help to check out the [“Advanced automatic differentiation” guides](advanced_guides.html#advanced-guides) prior to continuing here.

### `jax.checkpoint` fundamentals[\#](#jax-checkpoint-fundamentals "Link to this heading")

In both [`jax.linearize()`](_autosummary/jax.linearize.html#jax.linearize "jax.linearize") and [`jax.vjp()`](_autosummary/jax.vjp.html#jax.vjp "jax.vjp"), there is flexibility in how and when some values are computed. Different choices can trade off memory use against FLOPs. JAX provides control over these choices with [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint").

One such choice is whether to perform Jacobian coefficient computations on the forward pass, as soon as the inputs are available, or on the backward pass, just before the coefficients are needed. Consider the example of `sin_vjp`:

    def sin_vjp(x):
      y = jnp.sin(x)
      cos_x = jnp.cos(x)
      return y, lambda y_bar: cos_x * y_bar

Another valid implementation would compute the value of `jnp.cos(x)` on the backward pass rather than on the forward pass:

    def sin_vjp2(x):
      y = jnp.sin(x)
      return y, lambda y_bar: jnp.cos(x) * y_bar

For this particular function, the amount of memory used by the two versions is the same, though you’ve reduced the FLOPs for the primal computation (the forward pass) and increased the FLOPs for the cotangent computation (the backward pass).

There’s another choice when it comes to function composition. Recall the VJP rule for a composition of two functions:

    def f(x):
      y = g(x)
      z = h(y)
      return z

    def f_vjp(x):
      y, g_vjp = jax.vjp(g, x)
      z, h_vjp = jax.vjp(h, y)
      def f_bwd(z_bar):
        y_bar, = h_vjp(z_bar)
        x_bar, = g_vjp(y_bar)
        return x_bar
      return z, f_bwd

An alternative is:

    def f_vjp_checkpoint(x):
      y = g(x)
      z, h_vjp = jax.vjp(h, y)
      def f_bwd2(z_bar):
        y_bar, = h_vjp(z_bar)
        _, g_vjp = jax.vjp(g, x)
        x_bar, = g_vjp(y_bar)
        return x_bar
      return z, f_bwd2

Using words, this alternative implementation doesn’t compute `g_vjp`, or the residual values in its closure, on the forward pass. Instead, it only computes them in the backward pass `f_bwd2`. That means `f_vjp_checkpoint` requires less memory: if `g` and `h` each required similar amounts of memory for their residuals, each much larger than `x`, then the function produced by `f_vjp_checkpoint(x)` requires half the memory as that of `f_vjp(x)`!

The cost you pay is redundant work: in `f_bwd2` you must re-evaluate `g(x)` as part of `jax.vjp(g,`` ``x)` just to discard its value (in the underscore variable on the line `_,`` ``g_vjp`` ``=`` ``jax.vjp(g,`` ``x)`).

You can get this VJP behavior in autodiff — without having to write VJP functions directly — by instead using [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") in an alternative definition of the original function `f`:

    def f_checkpoint(x):
      y = jax.checkpoint(g)(x)
      z = h(y)
      return z

In other words, you apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to `g` — the first stage of `f` — rather than to `f` itself. This way, when you evaluate `jax.grad(f_checkpoint)(x)`, you’d get a computation like:

1.  Run the forward pass of `g`, discarding residual values.

2.  Run the forward pass of `h`, saving residuals.

3.  Run the backward pass of `h`, consuming residuals from step 2.

4.  Re-run the forward pass of `g`, saving residuals.

5.  Run the backward pass of `g`, consuming residuals from step 4.

That is, by evaluating `jax.grad(f_checkpoint)(x)` we’d get the same computation as:

    def f_checkpoint_grad(x):
      y = g(x)                  # step 1
      _, h_vjp = jax.vjp(h)(y)  # step 2
      y_bar, = h_vjp(1.0)       # step 3
      _, g_vjp = jax.vjp(g, x)  # step 4
      x_bar, = g_vjp(y_bar)     # step 5
      return x_bar

In general, `jax.checkpoint(foo)` is a new function which has the same input-output behavior as `foo`, but behaves differently under autodiff, particularly under [`jax.linearize()`](_autosummary/jax.linearize.html#jax.linearize "jax.linearize") and [`jax.vjp()`](_autosummary/jax.vjp.html#jax.vjp "jax.vjp") (and their wrappers, like [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad")) but not [`jax.jvp()`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp"). When differentiated, only the input to a [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")-differentiated function is stored on the forward pass. On the backward pass, the residuals (intermediates from `foo` and its Jacobian coefficient values needed for the backward pass) are recomputed.

Notice that if `f`` ``=`` ``lambda`` ``x:`` ``h(g(x))` is the function you want to differentiate (in other words, if you want to apply `jax.grad(f)`) you don’t get any memory savings by applying [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to `f` itself. That’s because evaluating `jax.grad(jax.checkpoint(f))(x)` would lead to a computation, such as:

1.  Run the forward pass, discarding all residuals.

2.  Immediately re-run the forward pass, saving residuals.

3.  Run the backward pass, consuming residuals from step 2.

In code, you’d have something like:

    def f_grad_bad(x):
      _ = f(x)                  # step 1
      _, f_vjp = jax.vjp(f, x)  # step 2
      x_bar, = f_vjp(1.0)       # step 3
      return x_bar

You also wouldn’t get any memory savings by applying [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to `h`, the second stage of `f`. That’s because evaluating `jax.grad(lambda`` ``x:`` ``jax.checkpoint(h)(g(x)))` would lead to a computation, such as:

1.  Run the forward pass of `g`, saving residuals.

2.  Run the forward pass of `h`, discarding residuals.

3.  Immediately re-run the forward pass of `h`, saving residuals.

4.  Run the backward pass of `h`, consuming residuals from step 3.

5.  Run the backward pass of `g`, consuming residuals from step 1.

In code you’d have something like:

    def f_grad_bad2(x):
      y, g_vjp = jax.vjp(g, x)  # step 1
      z = h(y)                  # step 2
      _, h_vjp = jax.vjp(h, y)  # step 3
      y_bar, = h_vjp(1.0)       # step 3
      x_bar, = g_vjp(y_bar)     # step 5
      return x_bar

Slightly more generally, if you had a chain composition of functions, such as `f`` ``=`` ``lambda`` ``x:`` ``f3(f2(f1(x)))`, and were interested in evaluating `jax.grad(f)`, you could say that you:

- Shouldn’t apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to the whole function `f`, since that wouldn’t save any memory (and will perform wasteful recomputation).

- Shouldn’t apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to the last sub-function `f3`, since that wouldn’t save any memory (and will perform wasteful recomputation).

- Could apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to `f1`, `f2`, or their composition `lambda`` ``x:`` ``f2(f1(x))`, since any of those might save memory and would express different memory/recompute tradeoffs.

### Custom policies for what’s saveable[\#](#custom-policies-for-what-s-saveable "Link to this heading")

As shown so far, using [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") switches from one extreme to another:

- Without [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint"), JAX’s autodiff tends to compute everything possible on the forward pass and store it for the backward pass.

- With a [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") decorator, you instead compute as little as possible on the forward pass and recompute values as needed on the backward pass.

To operate between these two extremes, saving some things and not others, you can carefully place [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") decorators on sub-functions. But that requires editing the function to be differentiated, e.g. model code, which may be inconvenient. It can also be hard to experiment with variations.

So an alternative is to use the `policy` argument to [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint"). A policy is a callable (i.e. a function) which takes as input a type-level specification of a first order primitive application and returns a boolean indicating whether the corresponding output value(s) are allowed to be saved as residuals (or instead must be recomputed in the (co)tangent computation as needed). To write robust code, a policy should be selected from the attributes on `jax.checkpoint_policies`, like [`jax.checkpoint_policies.dots_with_no_batch_dims_saveable()`](_autosummary/jax.checkpoint_policies.dots_with_no_batch_dims_saveable.html#jax.checkpoint_policies.dots_with_no_batch_dims_saveable "jax.checkpoint_policies.dots_with_no_batch_dims_saveable"), since the API for writing custom policy callables is considered internal.

For example, consider this function to be differentiated:

    def loss(params, x, y):
      return jnp.sum((predict(params, x) - y)**2)

    def predict(params, x):
      *Ws, Wlast = params
      for W in Ws:
        x = layer(W, x)
      x = jnp.dot(Wlast, x)
      return x

    def layer(W, x):
      return jnp.sin(jnp.dot(W, x))

    W1 = W2 = W3 = jnp.ones((4, 4))
    params = [W1, W2, W3]
    x = jnp.ones(4)
    y = jnp.ones(4)

    print_saved_residuals(loss, params, x, y)

    f32[4,4] from the argument params[0]
    f32[4,4] from the argument params[1]
    f32[4,4] from the argument params[2]
    f32[4] from the argument x
    f32[4] output of sin from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] output of cos from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] output of sin from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] output of cos from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] output of mul from /tmp/ipykernel_1933/4230705069.py:2:17 (loss)

Instead of saving so many values on the forward pass, perhaps you only want to save the results of matrix multiplications with no batch dimension (since they may be FLOP- rather than memory-bound). You can do that using the policy [`jax.checkpoint_policies.dots_with_no_batch_dims_saveable()`](_autosummary/jax.checkpoint_policies.dots_with_no_batch_dims_saveable.html#jax.checkpoint_policies.dots_with_no_batch_dims_saveable "jax.checkpoint_policies.dots_with_no_batch_dims_saveable"):

    loss_checkpoint = jax.checkpoint(loss, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    print_saved_residuals(loss_checkpoint, params, x, y)

    f32[4,4] from the argument params[0]
    f32[4,4] from the argument params[1]
    f32[4,4] from the argument params[2]
    f32[4] from the argument x
    f32[4] from the argument y
    f32[4] output of reduce_precision from /tmp/ipykernel_1933/4230705069.py:12:17 (layer)
    f32[4] output of reduce_precision from /tmp/ipykernel_1933/4230705069.py:12:17 (layer)
    f32[4] output of reduce_precision from /tmp/ipykernel_1933/4230705069.py:8:6 (predict)

Notice also that by providing a policy, you didn’t need to edit the code defining `loss`, `predict`, or `layer`. That is particularly convenient if you want to experiment with policies in calling code (such as a training script) without changing library code (for example, the neural network library).

Some policies can refer to values named with [`jax.ad_checkpoint.checkpoint_name()`](_autosummary/jax.ad_checkpoint.checkpoint_name.html#jax.ad_checkpoint.checkpoint_name "jax.ad_checkpoint.checkpoint_name"):

    from jax.ad_checkpoint import checkpoint_name

    def predict(params, x):
      *Ws, Wlast = params
      for i, W in enumerate(Ws):
        x = layer(W, x)
        x = checkpoint_name(x, name=f'layer{i}_output')
      x = jnp.dot(Wlast, x)
      return x

By itself, [`jax.ad_checkpoint.checkpoint_name()`](_autosummary/jax.ad_checkpoint.checkpoint_name.html#jax.ad_checkpoint.checkpoint_name "jax.ad_checkpoint.checkpoint_name") is just an identity function. But because some policy functions know to look for them, you can use the names to control whether certain values output by [`jax.ad_checkpoint.checkpoint_name()`](_autosummary/jax.ad_checkpoint.checkpoint_name.html#jax.ad_checkpoint.checkpoint_name "jax.ad_checkpoint.checkpoint_name") are considered saveable:

    print_saved_residuals(loss, params, x, y)

    f32[4,4] from the argument params[0]
    f32[4,4] from the argument params[1]
    f32[4,4] from the argument params[2]
    f32[4] from the argument x
    f32[4] output of cos from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] named 'layer0_output' from /tmp/ipykernel_1933/178264713.py:7:8 (predict)
    f32[4] output of cos from /tmp/ipykernel_1933/4230705069.py:12:9 (layer)
    f32[4] named 'layer1_output' from /tmp/ipykernel_1933/178264713.py:7:8 (predict)
    f32[4] output of mul from /tmp/ipykernel_1933/4230705069.py:2:17 (loss)

    loss_checkpoint2 = jax.checkpoint(loss, policy=jax.checkpoint_policies.save_any_names_but_these('layer1_output'))
    print_saved_residuals(loss_checkpoint2, params, x, y)

    f32[4,4] from the argument params[0]
    f32[4,4] from the argument params[1]
    f32[4,4] from the argument params[2]
    f32[4] from the argument x
    f32[4] from the argument y

Another policy which refers to names is `jax.checkpoint_policies.save_only_these_names`.

### Custom policies for offload[\#](#custom-policies-for-offload "Link to this heading")

You may consider offloading to CPU memory instead of recomputing when checkpointing to save accelerator memory. `jax.checkpoint_policies.offload_dot_with_no_batch_dims` can offload the results of matrix multiplications with no batch dimensions to the CPU.

    from jax import checkpoint

    def checkpoint_offload_dot_with_no_batch_dims(self):
      policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
          "device", "pinned_host")

      @functools.partial(checkpoint, policy=policy)
      def f(x):
        x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
        x = jnp.sin(x)
        x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
        x = jnp.sin(x)
        x = jnp.einsum('ij,jk->ik', x, x, precision=lax.Precision.HIGHEST)
        x = jnp.sin(x)
        x = jnp.sum(x)
        return x

One of JAX’s checkpoint policies allows specified checkpoint names to be offloaded to CPUs. This policy is implemented through `jax.checkpoint_policies.save_and_offload_only_these_names`, which has four arguments: `names_which_can_be_saved`, `names_which_can_be_offloaded`, the offloading source, and destination. Names listed in `names_which_can_be_saved` are kept on the device, names listed in `names_which_can_be_offloaded` are moved to CPU memory, and other names or operations without names are recomputed. For example, if we have checkpoint names `y`, `z`, and `w`, `y` can be saved on the device, `z` can be offloaded to CPU memory, and `w` can be recomputed.

    from jax import checkpoint
    from jax.ad_checkpoint import checkpoint_name
    from jax._src import test_util as jtu

    def checkpoint_names_saved_offloaded_recomputed(self):
      mesh = jtu.create_mesh((2,), ("x",))
      shape = (256, 128)
      np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
      s = NamedSharding(mesh, P("x"))
      inp = jax.device_put(np_inp, s)

      policy = jax.checkpoint_policies.save_and_offload_only_these_names(
          names_which_can_be_saved=["y"], names_which_can_be_offloaded=["z"],
          offload_src='device', offload_dst='pinned_host')

      @functools.partial(checkpoint, policy=policy)
      def f(x):
        def g(ys, _):
          y, _ = ys
          y = checkpoint_name(jnp.sin(y), "y")
          z = checkpoint_name(jnp.sin(y), "z")
          z = z.T
          w = checkpoint_name(jnp.sin(z), "w")
          return (w.T, jnp.sum(w)), None
        _, scan_out = jax.lax.scan(g, (x, np.array(1, dtype=np.float32)), [np_inp])[0]
        return scan_out

The code defines a function `f` that which applies checkpointing with a custom policy. This policy determines which computations can be saved or offloaded during execution. Inside `f`, there is a nested function `g` that performs the core computations. The `jax.lax.scan` function is used to apply `g` repeatedly over the input data.

### List of policies[\#](#list-of-policies "Link to this heading")

The policies can be found [here](https://docs.jax.dev/en/latest/jax.html#checkpoint-policies).

Policies only indicate what is saveable; a value is only saved if it’s actually needed by the backward pass.

### Advanced: Recursive `jax.checkpoint`[\#](#advanced-recursive-jax-checkpoint "Link to this heading")

By applying [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") in the right way, there are many tradeoffs between memory usage and (re)computation that can be expressed. One surprising example is *recursive* checkpointing, where you apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") to a function which itself calls [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint")-decorated functions in a way so that memory usage from the chain composition of \\D\\ functions scales like \\\mathcal{O}(\log_2 D)\\ rather than \\\mathcal{O}(D)\\.

As a toy example, consider the chain composition of multiple [`jax.numpy.sin()`](_autosummary/jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin") functions:

    def chain_compose(funs):
      def f(x):
        for fun in funs:
          x = fun(x)
        return x
      return f

    f = chain_compose([jnp.sin] * 8)
    print_saved_residuals(f, 3.)

    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)

In general, the number of stored residuals scales linearly with the length of the chain:

    f = chain_compose([jnp.sin] * 16)
    print_saved_residuals(f, 3.)

    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)
    f32[] output of cos from /tmp/ipykernel_1933/410288286.py:4:10 (chain_compose.<locals>.f)

But you can apply [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") recursively to improve the scaling:

    def recursive_checkpoint(funs):
      if len(funs) == 1:
        return funs[0]
      elif len(funs) == 2:
        f1, f2 = funs
        return lambda x: f1(f2(x))
      else:
        f1 = recursive_checkpoint(funs[:len(funs)//2])
        f2 = recursive_checkpoint(funs[len(funs)//2:])
        return lambda x: f1(jax.checkpoint(f2)(x))

    f = recursive_checkpoint([jnp.sin] * 8)
    print_saved_residuals(f, 3.)

    f32[] from the argument x
    f32[] output of sin from /tmp/ipykernel_1933/1943107544.py:6:21 (recursive_checkpoint.<locals>.<lambda>)
    f32[] output of cos from /tmp/ipykernel_1933/1943107544.py:6:24 (recursive_checkpoint.<locals>.<lambda>)
    f32[] output of cos from /tmp/ipykernel_1933/1943107544.py:6:21 (recursive_checkpoint.<locals>.<lambda>)

    f = recursive_checkpoint([jnp.sin] * 16)
    print_saved_residuals(f, 3.)

    f32[] from the argument x
    f32[] output of sin from /tmp/ipykernel_1933/1943107544.py:6:21 (recursive_checkpoint.<locals>.<lambda>)
    f32[] output of sin from /tmp/ipykernel_1933/1943107544.py:6:21 (recursive_checkpoint.<locals>.<lambda>)
    f32[] output of cos from /tmp/ipykernel_1933/1943107544.py:6:24 (recursive_checkpoint.<locals>.<lambda>)
    f32[] output of cos from /tmp/ipykernel_1933/1943107544.py:6:21 (recursive_checkpoint.<locals>.<lambda>)

The cost here, as usual, is recomputation: in particular, you end up performing \\\mathcal{O}(\log_2 D)\\ times as many FLOPs:

    f = chain_compose([jnp.sin] * 8)
    print_fwd_bwd(f, 3.)

```
                                                                                                                                 
  forward computation:                  backward computation:                                                                    
                                                                                                                                 
  { lambda ; a:f32[]. let               { lambda ; a:f32[] b:f32[] c:f32[] d:f32[] e:f32[] f:f32[] g:f32[] h:f32[] i:f32[]. let  
      b:f32[] = sin a                       j:f32[] = mul i h                                                                    
      c:f32[] = cos a                       k:f32[] = mul j g                                                                    
      d:f32[] = sin b                       l:f32[] = mul k f                                                                    
      e:f32[] = cos b                       m:f32[] = mul l e                                                                    
      f:f32[] = sin d                       n:f32[] = mul m d                                                                    
      g:f32[] = cos d                       o:f32[] = mul n c                                                                    
      h:f32[] = sin f                       p:f32[] = mul o b                                                                    
      i:f32[] = cos f                       q:f32[] = mul p a                                                                    
      j:f32[] = sin h                     in (q,) }                                                                              
      k:f32[] = cos h                                                                                                            
      l:f32[] = sin j                                                                                                            
      m:f32[] = cos j                                                                                                            
      n:f32[] = sin l                                                                                                            
      o:f32[] = cos l                                                                                                            
      p:f32[] = sin n                                                                                                            
      q:f32[] = cos n                                                                                                            
    in (p, c, e, g, i, k, m, o, q) }                                                                                             
```

    f = recursive_checkpoint([jnp.sin] * 8)
    print_fwd_bwd(f, 3.)

```
                                                                                                                                        
  forward computation:                                                              backward computation:                               
                                                                                                                                        
  { lambda ; a:f32[]. let                                                           { lambda ; a:f32[] b:f32[] c:f32[] d:f32[]. let     
      b:f32[] = remat2[                                                                 e:f32[] = mul d c                               
        differentiated=False                                                            f:f32[] = mul e b                               
        jaxpr={ lambda ; c:f32[]. let d:f32[] = sin c; e:f32[] = sin d in (e,) }        g:f32[] = remat2[                               
        policy=None                                                                       differentiated=True                           
        prevent_cse=True                                                                  jaxpr={ lambda ; h:f32[] i:f32[]. let         
      ] a                                                                                     j:f32[] = sin h                           
      f:f32[] = sin b                                                                         k:f32[] = cos h                           
      g:f32[] = sin f                                                                         l:f32[] = cos j                           
      h:f32[] = sin g                                                                         m:f32[] = mul i l                         
      i:f32[] = sin h                                                                         n:f32[] = mul m k                         
      j:f32[] = sin i                                                                       in (n,) }                                   
      k:f32[] = cos i                                                                     policy=None                                   
      l:f32[] = sin j                                                                     prevent_cse=True                              
      m:f32[] = cos j                                                                   ] a f                                           
    in (l, a, g, k, m) }                                                                o:f32[] = remat2[                               
                                                                                          differentiated=True                           
                                                                                          jaxpr={ lambda ; p:f32[] q:f32[]. let         
                                                                                              r:f32[] = sin p                           
                                                                                              s:f32[] = sin r                           
                                                                                              t:f32[] = sin s                           
                                                                                              u:f32[] = cos s                           
                                                                                              v:f32[] = cos t                           
                                                                                              w:f32[] = mul q v                         
                                                                                              x:f32[] = mul w u                         
                                                                                              y:f32[] = remat2[                         
                                                                                                differentiated=True                     
                                                                                                jaxpr={ lambda ; z:f32[] ba:f32[]. let  
                                                                                                    bb:f32[] = sin z                    
                                                                                                    bc:f32[] = cos z                    
                                                                                                    bd:f32[] = cos bb                   
                                                                                                    be:f32[] = mul ba bd                
                                                                                                    bf:f32[] = mul be bc                
                                                                                                  in (bf,) }                            
                                                                                                policy=None                             
                                                                                                prevent_cse=True                        
                                                                                              ] p x                                     
                                                                                            in (y,) }                                   
                                                                                          policy=None                                   
                                                                                          prevent_cse=True                              
                                                                                        ] 3.0:f32[] g                                   
                                                                                      in (o,) }                                         
```

## Practical notes[\#](#practical-notes "Link to this heading")

When differentiated functions are staged out to XLA for compilation — for example by applying [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") to a function which contains a [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") call — XLA will automatically optimize the computation, including decisions about when to compute or rematerialize values. As a result, **[`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") often isn’t needed for differentiated functions under a [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit")**. XLA will optimize things for you.

One exception is when using staged-out control flow, like [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan"). Automatic compiler optimizations across multiple control flow primitives (for example, across a forward-pass `scan` and the corresponding backward-pass `scan`), typically aren’t as thorough. As a result, it’s often a good idea to use [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") on the body function passed to [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan").

For example, one common pattern in large [Transformer models](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) is to express the architecture as a [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan") over layers so as to reduce compilation times. That is, using a simple fully-connected network as an analogy, instead of writing something like this:

    LayerParam = tuple[jnp.ndarray, jnp.ndarray]  # Weights-bias pair for a layer.
    ParamsList = list[LayerParam]

    def net(params: ParamsList, x: jnp.ndarray):
      for W, b in params:
        x = jnp.maximum(jnp.dot(x, W) + b, 0.)
      return x

Instead, iterate over the layer application with [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan"):

    params = [(jnp.array([[0.5, 0.5], [1., 1.]]), jnp.array([0.5, 0.5])),
              (jnp.array([[0.5, 0.5], [1., 1.]]), jnp.array([0.5, 0.5]))]

    all_weights = jnp.stack([W for W, _ in params])
    all_biases = jnp.stack([b for _, b in params])

    def layer(x, W_b_pair):
      W, b = W_b_pair
      out = jnp.maximum(jnp.dot(x, W) + b, 0.)
      return out, None

    def net(all_weights, all_biases, x):
      x, _ = jax.lax.scan(layer, x, (all_weights, all_biases))
      return x

This scan-over-layers version reduces compile times, but by foiling some compiler optimizations it can lead to inefficient computation of gradients. To mitigate the issue, you can use [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") on the scanned function:

    from functools import partial

    @partial(jax.checkpoint,
             policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def layer(x, W_b_pair):
      W, b = W_b_pair
      out = jnp.maximum(jnp.dot(x, W) + b, 0.)
      return out, None

By using [`jax.checkpoint()`](_autosummary/jax.checkpoint.html#jax.checkpoint "jax.checkpoint") this way, you’re manually controlling which values JAX’s autodiff saves between the forward and backward passes, and therefore not relying on XLA optimizations to choose for you.

[](notebooks/cute_dsl_jax.html "previous page")

previous

Writing High-Performance GPU Kernels with CuTe DSL and JAX

[](aot.html "next page")

next

Ahead-of-time lowering and compilation

Contents

- [Let’s think step by step](#let-s-think-step-by-step)
  - [`jax.checkpoint` fundamentals](#jax-checkpoint-fundamentals)
  - [Custom policies for what’s saveable](#custom-policies-for-what-s-saveable)
  - [Custom policies for offload](#custom-policies-for-offload)
  - [List of policies](#list-of-policies)
  - [Advanced: Recursive `jax.checkpoint`](#advanced-recursive-jax-checkpoint)
- [Practical notes](#practical-notes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
