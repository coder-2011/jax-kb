- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- JAX internals: The jaxpr language

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/jaxpr.md "Download source file")
-  .pdf

# JAX internals: The jaxpr language

## Contents

- [`jax.core.ClosedJaxpr`](#jax-core-closedjaxpr)
- [Handling pytrees](#handling-pytrees)
- [Constant variables (vars)](#constant-variables-vars)
- [Higher-order JAX primitives](#higher-order-jax-primitives)
  - [`cond` primitive (conditionals)](#cond-primitive-conditionals)
  - [`while` primitive](#while-primitive)
  - [`scan` primitive](#scan-primitive)
  - [`(p)jit` primitive](#p-jit-primitive)

# JAX internals: The jaxpr language[\#](#jax-internals-the-jaxpr-language "Link to this heading")

Jaxprs are JAX’s internal intermediate representation (IR) of programs. They are explicitly typed, functional, first-order, and in algebraic normal form (ANF).

Conceptually, one can think of JAX transformations, such as [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") or [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), as first trace-specializing the Python function to be transformed into a small and well-behaved intermediate form that is then interpreted with transformation-specific interpretation rules.

One of the reasons JAX can pack so much power into such a small software package is that it starts with a familiar and flexible programming interface (Python with NumPy) and it uses the actual Python interpreter to do most of the heavy lifting to distill the essence of the computation into a simple statically-typed expression language with limited higher-order features.

That language is the jaxpr language. The jaxpr term syntax looks as follows:

    jaxpr ::=
      { lambda <binder> , ... .
        let <eqn>
            ...
        in ( <atom> , ... ) }

    binder ::= <var>:<array_type>
    var ::= a | b | c | ...
    atom ::= <var> | <literal>
    literal ::= <int32> | <int64> | <float32> | <float64>

    eqn ::= <binder> , ... = <primitive> [ <params> ] <atom> , ...

Not all Python programs can be processed this way, but it turns out that many scientific computing and machine learning programs can.

Before you proceed, remember that not all JAX transformations literally materialize a jaxpr as described above. Some of them, such as differentiation or batching, will apply transformations incrementally during tracing. Nevertheless, if one wants to understand how JAX works internally, or to make use of the result of JAX tracing, it is useful to understand jaxprs.

## `jax.core.ClosedJaxpr`[\#](#jax-core-closedjaxpr "Link to this heading")

A jaxpr instance represents a function with one or more typed parameters (input variables) and one or more typed results. The results depend only on the input variables; there are no free variables captured from enclosing scopes. The inputs and outputs have types, which in JAX are represented as abstract values.

There are two related representations in the code for jaxprs, `jax.core.Jaxpr` and `jax.core.ClosedJaxpr`. A `jax.core.ClosedJaxpr` represents a partially-applied `jax.core.Jaxpr`, and is what you obtain when you use [`jax.make_jaxpr()`](_autosummary/jax.make_jaxpr.html#jax.make_jaxpr "jax.make_jaxpr") to inspect jaxprs. It has the following fields:

- `jaxpr`: is a `jax.core.Jaxpr` representing the actual computation content of the function (described below).

- `consts` is a list of constants.

The most interesting part of the `ClosedJaxpr` is the actual execution content, represented as a `jax.core.Jaxpr` as printed using the following grammar:

    jaxpr ::= { lambda Var* ; Var+.
                let Eqn*
                in  [Expr+] }

where:

- The parameters of the jaxpr are shown as two lists of variables separated by `;`:

  - The first set of variables are the ones that have been introduced to stand for constants that have been hoisted out. These are called the `constvars`, and in a `jax.core.ClosedJaxpr` the `consts` field holds corresponding values.

  - The second list of variables, called `invars`, correspond to the inputs of the traced Python function.

- `Eqn*` is a list of equations, defining intermediate variables referring to intermediate expressions. Each equation defines one or more variables as the result of applying a primitive on some atomic expressions. Each equation uses only input variables and intermediate variables defined by previous equations.

- `Expr+`: is a list of output atomic expressions (literals or variables) for the jaxpr.

Equations are printed as follows:

    Eqn  ::= let Var+ = Primitive [ Param* ] Expr+

where:

- `Var+` are one or more intermediate variables to be defined as the output of a primitive invocation (some primitives can return multiple values).

- `Expr+` are one or more atomic expressions, each either a variable or a literal constant. A special variable `unitvar` or literal `unit`, printed as `*`, represents a value that is not needed in the rest of the computation and has been elided. That is, units are just placeholders.

- `Param*` are zero or more named parameters to the primitive, printed in square brackets. Each parameter is shown as `Name`` ``=`` ``Value`.

Most jaxpr primitives are first-order (they take just one or more Expr as arguments):

    Primitive := add | sub | sin | mul | ...

The most common jaxpr primitives are documented in the [`jax.lax`](jax.lax.html#module-jax.lax "jax.lax") module.

For example, here is the jaxpr produced for the function `func1` below:

    from jax import make_jaxpr
    import jax.numpy as jnp

    def func1(first, second):
       temp = first + jnp.sin(second) * 3.
       return jnp.sum(temp)

    print(make_jaxpr(func1)(jnp.zeros(8), jnp.ones(8)))

    { lambda ; a:f32[8] b:f32[8]. let
        c:f32[8] = sin b
        d:f32[8] = mul c 3.0:f32[]
        e:f32[8] = add a d
        f:f32[] = reduce_sum[axes=(0,) out_sharding=None] e
      in (f,) }

Here there are no constvars, `a` and `b` are the input variables and they correspond respectively to `first` and `second` function parameters. The scalar literal `3.0` is kept inline. The `reduce_sum` primitive has named parameters `axes` and `input_shape`, in addition to the operand `e`.

Note that even though execution of a program that calls into JAX builds a jaxpr, Python-level control-flow and Python-level functions execute normally. This means that just because a Python program contains functions and control-flow, the resulting jaxpr does not have to contain control-flow or higher-order features.

For example, when tracing the function `func3` JAX will inline the call to `inner` and the conditional `if`` ``second.shape[0]`` ``>`` ``4`, and will produce the same jaxpr as before:

    def func2(inner, first, second):
      temp = first + inner(second) * 3.
      return jnp.sum(temp)

    def inner(second):
      if second.shape[0] > 4:
        return jnp.sin(second)
      else:
        assert False

    def func3(first, second):
      return func2(inner, first, second)

    print(make_jaxpr(func3)(jnp.zeros(8), jnp.ones(8)))

    { lambda ; a:f32[8] b:f32[8]. let
        c:f32[8] = sin b
        d:f32[8] = mul c 3.0:f32[]
        e:f32[8] = add a d
        f:f32[] = reduce_sum[axes=(0,) out_sharding=None] e
      in (f,) }

## Handling pytrees[\#](#handling-pytrees "Link to this heading")

In jaxpr there are no tuple types; instead primitives take multiple inputs and produce multiple outputs. When processing a function that has structured inputs or outputs, JAX will flatten those and in jaxpr they will appear as lists of inputs and outputs. For more details, refer to the [Pytrees](pytrees.html#pytrees) tutorial.

For example, the following code produces an identical jaxpr to what you saw earlier (with two input vars, one for each element of the input tuple):

    def func4(arg):  # The `arg` is a pair.
      temp = arg[0] + jnp.sin(arg[1]) * 3.
      return jnp.sum(temp)

    print(make_jaxpr(func4)((jnp.zeros(8), jnp.ones(8))))

    { lambda ; a:f32[8] b:f32[8]. let
        c:f32[8] = sin b
        d:f32[8] = mul c 3.0:f32[]
        e:f32[8] = add a d
        f:f32[] = reduce_sum[axes=(0,) out_sharding=None] e
      in (f,) }

## Constant variables (vars)[\#](#constant-variables-vars "Link to this heading")

Some values in jaxprs are constants, in that their value does not depend on the jaxpr’s arguments. When these values are scalars they are represented directly in the jaxpr equations. Non-scalar array constants are instead hoisted out to the top-level jaxpr, where they correspond to constant variables (“constvars”). These constvars differ from the other jaxpr parameters (“invars”) only as a bookkeeping convention.

## Higher-order JAX primitives[\#](#higher-order-jax-primitives "Link to this heading")

Jaxpr includes several higher-order JAX primitives. They are more complicated because they include sub-jaxprs.

### `cond` primitive (conditionals)[\#](#cond-primitive-conditionals "Link to this heading")

JAX traces through normal Python conditionals. To capture a conditional expression for dynamic execution, one must use the [`jax.lax.switch()`](_autosummary/jax.lax.switch.html#jax.lax.switch "jax.lax.switch") and [`jax.lax.cond()`](_autosummary/jax.lax.cond.html#jax.lax.cond "jax.lax.cond") constructors, which have the signatures:

    lax.switch(index: int, branches: Sequence[A -> B], operand: A) -> B

    lax.cond(pred: bool, true_body: A -> B, false_body: A -> B, operand: A) -> B

Both of these will bind a primitive called `cond` internally. The `cond` primitive in jaxprs reflects the more general signature of `lax.switch()`: it takes an integer denoting the index of the branch to execute (clamped into valid indexing range).

For example:

    from jax import lax

    def one_of_three(index, arg):
      return lax.switch(index, [lambda x: x + 1.,
                                lambda x: x - 2.,
                                lambda x: x + 3.],
                        arg)

    print(make_jaxpr(one_of_three)(1, 5.))

    { lambda ; a:i32[] b:f32[]. let
        c:i32[] = convert_element_type[new_dtype=int32 weak_type=False] a
        d:i32[] = clamp 0:i32[] c 2:i32[]
        e:f32[] = cond[
          branches=(
            { lambda ; f:f32[]. let g:f32[] = add f 1.0:f32[] in (g,) }
            { lambda ; h:f32[]. let i:f32[] = sub h 2.0:f32[] in (i,) }
            { lambda ; j:f32[]. let k:f32[] = add j 3.0:f32[] in (k,) }
          )
        ] d b
      in (e,) }

The `cond` primitive has a number of parameters:

- `branches` are jaxprs that correspond to the branch functionals. In this example, those functionals each take one input variable, corresponding to `x`.

- `linear` is a tuple of booleans that is used internally by the auto-differentiation machinery to encode which of the input parameters are used linearly in the conditional.

The above instance of the cond primitive takes two operands. The first one (`d`) is the branch index, then `b` is the operand (`arg`) to be passed to whichever jaxpr in `branches` is selected by the branch index.

Another example, using [`jax.lax.cond()`](_autosummary/jax.lax.cond.html#jax.lax.cond "jax.lax.cond"):

    from jax import lax

    def func7(arg):
      return lax.cond(arg >= 0.,
                      lambda xtrue: xtrue + 3.,
                      lambda xfalse: xfalse - 3.,
                      arg)

    print(make_jaxpr(func7)(5.))

    { lambda ; a:f32[]. let
        b:bool[] = ge a 0.0:f32[]
        c:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
        d:f32[] = cond[
          branches=(
            { lambda ; e:f32[]. let f:f32[] = sub e 3.0:f32[] in (f,) }
            { lambda ; g:f32[]. let h:f32[] = add g 3.0:f32[] in (h,) }
          )
        ] c a
      in (d,) }

In this case, the boolean predicate is converted to an integer index (0 or 1), and `branches` are jaxprs that correspond to the false and true branch functionals, in that order. Again, each function takes one input variable, corresponding to `xfalse` and `xtrue` respectively.

The following example shows a more complicated situation when the input to the branch functionals is a tuple, and the `false` branch functional contains a constant `jnp.ones(1)` that is hoisted as a `constvar`.

    def func8(arg1, arg2):  # Where `arg2` is a pair.
      return lax.cond(arg1 >= 0.,
                      lambda xtrue: xtrue[0],
                      lambda xfalse: jnp.array([1]) + xfalse[1],
                      arg2)

    print(make_jaxpr(func8)(5., (jnp.zeros(1), 2.)))

    { lambda a:i32[1]; b:f32[] c:f32[1] d:f32[]. let
        e:bool[] = ge b 0.0:f32[]
        f:i32[] = convert_element_type[new_dtype=int32 weak_type=False] e
        g:f32[1] = cond[
          branches=(
            { lambda ; h:i32[1] i:f32[1] j:f32[]. let
                k:f32[1] = convert_element_type[new_dtype=float32 weak_type=True] h
                l:f32[1] = add k j
              in (l,) }
            { lambda ; m:i32[1] n:f32[1] o:f32[]. let  in (n,) }
          )
        ] f a c d
      in (g,) }

### `while` primitive[\#](#while-primitive "Link to this heading")

Just like for conditionals, Python loops are inlined during tracing. If you want to capture a loop for dynamic execution, you must use one of several special operations, [`jax.lax.while_loop()`](_autosummary/jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop") (a primitive) and [`jax.lax.fori_loop()`](_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop "jax.lax.fori_loop") (a helper that generates a while_loop primitive):

    lax.while_loop(cond_fun: (C -> bool), body_fun: (C -> C), init: C) -> C
    lax.fori_loop(start: int, end: int, body: (int -> C -> C), init: C) -> C

In the above signature, `C` stands for the type of the loop “carry” value. For example, here is an example `fori_loop`:

    import numpy as np

    def func10(arg, n):
      ones = jnp.ones(arg.shape)  # A constant.
      return lax.fori_loop(0, n,
                           lambda i, carry: carry + ones * 3. + arg,
                           arg + ones)

    print(make_jaxpr(func10)(np.ones(16), 5))

    { lambda ; a:f32[16] b:i32[]. let
        c:f32[16] = broadcast_in_dim 1.0:f32[]
        d:f32[16] = add a c
        _:i32[] _:i32[] e:f32[16] = while[
          body_jaxpr={ lambda ; f:f32[16] g:f32[16] h:i32[] i:i32[] j:f32[16]. let
              k:i32[] = add h 1:i32[]
              l:f32[16] = mul f 3.0:f32[]
              m:f32[16] = add j l
              n:f32[16] = add m g
            in (k, i, n) }
          body_nconsts=2
          cond_jaxpr={ lambda ; o:i32[] p:i32[] q:f32[16]. let
              r:bool[] = lt o p
            in (r,) }
          cond_nconsts=0
        ] c a 0:i32[] b d
      in (e,) }

The `while` primitive takes 5 arguments: `c`` ``a`` ``0`` ``b`` ``d`, as follows:

- 0 constants for `cond_jaxpr` (since `cond_nconsts` is 0)

- 2 constants for `body_jaxpr` (`c`, and `a`)

- 3 parameters for the initial value of carry

### `scan` primitive[\#](#scan-primitive "Link to this heading")

JAX supports a special form of loop over the elements of an array (with statically known shape). The fact that there are a fixed number of iterations makes this form of looping easily reverse-differentiable. Such loops are constructed with the [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan") function:

    lax.scan(body_fun: (C -> A -> (C, B)), init_carry: C, in_arr: Array[A]) -> (C, Array[B])

This is written in terms of a [Haskell type signature](https://wiki.haskell.org/Type_signature): `C` is the type of the `scan` carry, `A` is the element type of the input array(s), and `B` is the element type of the output array(s).

For the example consider the function `func11` below:

    def func11(arr, extra):
      ones = jnp.ones(arr.shape)  #  A constant
      def body(carry, aelems):
        # carry: running dot-product of the two arrays
        # aelems: a pair with corresponding elements from the two arrays
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
      return lax.scan(body, 0., (arr, ones))

    print(make_jaxpr(func11)(np.ones(16), 5.))

    { lambda ; a:f32[16] b:f32[]. let
        c:f32[16] = broadcast_in_dim 1.0:f32[]
        d:f32[] e:f32[16] = scan[
          jaxpr={ lambda ; f:f32[] g:f32[] h:f32[] i:f32[]. let
              j:f32[] = mul h i
              k:f32[] = convert_element_type[new_dtype=float32 weak_type=False] g
              l:f32[] = add k j
              m:f32[] = convert_element_type[new_dtype=float32 weak_type=False] f
              n:f32[] = add l m
            in (n, g) }
          length=16
          num_carry=1
          num_consts=1
          reverse=False
          unroll=1
        ] b 0.0:f32[] a c
      in (d, e) }

The `linear` parameter describes for each of the input variables whether they are guaranteed to be used linearly in the body. Once the `scan` goes through linearization, more arguments will be linear.

The `scan` primitive takes 4 arguments: `b`` ``0.0`` ``a`` ``c`, of which:

- One is the free variable for the body

- One is the initial value of the carry

- The next 2 are the arrays over which the scan operates

### `(p)jit` primitive[\#](#p-jit-primitive "Link to this heading")

The call primitive arises from JIT compilation, and it encapsulates a sub-jaxpr along with parameters that specify the backend and the device on which the computation should run. For example:

    from jax import jit

    def func12(arg):
      @jit
      def inner(x):
        return x + arg * jnp.ones(1)  # Include a constant in the inner function.
      return arg + inner(arg - 2.)

    print(make_jaxpr(func12)(1.))

    { lambda ; a:f32[]. let
        b:f32[] = sub a 2.0:f32[]
        c:f32[1] = jit[
          name=inner
          jaxpr={ lambda ; a:f32[] b:f32[]. let
              d:f32[1] = broadcast_in_dim 1.0:f32[]
              e:f32[] = convert_element_type[new_dtype=float32 weak_type=False] a
              f:f32[1] = mul e d
              g:f32[] = convert_element_type[new_dtype=float32 weak_type=False] b
              c:f32[1] = add g f
            in (c,) }
        ] a b
        h:f32[] = convert_element_type[new_dtype=float32 weak_type=False] a
        i:f32[1] = add h c
      in (i,) }

[](jax-primitives.html "previous page")

previous

JAX Internals: primitives

[](jax.html "next page")

next

API Reference

Contents

- [`jax.core.ClosedJaxpr`](#jax-core-closedjaxpr)
- [Handling pytrees](#handling-pytrees)
- [Constant variables (vars)](#constant-variables-vars)
- [Higher-order JAX primitives](#higher-order-jax-primitives)
  - [`cond` primitive (conditionals)](#cond-primitive-conditionals)
  - [`while` primitive](#while-primitive)
  - [`scan` primitive](#scan-primitive)
  - [`(p)jit` primitive](#p-jit-primitive)

By The JAX authors

© Copyright 2024, The JAX Authors.\
