- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.StepTraceAnnotation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.StepTraceAnnotation.rst "Download source file")
-  .pdf

# jax.profiler.StepTraceAnnotation

## Contents

- [`StepTraceAnnotation`](#jax.profiler.StepTraceAnnotation)
  - [`StepTraceAnnotation.__init__()`](#jax.profiler.StepTraceAnnotation.__init__)

# jax.profiler.StepTraceAnnotation[\#](#jax-profiler-steptraceannotation "Link to this heading")

*class* jax.profiler.StepTraceAnnotation(*name*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L363-L387)[\#](#jax.profiler.StepTraceAnnotation "Link to this definition")  
Context manager that generates a step trace event in the profiler.

The step trace event spans the duration of the code enclosed by the context. The profiler will provide the performance analysis for each step trace event.

For example, it can be used to mark training steps and enable the profiler to provide the performance analysis per step:

    >>> while global_step < NUM_STEPS:                                           
    ...   with jax.profiler.StepTraceAnnotation("train", step_num=global_step):  
    ...     train_step()                                                         
    ...     global_step += 1                                                     

This will cause a “train xx” event to show up on the trace timeline if the event occurs while the process is being traced by TensorBoard. In addition, if using accelerators, the device trace timeline will also show a “train xx” event. Note that “step_num” can be set as a keyword argument to pass the global step number to the profiler.

Parameters:  
**name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

\_\_init\_\_(*self*, *arg0: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*, */*, *\*\*kwargs*) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L385-L387)[\#](#jax.profiler.StepTraceAnnotation.__init__ "Link to this definition")  
Parameters:  
**name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.profiler.StepTraceAnnotation.__init__ "jax.profiler.StepTraceAnnotation.__init__")(self, arg0, /, \*\*kwargs) |  |

Attributes

|                |     |
|----------------|-----|
| `is_enabled`   |     |
| `set_metadata` |     |

[](jax.profiler.TraceAnnotation.html "previous page")

previous

jax.profiler.TraceAnnotation

[](jax.profiler.register_subprocess.html "next page")

next

jax.profiler.register_subprocess

Contents

- [`StepTraceAnnotation`](#jax.profiler.StepTraceAnnotation)
  - [`StepTraceAnnotation.__init__()`](#jax.profiler.StepTraceAnnotation.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
