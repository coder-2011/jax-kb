- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- Fault Tolerant Distributed JAX

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/fault_tolerance.rst "Download source file")
-  .pdf

# Fault Tolerant Distributed JAX

## Contents

- [Part 1: Fault Tolerance Basics](#part-1-fault-tolerance-basics)
  - [Fault Intolerant By Default](#fault-intolerant-by-default)
  - [Surviving Faults](#surviving-faults)
  - [Getting Stuck in Collectives](#getting-stuck-in-collectives)
  - [Cancelling Collectives](#cancelling-collectives)
  - [Knowing Who’s Alive](#knowing-who-s-alive)
  - [Barrier Semantics](#barrier-semantics)
  - [Atomicity](#atomicity)
- [Part 2: Examples](#part-2-examples)
  - [Example 1: Fault Tolerant Data Parallel Training](#example-1-fault-tolerant-data-parallel-training)
  - [Example 2: Fault Tolerant Data Parallel Training With Recovery](#example-2-fault-tolerant-data-parallel-training-with-recovery)
- [Part 3: Implementation Details](#part-3-implementation-details)
  - [The Coordination Service](#the-coordination-service)
  - [Live Processes](#live-processes)
  - [Barrier Semantics](#id2)
  - [Formal Semantics](#formal-semantics)
  - [Atomicity](#id3)
  - [Cancelling Collectives](#id4)

# Fault Tolerant Distributed JAX[\#](#fault-tolerant-distributed-jax "Link to this heading")

Recall that [multi-controller JAX](https://docs.jax.dev/en/latest/multi_process.html) allows you to run a JAX program distributed across multiple machines. By default, if *any* of these machines fail, then *every* machine will fail. That is, multi-controller JAX is not **fault-tolerant** by default.

This article has three parts. In the first part, we’ll explain the basics of how to write fault tolerant multi-controller JAX programs. In the second part, we’ll show some example fault-tolerant multi-controller JAX programs. In the third part, we’ll take a look under the covers at how multi-controller JAX implements fault tolerance.

Warning

JAX’s support for fault tolerance is still experimental. It currently only works fully on GPUs. It has rough edges, is probably buggy, and is subject to change. Use at your own risk.

Note

If you’re looking for an alternative to fault-tolerant multi-controller JAX on TPU, consider [Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/pathways-intro).

## Part 1: Fault Tolerance Basics[\#](#part-1-fault-tolerance-basics "Link to this heading")

### Fault Intolerant By Default[\#](#fault-intolerant-by-default "Link to this heading")

By default, multi-controller JAX programs are not fault tolerant. If *any* process crashes, then *all* other processes will also intentionally crash. To make this concrete, consider the following trivial script, `example.py`, that initializes multi-controller JAX by calling `jax.distributed.initialize` and then enters an infinite loop.

`example.py`[\#](#id5 "Link to this code")

     1from absl import app
     2from absl import flags
     3from collections.abc import Sequence
     4import jax
     5import time
     6
     7_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
     8_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
     9
    10
    11def main(_: Sequence[str]) -> None:
    12  jax.distributed.initialize(
    13      coordinator_address="localhost:9000",
    14      num_processes=_NUM_PROCESSES.value,
    15      process_id=_PROCESS_ID.value,
    16      local_device_ids=[_PROCESS_ID.value],
    17      heartbeat_timeout_seconds=10,
    18  )
    19  print(f'{jax.devices()=}')
    20  print(f'{jax.local_devices()=}')
    21  while True:
    22    print(time.time())
    23    time.sleep(1)
    24
    25
    26if __name__ == "__main__":
    27  app.run(main)

Run `example.py` across four processes on a VM with four GPUs by running the following four commands, each in a different terminal. The `local_device_ids` argument to `jax.distributed.initialize` ensures each process is assigned only one of the four GPUs. We’ll explain the `heartbeat_timeout_seconds` argument in just a second.

    python example.py --i=0 --n=4  # in terminal 1
    python example.py --i=1 --n=4  # in terminal 2
    python example.py --i=2 --n=4  # in terminal 3
    python example.py --i=3 --n=4  # in terminal 4

When you run these commands, you’ll see the processes dutifully printing out the current time every second. Next, fail the fourth process: `pkill`` ``-9`` ``-f`` ``'python`` ``example.py`` ``--i=3`` ``--n=4'`. After about ten seconds, the other processes will also terminate and spit out error messages that look something like this:

    E0926 17:26:32.075402  157988 coordination_service_agent.cc:332] Polled an error from coordination service (this can be an error from this or another task).
    F0926 17:26:32.075587  157988 client.h:77] Terminating process because the JAX distributed service detected fatal errors. This most likely indicates that another task died; see the other task logs for more details. Disable Python buffering, i.e. `python -u`, to be sure to see all the previous output. absl::Status: UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats):
    /job:jax_worker/replica:0/task:3
    The tasks have crashed. Check the task logs for an earlier error, or scheduler events (e.g. preemption, eviction) to debug further.

    RPC: /tensorflow.CoordinationService/PollForError [type.googleapis.com/tensorflow.CoordinationServiceError='']

When a process in a multi-controller JAX program notices that a peer process has crashed, it decides to crash as well. The processes [share fate](https://en.wikipedia.org/wiki/Fate-sharing). The `heartbeat_timeout_seconds` argument to `jax.distributed.initialize` determines how long a process waits before concluding a peer process has died. The first three processes crash about ten seconds after you kill the fourth because we passed `heartbeat_timeout_seconds=10` as an argument to `jax.distributed.initialize`.

### Surviving Faults[\#](#surviving-faults "Link to this heading")

We can disable fate-sharing by adding the `--xla_gpu_nccl_terminate_on_error=false` flag and the `jax_enable_recoverability` configuration option to `example.py`, as shown below:

     1import os
     2os.environ['XLA_FLAGS'] = '--xla_gpu_nccl_terminate_on_error=false'
     3
     4from absl import app
     5from absl import flags
     6from collections.abc import Sequence
     7import jax
     8import time
     9
    10_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
    11_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
    12
    13
    14def main(_: Sequence[str]) -> None:
    15  jax.config.update("jax_enable_recoverability", True)
    16  jax.distributed.initialize(
    17      coordinator_address="localhost:9000",
    18      num_processes=_NUM_PROCESSES.value,
    19      process_id=_PROCESS_ID.value,
    20      local_device_ids=[_PROCESS_ID.value],
    21      heartbeat_timeout_seconds=10,
    22  )
    23  print(f'{jax.devices()=}')
    24  print(f'{jax.local_devices()=}')
    25  while True:
    26    print(time.time())
    27    time.sleep(1)
    28
    29
    30if __name__ == "__main__":
    31  app.run(main)

Again run the script across four processes and then kill the fourth. Notice that now, the other three processes happily continue executing.

Next try failing process 0. Notice that all four processes terminate with error messages that look something like the following:

    E0929 17:42:48.594192 1044529 coordination_service_agent.cc:332] Polled an error from coordination service (this can be an error from this or another task).
    F0929 17:42:48.594200 1044529 client.h:77] Terminating process because the JAX distributed service detected fatal errors. This most likely indicates that another task died; see the other task logs for more details. Disable Python buffering, i.e. `python -u`, to be sure to see all the previous output. absl::Status: UNAVAILABLE: Failed to send RPC to coordination service. Either the leader task was preempted/died/restarted unexpectedly or this task is experiencing network issues. Check earlier logs from 1) this task, 2) the leader (usually slice 0 task 0), and 3) cluster scheduler to debug further.
    Additional GRPC error information from remote target coordination_service while calling /tensorflow.CoordinationService/PollForError:
    :UNKNOWN:Error received from peer  {grpc_message:"Socket closed", grpc_status:14}

Process 0 is special. If process 0 fails, every process will fail, even with fate-sharing disabled. Why? Process 0 runs an RPC service called the coordination service that all processes use to coordination with each other. If the coordination service fails, all other processes have no choice but to fail. See [Part 3: Implementation Details](#part3) for more details.

### Getting Stuck in Collectives[\#](#getting-stuck-in-collectives "Link to this heading")

`example.py` is now able to survive faults, but the processes do not communicate with each other at all. Any realistic multi-controller JAX program would involve communication between the processes (otherwise, what’s the point of using multi-controller JAX?). Let’s edit `example.py` so that the processes perform a collective `jnp.sum` in every iteration of the loop.

     1import os
     2os.environ['XLA_FLAGS'] = '--xla_gpu_nccl_terminate_on_error=false'
     3
     4from absl import app
     5from absl import flags
     6from collections.abc import Sequence
     7import jax
     8import jax.numpy as jnp
     9import time
    10
    11_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
    12_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
    13
    14
    15def main(_: Sequence[str]) -> None:
    16  jax.config.update("jax_enable_recoverability", True)
    17  jax.distributed.initialize(
    18      coordinator_address="localhost:9000",
    19      num_processes=_NUM_PROCESSES.value,
    20      process_id=_PROCESS_ID.value,
    21      local_device_ids=[_PROCESS_ID.value],
    22      heartbeat_timeout_seconds=10,
    23  )
    24  print(f'{jax.devices()=}')
    25  print(f'{jax.local_devices()=}')
    26
    27  n = jax.device_count()
    28  jax.set_mesh(jax.make_mesh((n,), ("i",)))
    29  x = jax.device_put(jnp.arange(n), jax.P("i"))
    30  while True:
    31    print(jnp.sum(x))
    32    time.sleep(1)
    33
    34
    35if __name__ == "__main__":
    36  app.run(main)

In the highlighted code above, the processes create an array `x` sharded across the four processes and then perform a distributed `jnp.sum`. Again run the program and fail the fourth process. You’ll notice that the first three process do not crash, but they do get *stuck*. By default, if a process fails while participating in a distributed computation (like `jnp.sum`), then the rest of the processes participating in the computation will get stuck *forever*.

### Cancelling Collectives[\#](#cancelling-collectives "Link to this heading")

We can avoid getting stuck by cancelling collectives with a failed participant. We can enable collective cancelling by providing a few more flags and environment variables, highlighted below.

     1import os
     2os.environ['XLA_FLAGS'] = ' '.join([
     3  '--xla_gpu_nccl_terminate_on_error=false',
     4  '--xla_gpu_nccl_async_execution=true',
     5  '--xla_gpu_nccl_blocking_communicators=false',
     6])
     7os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
     8os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'
     9
    10from absl import app
    11from absl import flags
    12from collections.abc import Sequence
    13import jax
    14import jax.numpy as jnp
    15import time
    16
    17_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
    18_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
    19
    20
    21def main(_: Sequence[str]) -> None:
    22  jax.config.update("jax_enable_recoverability", True)
    23  jax.distributed.initialize(
    24      coordinator_address="localhost:9000",
    25      num_processes=_NUM_PROCESSES.value,
    26      process_id=_PROCESS_ID.value,
    27      local_device_ids=[_PROCESS_ID.value],
    28      heartbeat_timeout_seconds=10,
    29  )
    30  print(f'{jax.devices()=}')
    31  print(f'{jax.local_devices()=}')
    32
    33  # Don't do this. Use live_devices instead.
    34  from jax.experimental.multihost_utils import _live_devices
    35  _live_devices(jax._src.distributed.global_state.client, jax.devices())
    36
    37  n = jax.device_count()
    38  jax.set_mesh(jax.make_mesh((n,), ("i",)))
    39  x = jax.device_put(jnp.arange(n), jax.P("i"))
    40  while True:
    41    print(jnp.sum(x))
    42    time.sleep(1)
    43
    44
    45if __name__ == "__main__":
    46  app.run(main)

We also need to insert a call to `jax.experimental.multihost_utils._live_devices` to make the script work. You should normally not do this. You should instead use the `live_devices` API that we’ll introduce momentarily. For now, `_live_devices` is a hack to get the script working before we explain the proper API.

Again run the script and fail the fourth process. The first three processes will be stuck in their call to `jnp.sum`, but after about ten seconds, the call will be cancelled and `jnp.sum` will raise an exception that looks something like this:

    jaxlib._jax.XlaRuntimeError: FAILED_PRECONDITION: Task with incarnation id 3446767950926952685 is not connected

### Knowing Who’s Alive[\#](#knowing-who-s-alive "Link to this heading")

After a process dies, the remaining *alive* procesess need to learn who is dead and who is alive. For this, we can use the core JAX fault tolerance API: `live_devices`. `live_devices` is a context manager that takes a list of devices as an argument and returns the subset of these devices that are alive. Below, we edit `example.py` to call `live_devices`.

     1import os
     2os.environ['XLA_FLAGS'] = ' '.join([
     3  '--xla_gpu_nccl_terminate_on_error=false',
     4  '--xla_gpu_nccl_async_execution=true',
     5  '--xla_gpu_nccl_blocking_communicators=false',
     6])
     7os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
     8os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'
     9
    10from absl import app
    11from absl import flags
    12from collections.abc import Sequence
    13from jax.experimental.multihost_utils import live_devices
    14import jax
    15import jax.numpy as jnp
    16import time
    17
    18_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
    19_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
    20
    21
    22def main(_: Sequence[str]) -> None:
    23  jax.config.update("jax_enable_recoverability", True)
    24  jax.distributed.initialize(
    25      coordinator_address="localhost:9000",
    26      num_processes=_NUM_PROCESSES.value,
    27      process_id=_PROCESS_ID.value,
    28      local_device_ids=[_PROCESS_ID.value],
    29      heartbeat_timeout_seconds=10,
    30  )
    31  print(f'{jax.devices()=}')
    32  print(f'{jax.local_devices()=}')
    33
    34  while True:
    35    try:
    36      with live_devices(jax.devices()) as devices:
    37        print(f'{devices=}')
    38        n = len(devices)
    39        jax.set_mesh(jax.make_mesh((n,), ("i",), devices=devices))
    40        x = jax.device_put(jnp.arange(n), jax.P("i"))
    41        print(jnp.sum(x))
    42    except Exception as e:
    43      print('FAIL:', e)
    44    else:
    45      print('PASS')
    46    time.sleep(1)
    47
    48
    49if __name__ == "__main__":
    50  app.run(main)

In the highlighted code above, we call `live_devices` with all devices (`jax.devices()`) to get the set `devices` of live devices. We then shard array `x` over these devices and perform a `jnp.sum`. If a process fails while executing the `jnp.sum`, then `jnp.sum` will be cancelled and raise an exception on the remaining live devices. Technically, the collective is not guaranteed to fail. We’ll revisit this in [Atomicity](#atomicity). For now, assume it will fail.

Note

`jax.devices()` always returns the set of *all* devices, even if some of these devices are on failed processes. Use `jax.experimental.multihost_utils.live_devices` to learn which of these devices are live.

Again run the script and fail the fourth process. Notice that the remaining three alive processes catch the exception raised by `jnp.sum` and continue to the next iteration of the while loop. In this next iteration, `devices` does not include the device on the failed fourth process. The three alive processes continue to execute correctly even though the fourth process is dead.

Next, restart the fourth process. Notice that after the fourth process restarts, its device is again included in the set of alive devices returned by `live_devices`. All four processes then continue executing normally.

At first blush, `live_devices` seems trivial. You give it a list of devices, and it returns the ones that are alive. How complicated can that be? Unfortunately, as with [many things in distributed systems](https://en.wikipedia.org/wiki/Fallacies_of_distributed_computing), there are a lot subtleties to iron out. Next, we explain the **barrier** semantics and **atomicity** properties of `live_devices`.

### Barrier Semantics[\#](#barrier-semantics "Link to this heading")

Recall that every process in a [multi-controller JAX](https://docs.jax.dev/en/latest/multi_process.html) program should run in lockstep. The processes should execute the same instructions in the same order. Failing to do so will *almost certainly* lead to deadlocks, crashes, or anomalous behavior.

In the context of `live_devices`, we need to ensure that every process agrees on which processes are currently alive. This is difficult to ensure because every process is executing independently at potentially different speeds and processes can fail at any time. Consider again the `example.py` script from above running on four processes. Imagine process 1 and 2 call `live_devices`, then process 4 fails, and then process 3 calls `live_devices`. Process 1 and 2 might think process 4 is alive while process 3 thinks it is dead.

To avoid situations like these, `live_devices` guarantees that it returns the same set of live devices to every process. It accomplishes this using a barrier. A call to `live_devicess(devices)` blocks until every live process hosting a device in `devices` has also called `live_devices`. Once every live process is in the `live_devices` barrier, `live_devices` returns the same set of live devices to every process.

Important

`live_devices` uses a barrier to ensure that it will *always* return the same set of live devices to every live process.

Because `live_devices` implements a barrier it is susceptible to deadlock if used improperly. We recommend only having a single `with`` ``live_devices` block in a program. Multiple calls to `live_devices` is hard to reason about and can lead to deadlock.

See [Part 3: Implementation Details](#part3) for details on how the `live_devices` barrier is implemented as well as a formal semantics based on [linearizability](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf).

### Atomicity[\#](#atomicity "Link to this heading")

A distributed computation is **atomic** if every participant in the computation agrees on whether the operation succeeds or fails. In the `example.py` script above, we saw that when a process failed during the execution of a `jnp.sum`, then `jnp.sum` would abort and raise an exception on the remaining live processes. So `jnp.sum` is atomic?

Unfortunately, it’s not.

When a process fails during the execution of a collective operation (like `jnp.sum`), the remaining processes may cancel the operation and raise an exception or they may complete the operation successfully. Collective operations in JAX do not have any inherent atomicity properties.

If collective operations are not atomic, however, then multi-controller JAX processes might diverge. For example, if a process fails during a training step of a machine learning model, some processes might detect the failure and roll the model back to a checkpoint while other processes might think the step succeeded and keep training.

To avoid the complexities of non-atomic execution, `live_devices` provides its own atomicity guarantees despite the fact that collectives are not atomic. Specifically, the body of a `with`` ``live_devices` block is guaranteed to either complete successfully on all processes or raise an exception on all processes. More concretely, if we consider the code snippet below, either every process executes branch A or every process executes branch B. It is impossible for some processes to execute A while others execute B.

    try:
      with live_devices(jax.devices()) as devices:
        ...
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B

Warning

A `with`` ``live_devices` block does not guarantee atomicity if the code block non-deterministically raises exceptions for reasons other than collectives that fail because of a crashed process. For example, if one process raises an exception because it runs out of memory, this exception will not be propagated to the other processes.

Recall that JAX uses [asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html). Operations like `jnp.sum` do not block until the operation is complete. Instead, they return `jax.Arrays` that act as futures. This asynchrony can interact with `live_devices` in unexpected ways. For example, consider the following code that performs a `jnp.sum`, assigns the result to `y`, and then prints `y`:

    x = ...
    y = ...
    try:
      with live_devices(jax.devices()) as devices:
        y = jnp.sum(x)
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B
    print(y)

Imagine that the `with`` ``live_devices` block executes successfully on all processes. That is, all processes execute branch B. This only guarantees that every process successfully created a future and assigned it to `y`. The actual computation of the `jnp.sum` may be delayed until outside the block. Thus, some processes might successfully complete the `jnp.sum` and print the value of `y` while other processes fail to complete the `jnp.sum` and raise an exception when trying to print `y`.

To avoid this, use `jax.block_until_ready` to ensure that computations are performed within the `with`` ``live_devices` block. The code snippet below, which now calls `jax.block_until_ready` when assigning to `y`, guarantees that every process will successfully execute the `jnp.sum` or every process will raise an exception.

    x = ...
    y = ...
    try:
      with live_devices(jax.devices()) as devices:
        y = jax.block_until_ready(jnp.sum(x))
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B
    print(y)

See [Part 3: Implementation Details](#part3) for details on how atomicity is implemented.

## Part 2: Examples[\#](#part-2-examples "Link to this heading")

`live_devices` is not a panacea; it is a tool. It does not magically make multi-controller JAX programs fault tolerant. Rather, it allows you to implement fault tolerance yourself in the way that is best for your application.

The exact details of how you implement fault-tolerance will vary greatly based on the nature of your application. In this section, we present some examples of how to use `live_devices`. The examples are meant to be illustrative but not prescriptive. There are many other ways to implement fault tolerance.

### Example 1: Fault Tolerant Data Parallel Training[\#](#example-1-fault-tolerant-data-parallel-training "Link to this heading")

In this example, we train a trivial single-parameter linear model (\\y = \alpha x\\) with data parallelism across four processes. The example is contrived—you would never train a model with a single parameter across four machines—but we intentionally keep the model simple to focus on fault tolerance.

Data parallelism makes implementing fault tolerance relatively straightforward. Because every process has a full copy of the model weights, if a process fails, we can simply ignore it and continue training. This example tolerates an arbitrary number of process failures (excluding process 0), but once a process fails, we assume it does not recover. The next example shows how to handle process recovery.

First, we set some flags to disable fate-sharing and enable collective cancelling. We also make the necessary imports and define some flags.

     1import os
     2os.environ['XLA_FLAGS'] = ' '.join([
     3    '--xla_gpu_nccl_terminate_on_error=false',
     4    '--xla_gpu_nccl_async_execution=true',
     5    '--xla_gpu_nccl_blocking_communicators=false',
     6])
     7os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
     8os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'
     9
    10from absl import app
    11from absl import flags
    12from collections.abc import Sequence
    13from jax.experimental.multihost_utils import live_devices
    14import jax
    15import jax.numpy as jnp
    16import time
    17
    18_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
    19_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")

Next, we define a `replicated` function that returns an array replicated across a set of devices. Note that `replicated` doesn’t actually move any data. It assumes the argument `x` already has equal value across all processes. It simply returns a new view of that data, in a process-spanning jax.Array with a replicated sharding.

    21def replicated(x: jax.Array, devices: list[jax.Device]):
    22  """Return x replicated across the provided devices.
    23
    24  Note that replicated(x) doesn't actually move any data. It simply creates a
    25  logically replicated array with x as the local replica.
    26  """
    27  n = len(devices)
    28  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
    29  spec = jax.sharding.PartitionSpec(None)
    30  sharding = jax.sharding.NamedSharding(mesh, spec)
    31  shards = [
    32      jax.device_put(x.addressable_shards[0].data, d) for d in devices
    33      if d.process_index == jax.process_index()
    34  ]
    35  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)

We define a similar `sharded` function that returns an array sharded across a set of devices. Again, `sharded` is not actually moving any data between processes.

    38def sharded(x: jax.Array, devices: list[jax.Device]):
    39  """Return x sharded across the provided devices.
    40
    41  Note that sharded(x) doesn't actually move any data. It simply creates a
    42  logically sharded array. x should have the same shape as the global array.
    43  """
    44  n = len(devices)
    45  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
    46  spec = jax.sharding.PartitionSpec("i")
    47  sharding = jax.sharding.NamedSharding(mesh, spec)
    48  m = sharding.addressable_devices_indices_map(x.shape)
    49  shards = [jax.device_put(x[m[d]], d) for d in jax.local_devices()]
    50  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)

Now, we’re ready to start writing our training loop. We begin by initializing multi-controller JAX by calling `jax.distributed.initialize`.

    53def main(_: Sequence[str]) -> None:
    54  # Parse command line arguments and initialize multi-controller JAX.
    55  jax.config.update("jax_enable_recoverability", True)
    56  jax.distributed.initialize(coordinator_address="localhost:8000",
    57                             process_id=_PROCESS_ID.value,
    58                             num_processes=_NUM_PROCESSES.value,
    59                             local_device_ids=[_PROCESS_ID.value],
    60                             heartbeat_timeout_seconds=10)
    61  print(f'{jax.devices()=}')
    62  print(f'{jax.local_devices()=}')

Then, we define our simple linear model, generate some random training data, and initialize some basic hyperparameters.

    64  # Initialize the model's weights.
    65  keys = iter(jax.random.split(jax.random.key(seed=42), num=3))
    66  weights = jax.random.normal(next(keys), shape=(1, ))
    67
    68  # We'll learn a trivial linear model: a*x.
    69  def predict(weights, X):
    70    return weights * X
    71
    72  # We'll use mean squared error loss.
    73  def loss(weights, X, Y):
    74    return jnp.mean((predict(weights, X) - Y)**2)
    75
    76  # Initialize the (noisy) training data with a=10.
    77  X = jax.random.permutation(next(keys), jnp.arange(-300., 300.))
    78  Y = 10 * X + jax.random.normal(next(keys), X.shape)
    79
    80  # Hyperparameters.
    81  loss_and_grad = jax.jit(jax.value_and_grad(loss))
    82  learning_rate = 1e-6
    83  device_batch_size = 10

Finally, we enter the main training loop.

     85  step = 0
     86  while True:
     87    try:
     88      with live_devices(jax.devices()) as devices:
     89        print(f'=== Running step {step} with live devices = {devices} ===')
     90
     91        # Replicate the model weights.
     92        weights = replicated(weights, devices)
     93
     94        # Shard the batch.
     95        batch_size = device_batch_size * len(devices)
     96        start = (step * batch_size) % len(X)
     97        stop = start + batch_size
     98        X_batch = sharded(X[start:stop], devices)
     99        Y_batch = sharded(Y[start:stop], devices)
    100
    101        # Compute gradients and update weights.
    102        l, grad = loss_and_grad(weights, X_batch, Y_batch)
    103        new_weights = jax.block_until_ready(weights - learning_rate * grad)
    104    except Exception as e:
    105      print(f'Step {step} failed: {e}')
    106    else:
    107      print(f'Step {step} succeeded: loss = {l}')
    108      step += 1
    109      weights = new_weights
    110
    111    time.sleep(1)

- Every iteration of the loop, we call `live_devices` to learn which devices are currently alive.

- We then ensure that the model weights are replicated across these devices and ensure that the training data is sharded across these devices. Note that this doesn’t actually move any data between the devices; it simply creates JAX arrays with the appropriate replication and sharding metadata.

- We call `loss_and_grad` to compute the gradient of the weights with respect to the current batch of data and then compute the new weights. Notice that we assign the new weights to `new_weights` rather than assigning to `weights` in case the training step fails. We also call `jax.block_until_ready` to ensure that every process has computed the new weights when we exit the `live_devices` block.

- If no processes failed during the execution of the training step, then the `else` branch is taken. The step is incremented, and `weights` is updated. Otherwise, an exception will be raised and the `except` branch is taken. In this case, we do not update `step` or `weights` and retry the step on the next iteration with the new set of live devices.

Here is the full example:

      1import os
      2os.environ['XLA_FLAGS'] = ' '.join([
      3    '--xla_gpu_nccl_terminate_on_error=false',
      4    '--xla_gpu_nccl_async_execution=true',
      5    '--xla_gpu_nccl_blocking_communicators=false',
      6])
      7os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
      8os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'
      9
     10from absl import app
     11from absl import flags
     12from collections.abc import Sequence
     13from jax.experimental.multihost_utils import live_devices
     14import jax
     15import jax.numpy as jnp
     16import time
     17
     18_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
     19_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
     20
     21def replicated(x: jax.Array, devices: list[jax.Device]):
     22  """Return x replicated across the provided devices.
     23
     24  Note that replicated(x) doesn't actually move any data. It simply creates a
     25  logically replicated array with x as the local replica.
     26  """
     27  n = len(devices)
     28  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
     29  spec = jax.sharding.PartitionSpec(None)
     30  sharding = jax.sharding.NamedSharding(mesh, spec)
     31  shards = [
     32      jax.device_put(x.addressable_shards[0].data, d) for d in devices
     33      if d.process_index == jax.process_index()
     34  ]
     35  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
     36
     37
     38def sharded(x: jax.Array, devices: list[jax.Device]):
     39  """Return x sharded across the provided devices.
     40
     41  Note that sharded(x) doesn't actually move any data. It simply creates a
     42  logically sharded array. x should have the same shape as the global array.
     43  """
     44  n = len(devices)
     45  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
     46  spec = jax.sharding.PartitionSpec("i")
     47  sharding = jax.sharding.NamedSharding(mesh, spec)
     48  m = sharding.addressable_devices_indices_map(x.shape)
     49  shards = [jax.device_put(x[m[d]], d) for d in jax.local_devices()]
     50  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
     51
     52
     53def main(_: Sequence[str]) -> None:
     54  # Parse command line arguments and initialize multi-controller JAX.
     55  jax.config.update("jax_enable_recoverability", True)
     56  jax.distributed.initialize(coordinator_address="localhost:8000",
     57                             process_id=_PROCESS_ID.value,
     58                             num_processes=_NUM_PROCESSES.value,
     59                             local_device_ids=[_PROCESS_ID.value],
     60                             heartbeat_timeout_seconds=10)
     61  print(f'{jax.devices()=}')
     62  print(f'{jax.local_devices()=}')
     63
     64  # Initialize the model's weights.
     65  keys = iter(jax.random.split(jax.random.key(seed=42), num=3))
     66  weights = jax.random.normal(next(keys), shape=(1, ))
     67
     68  # We'll learn a trivial linear model: a*x.
     69  def predict(weights, X):
     70    return weights * X
     71
     72  # We'll use mean squared error loss.
     73  def loss(weights, X, Y):
     74    return jnp.mean((predict(weights, X) - Y)**2)
     75
     76  # Initialize the (noisy) training data with a=10.
     77  X = jax.random.permutation(next(keys), jnp.arange(-300., 300.))
     78  Y = 10 * X + jax.random.normal(next(keys), X.shape)
     79
     80  # Hyperparameters.
     81  loss_and_grad = jax.jit(jax.value_and_grad(loss))
     82  learning_rate = 1e-6
     83  device_batch_size = 10
     84
     85  step = 0
     86  while True:
     87    try:
     88      with live_devices(jax.devices()) as devices:
     89        print(f'=== Running step {step} with live devices = {devices} ===')
     90
     91        # Replicate the model weights.
     92        weights = replicated(weights, devices)
     93
     94        # Shard the batch.
     95        batch_size = device_batch_size * len(devices)
     96        start = (step * batch_size) % len(X)
     97        stop = start + batch_size
     98        X_batch = sharded(X[start:stop], devices)
     99        Y_batch = sharded(Y[start:stop], devices)
    100
    101        # Compute gradients and update weights.
    102        l, grad = loss_and_grad(weights, X_batch, Y_batch)
    103        new_weights = jax.block_until_ready(weights - learning_rate * grad)
    104    except Exception as e:
    105      print(f'Step {step} failed: {e}')
    106    else:
    107      print(f'Step {step} succeeded: loss = {l}')
    108      step += 1
    109      weights = new_weights
    110
    111    time.sleep(1)
    112
    113
    114if __name__ == "__main__":
    115  app.run(main)

### Example 2: Fault Tolerant Data Parallel Training With Recovery[\#](#example-2-fault-tolerant-data-parallel-training-with-recovery "Link to this heading")

Now, we modify the example above to allow failed processes to recover. When a process recovers, it needs to receive the current step and model weights. Because we assume process 0 never fails—recall that if process 0 fails, every process will fail—we have process 0 send the current step and weights to recovering processes.

First, we define `send` and `recv` functions that use a `shard_map` to send data from one device to another. The sender calls `send`, and the receiver calls `recv`.

    55def send(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
    56  """Sends x from one device to another."""
    57  assert isinstance(x, jax.Array)
    58  devices = [from_device, to_device]
    59  psum = lambda x: jax.lax.psum(x, "i")
    60  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
    61  spec = jax.sharding.PartitionSpec(None)
    62  x = replicated(x, [from_device, to_device])
    63  shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)
    64
    65
    66def recv(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
    67  """Receives x from a matching send."""
    68  assert isinstance(x, jax.Array)
    69  to_device = jax.local_devices()[0]
    70  devices = [from_device, to_device]
    71  psum = lambda x: jax.lax.psum(x, "i")
    72  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
    73  spec = jax.sharding.PartitionSpec(None)
    74  x = jnp.zeros_like(x)
    75  x = replicated(x, [from_device, to_device])
    76  return shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)

`allgather` performs an AllGather of a single float across a set of devices.

    79def allgather(x: float, devices: list[jax.Device]) -> list[float]:
    80  """Performs an AllGather across the provided devices."""
    81  n = len(devices)
    82  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
    83  spec = jax.sharding.PartitionSpec('i')
    84  p = lambda x: jax.lax.all_gather(x, "i", tiled=True)
    85  f = jax.shard_map(p, mesh=mesh, in_specs=spec, out_specs=spec)
    86  return jax.block_until_ready(f(np.array([x] * len(devices)))).addressable_shards[0].data

Finally, we modify the training loop to handle recovering processes, as shown in the highlighted code below.

    121  step = 0
    122  while True:
    123    try:
    124      with live_devices(jax.devices()) as devices:
    125        print(f'=== Running step {step} with live devices = {devices} ===')
    126
    127        # Handle recovering devices. A device is recovering if its step doesn't
    128        # match process 0's step. We assume process 0 never fails.
    129        print('all gathering steps...')
    130        steps = allgather(step, devices)
    131        print(f'{steps=}')
    132        recovering = [d for d, s in zip(devices, steps) if s != steps[0]]
    133        for d in recovering:
    134          # Process 0 sends weights and step to the recovering devices.
    135          if jax.process_index() == 0:
    136            print('sending...')
    137            send(weights, jax.devices()[0], d)
    138            send(jnp.array([step]), jax.devices()[0], d)
    139          elif d.process_index == jax.process_index():
    140            print('receiving...')
    141            weights = recv(weights, jax.devices()[0], d)
    142            step = recv(jnp.array([step]), jax.devices()[0], d)[0]
    143
    144        # Replicate the model weights.
    145        weights = replicated(weights, devices)
    146
    147        # Shard the batch.
    148        batch_size = device_batch_size * len(devices)
    149        start = (step * batch_size) % len(X)
    150        stop = start + batch_size
    151        X_batch = sharded(X[start:stop], devices)
    152        Y_batch = sharded(Y[start:stop], devices)
    153
    154        # Compute gradients and update weights.
    155        l, grad = loss_and_grad(weights, X_batch, Y_batch)
    156        new_weights = jax.block_until_ready(weights - learning_rate * grad)
    157    except Exception as e:
    158      print(f'Step {step} failed: {e}')
    159    else:
    160      print(f'Step {step} succeeded: loss = {l}')
    161      step += 1
    162      weights = new_weights
    163
    164    time.sleep(1)

Recovery is a two-step process. First, we need to detect which processes are recovering. Second, we need process 0 to send the step and weights to the recovering processes.

1.  To detect which processes are recovering, we perform an AllGather on all live processes’ steps. When a failed process recovers, its `step` will be `0`, while the `step` on process `0` will be some positive number, so if a process’ step is not equal to process 0’s step, then it is recovering.

2.  Then, we call the `send` and `recv` functions we defined above to transfer the current step and model weights from process 0 to the recovering processes.

Here is the full example:

      1import os
      2os.environ['XLA_FLAGS'] = ' '.join([
      3    '--xla_gpu_nccl_terminate_on_error=false',
      4    '--xla_gpu_nccl_async_execution=true',
      5    '--xla_gpu_nccl_blocking_communicators=false',
      6])
      7os.environ['XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE'] = '1'
      8os.environ['XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT'] = '1'
      9
     10from absl import app
     11from absl import flags
     12from collections.abc import Sequence
     13from jax.experimental.multihost_utils import live_devices
     14from jax.experimental import shard_map
     15import jax
     16import jax.numpy as jnp
     17import numpy as np
     18import time
     19
     20_PROCESS_ID = flags.DEFINE_integer("i", -1, "Process id")
     21_NUM_PROCESSES = flags.DEFINE_integer("n", -1, "Number of processes")
     22
     23def replicated(x: jax.Array, devices: list[jax.Device]):
     24  """Return x replicated across the provided devices.
     25
     26  Note that replicated(x) doesn't actually move any data. It simply creates a
     27  logically replicated array with x as the local replica.
     28  """
     29  n = len(devices)
     30  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
     31  spec = jax.sharding.PartitionSpec(None)
     32  sharding = jax.sharding.NamedSharding(mesh, spec)
     33  shards = [
     34      jax.device_put(x.addressable_shards[0].data, d) for d in devices
     35      if d.process_index == jax.process_index()
     36  ]
     37  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
     38
     39
     40def sharded(x: jax.Array, devices: list[jax.Device]):
     41  """Return x sharded across the provided devices.
     42
     43  Note that sharded(x) doesn't actually move any data. It simply creates a
     44  logically sharded array. x should have the same shape as the global array.
     45  """
     46  n = len(devices)
     47  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
     48  spec = jax.sharding.PartitionSpec("i")
     49  sharding = jax.sharding.NamedSharding(mesh, spec)
     50  m = sharding.addressable_devices_indices_map(x.shape)
     51  shards = [jax.device_put(x[m[d]], d) for d in jax.local_devices()]
     52  return jax.make_array_from_single_device_arrays(x.shape, sharding, shards)
     53
     54
     55def send(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
     56  """Sends x from one device to another."""
     57  assert isinstance(x, jax.Array)
     58  devices = [from_device, to_device]
     59  psum = lambda x: jax.lax.psum(x, "i")
     60  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
     61  spec = jax.sharding.PartitionSpec(None)
     62  x = replicated(x, [from_device, to_device])
     63  shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)
     64
     65
     66def recv(x: jax.Array, from_device: jax.Device, to_device: jax.Device):
     67  """Receives x from a matching send."""
     68  assert isinstance(x, jax.Array)
     69  to_device = jax.local_devices()[0]
     70  devices = [from_device, to_device]
     71  psum = lambda x: jax.lax.psum(x, "i")
     72  mesh = jax.make_mesh((2, ), ("i", ), devices=devices)
     73  spec = jax.sharding.PartitionSpec(None)
     74  x = jnp.zeros_like(x)
     75  x = replicated(x, [from_device, to_device])
     76  return shard_map.shard_map(psum, mesh=mesh, in_specs=spec, out_specs=spec)(x)
     77
     78
     79def allgather(x: float, devices: list[jax.Device]) -> list[float]:
     80  """Performs an AllGather across the provided devices."""
     81  n = len(devices)
     82  mesh = jax.make_mesh((n, ), ("i", ), devices=devices)
     83  spec = jax.sharding.PartitionSpec('i')
     84  p = lambda x: jax.lax.all_gather(x, "i", tiled=True)
     85  f = jax.shard_map(p, mesh=mesh, in_specs=spec, out_specs=spec)
     86  return jax.block_until_ready(f(np.array([x] * len(devices)))).addressable_shards[0].data
     87
     88
     89def main(_: Sequence[str]) -> None:
     90  # Parse command line arguments and initialize multi-controller JAX.
     91  jax.config.update("jax_enable_recoverability", True)
     92  jax.distributed.initialize(coordinator_address="localhost:8000",
     93                             process_id=_PROCESS_ID.value,
     94                             num_processes=_NUM_PROCESSES.value,
     95                             local_device_ids=[_PROCESS_ID.value],
     96                             heartbeat_timeout_seconds=10)
     97  print(f'{jax.devices()=}')
     98  print(f'{jax.local_devices()=}')
     99
    100  # Initialize the model's weights.
    101  keys = iter(jax.random.split(jax.random.key(seed=42), num=3))
    102  weights = jax.random.normal(next(keys), shape=(1, ))
    103
    104  # We'll learn a trivial linear model: a*x.
    105  def predict(weights, X):
    106    return weights * X
    107
    108  # We'll use mean squared error loss.
    109  def loss(weights, X, Y):
    110    return jnp.mean((predict(weights, X) - Y)**2)
    111
    112  # Initialize the (noisy) training data with a=10.
    113  X = jax.random.permutation(next(keys), jnp.arange(-300., 300.))
    114  Y = 10 * X + jax.random.normal(next(keys), X.shape)
    115
    116  # Hyperparameters.
    117  loss_and_grad = jax.jit(jax.value_and_grad(loss))
    118  learning_rate = 1e-6
    119  device_batch_size = 10
    120
    121  step = 0
    122  while True:
    123    try:
    124      with live_devices(jax.devices()) as devices:
    125        print(f'=== Running step {step} with live devices = {devices} ===')
    126
    127        # Handle recovering devices. A device is recovering if its step doesn't
    128        # match process 0's step. We assume process 0 never fails.
    129        print('all gathering steps...')
    130        steps = allgather(step, devices)
    131        print(f'{steps=}')
    132        recovering = [d for d, s in zip(devices, steps) if s != steps[0]]
    133        for d in recovering:
    134          # Process 0 sends weights and step to the recovering devices.
    135          if jax.process_index() == 0:
    136            print('sending...')
    137            send(weights, jax.devices()[0], d)
    138            send(jnp.array([step]), jax.devices()[0], d)
    139          elif d.process_index == jax.process_index():
    140            print('receiving...')
    141            weights = recv(weights, jax.devices()[0], d)
    142            step = recv(jnp.array([step]), jax.devices()[0], d)[0]
    143
    144        # Replicate the model weights.
    145        weights = replicated(weights, devices)
    146
    147        # Shard the batch.
    148        batch_size = device_batch_size * len(devices)
    149        start = (step * batch_size) % len(X)
    150        stop = start + batch_size
    151        X_batch = sharded(X[start:stop], devices)
    152        Y_batch = sharded(Y[start:stop], devices)
    153
    154        # Compute gradients and update weights.
    155        l, grad = loss_and_grad(weights, X_batch, Y_batch)
    156        new_weights = jax.block_until_ready(weights - learning_rate * grad)
    157    except Exception as e:
    158      print(f'Step {step} failed: {e}')
    159    else:
    160      print(f'Step {step} succeeded: loss = {l}')
    161      step += 1
    162      weights = new_weights
    163
    164    time.sleep(1)
    165
    166
    167if __name__ == "__main__":
    168  app.run(main)

## Part 3: Implementation Details[\#](#part-3-implementation-details "Link to this heading")

We now take a deep dive into the architecture of multi-controller JAX and the semantics and implementation of `live_devices`. If you’re only interested in writing fault-tolerant multi-controller JAX programs, the first two parts of this article suffice.

### The Coordination Service[\#](#the-coordination-service "Link to this heading")

When you launch a multi-controller JAX program, the first process (i.e. process 0) runs a standalone RPC server called the **coordination service**. Moreover, all processes (including process 0) create an RPC client to the coordination service. Concretely, the `coordinator_address` argument of [`jax.distributed.initialize()`](_autosummary/jax.distributed.initialize.html#jax.distributed.initialize "jax.distributed.initialize") is the address of the coordination service. This argument lets process 0 know on what address to run the server, and it lets all processes know which address to connect to.

The coordination service implements the multi-controller JAX **control plane**. For example, it can perform a distributed barrier across all processes, and it implements a key-value store that processes can use to exchange small amounts of metadata. Note, however, that the **data plane** (e.g., all collective operations on program data) is implemented directly between the processes and does not involve the coordination service.

One of the most important functionalities of the coordination service is health checking. Every process periodically sends a heartbeat to the coordination service. If a process fails, it stops sending heartbeats. If the coordination service hasn’t received a heartbeat from a process for a while, it assumes the process has failed.

This is shown in the interactive visualization below. The coordination service is shown at the top and three multi-controller JAX processes are shown at the bottom. Note how the processes periodically send heartbeats to the controller, and the controller keeps track of the health of each process based on when it last received a heartbeat. Try failing process 2 by clicking the “Fail” button. Observe how the process stops sending heartbeats and the coordination service eventually considers the process dead.

By default, when the coordination service detects that a process has failed, it sends a message to all other processes requesting that they self-terminate. In other words, all processes in a multi-controller JAX program [share fate](https://en.wikipedia.org/wiki/Fate-sharing). Again fail process 2 in the visualization below by clicking the “Fail” button and observe how the coordination service notifies the other processes to fail.

This fate sharing means that multi-controller JAX programs are not at all fault-tolerant. They are fault-*intolerant*. To enable fault-tolerance, we need to do two things:

- First, we need to remove fate sharing and allow processes to continue executing even when a peer process has died. This can be enabled using the `jax_enable_recoverability` option, as described in [Part 1: Fault Tolerance Basics](#part1). We’ll assume that this option is set.

- Second, we need to provide an API that processes can use to learn which processes are alive and which have failed. This is the `live_devices` API introduced in [Part 1: Fault Tolerance Basics](#part1).

There is a surprising amount of technical depth and subtlety in implementing the `live_devices` API. We’ll walk through the design and implementation of the API step-by-step. We’ll begin by introducing a simpler `live_processes` API and slowly improve it until we arrive at the `live_devices` API.

### Live Processes[\#](#live-processes "Link to this heading")

Let’s try to design a new hypothetical JAX API: `jax.live_processes`. As the name suggests, we want `jax.live_processes()` to return the set of all currently alive processes. Here is a naive but (as we’ll see momentarily) incorrect implementation. When a process calls `jax.live_processes()`, it sends an RPC request to the coordination service. Remember that the coordination service already uses heartbeats to keep track of which processes are dead and which are alive, so when it receives a `jax.live_processes` request, it responds with the set of processes it thinks are alive.

This is illustrated below. Below each process is a “Call live_processes” button. You can click this button to make the process call `jax.live_processes`. Note how the coordination service replies to a `live_processess` request with the set of alive processes. Fail process 2 by clicking the “Fail” button and see how it affects later calls to `jax.live_processes`.

This naive implementation is simple but incorrect. It is crucial that all processes in a multi-controller JAX job execute the same instructions in the same order. If the processes start to diverge, by executing different code paths in the JAX program, the job will behave erratically. Most likely, it will crash or hang or produce garbage values, and most certainly it will be very hard to reason about.

Our naive implementation of `jax.live_processes` can very easily lead to divergence. For example, consider a multi-controller JAX job with three processes. If process 0 and 1 both call `jax.live_processes` around the same time that process 2 fails, the coordination service might report to process 0 that all processes are alive but report to process 1 that only processes 0 and 1 are alive. Try to produce this scenario in the visualization below:

If processes disagree on which processes are alive, they will almost certainly diverge. Thankfully, we can avoid this divergence by augmenting `jax.live_processes` with barrier semantics.

### Barrier Semantics[\#](#id2 "Link to this heading")

Let’s change the implementation of `jax.live_processes` so that when the coordination service receives a `jax.live_processes()` request, it does not reply right away. Instead, the coordination service only replies once *every* live process has called `jax.live_processes()`. Once every alive process has entered the `jax.live_processess()` barrier, the coordination service returns the set of live processes. Crucially, the coordination service returns the *same* set of live processes to all processes, which prevents the processes from diverging.

This is illustrated below. Note that coordination server now keeps track of which devices are in the `live_processes` barrier. Try calling `live_processes` from every process. Notice how the coordination service doesn’t respond until every process has entered the barrier. Then fail process 2 and call `live_processes` from process 0 and process 1.

### Formal Semantics[\#](#formal-semantics "Link to this heading")

Distributed systems are notoriously complex. Machines can fail at arbitrary times, and network messages can be dropped, delayed, and reordered. In this section, we introduce a formal semantics of the `jax.live_processes` API to help tame this complexity. Thinking rigorously about the semantics of `jax.live_processes` will help us understand the behavior of the API even in pathological executions.

We’ll base the formal semantics of `jax.live_processes` on [linearizability](https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf): a popular formalism used to define the semantics of many distributed APIs. Concretely, we model our distributed system as a number of processes. Each process serially performs a number of events. There are four types of events:

1.  A process can **start** (👶). We’ll assume that when a process starts, it connects to the coordination service, so the coordination service is aware that is has started.

2.  A process can **fail** (💀). Unlike starting, the coordination service may not immediately be aware that a process has failed.

3.  A process can **send** a `jax.live_processes` request to the coordination service.

4.  A process can **receive** a reply to a `jax.live_processes` request from the coordination service.

Below is a diagram of an execution of three processes: 0, 1, and 2. Time progresses from left to right. First, all three processes start. This is shown with the baby emojis. Then all three processes send `jax.live_processes` requests to the coordination service. This is shown as the start of the thick colored regions. Later, all three processes receive a reply from the coordination service with `0,1,2` as the set of live devices.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIyNTAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSIzNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSIxMDAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjEwMCIgeDI9IjI1MCIgeTI9IjEwMCIgY2xhc3M9InJwYyBwMi1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSI4NSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KICA8L3N2Zz4=)

In this simple execution, it is clear that `jax.live_processes` is behaving correctly. We can formalize this intuition with the following formal semantics.

Attention

An execution is valid if whenever `jax.live_processes` returns a set `P` of live processes, there exists an instantaneous moment in time at which every process in `P` was in the `live_processes` barrier and every other process was dead. An implementation of `live_processes` is correct if it only allows for valid executions.

Later, we will amend these formal semantics to cover some subtle corner cases, but assume this simplified semantics for now.

In the example above, `live_processes` returns `0,1,2`. In the visualization below, we show that there does exist an instantaneous moment of time in which processes 0, 1, and 2 are all in the barrier and all other processes (there are none) are dead. The moment in time is drawn as a vertical red bar.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIyNTAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSIzNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSIxMDAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjEwMCIgeDI9IjI1MCIgeTI9IjEwMCIgY2xhc3M9InJwYyBwMi1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSI4NSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFNuYXBzaG90IC0tPgogICAgPGxpbmUgeDE9IjE1MCIgeTE9Ii0yMCIgeDI9IjE1MCIgeTI9IjEyMCIgY2xhc3M9InNuYXBzaG90Ij48L2xpbmU+CiAgPC9zdmc+)

There is nothing special about the specific moment in time we chose in the visualization above. All that’s important is that *there exists some* moment in time where all processes in P are in the barrier and all other processes are dead. There are many moments in time that satisfy this property, as shown below.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIyNTAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSIzNSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMiAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSIxMDAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjEwMCIgeDI9IjI1MCIgeTI9IjEwMCIgY2xhc3M9InJwYyBwMi1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSI4NSIgY2xhc3M9InJlcGx5Ij4wLDEsMjwvdGV4dD4KCiAgICA8IS0tIFNuYXBzaG90IC0tPgogICAgPGxpbmUgeDE9IjE1MCIgeTE9Ii0yMCIgeDI9IjE1MCIgeTI9IjEyMCIgY2xhc3M9InNuYXBzaG90Ij4KICAgICAgPGFuaW1hdGUgYXR0cmlidXRlbmFtZT0ieDEiIHZhbHVlcz0iNTA7MjUwOzUwIiBkdXI9IjRzIiByZXBlYXRjb3VudD0iaW5kZWZpbml0ZSI+PC9hbmltYXRlPgogICAgICA8YW5pbWF0ZSBhdHRyaWJ1dGVuYW1lPSJ4MiIgdmFsdWVzPSI1MDsyNTA7NTAiIGR1cj0iNHMiIHJlcGVhdGNvdW50PSJpbmRlZmluaXRlIj48L2FuaW1hdGU+CiAgICA8L2xpbmU+CiAgPC9zdmc+)

In the next example, processes 0 and 1 start, call `live_devices`, and receive `0,1` as a reply. Process 2 is dead throughout the execution.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDIgLS0+CiAgICA8dGV4dCB4PSIyNSIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjUwIiB4Mj0iMjUwIiB5Mj0iNTAiIGNsYXNzPSJycGMgcDEtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI1MCIgeT0iMzUiIGNsYXNzPSJyZXBseSI+MCwxPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAyIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjEwMCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgogIDwvc3ZnPg==)

This is a valid execution under our formal semantics because there exists a moment a time in which processes 0 and 1 are in the barrier and process 2 is dead.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDIgLS0+CiAgICA8dGV4dCB4PSIyNSIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjUwIiB4Mj0iMjUwIiB5Mj0iNTAiIGNsYXNzPSJycGMgcDEtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI1MCIgeT0iMzUiIGNsYXNzPSJyZXBseSI+MCwxPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAyIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjEwMCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgoKICAgIDwhLS0gU25hcHNob3QgLS0+CiAgICA8bGluZSB4MT0iMTUwIiB5MT0iLTIwIiB4Mj0iMTUwIiB5Mj0iMTIwIiBjbGFzcz0ic25hcHNob3QiPjwvbGluZT4KICA8L3N2Zz4=)

In the following execution, process 0 calls `jax.live_processes` and receives a reply of `0`. Process 1 calls `jax.live_processes`, but dies before receiving a reply.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjE1MCIgeTI9IjUwIiBjbGFzcz0icnBjIHAxLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIxNTAiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkoA8L3RleHQ+CiAgPC9zdmc+)

Is this a valid execution? Yes. There exists a moment in time at which process 0 is in the barrier and process 1 is dead, as shown below. Even though process 1 called `jax.live_processes`, it is not guaranteed that process 1 will be included in the coordination service’s response.

For example, process 1’s `jax.live_processes` request may have been dropped by the network and never received by the coordination service. So from the coordination service’s perspective, process 1 is thoroughly dead and never even entered the `live_processes` barrier.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAxIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iNTAiIHkxPSI1MCIgeDI9IjE1MCIgeTI9IjUwIiBjbGFzcz0icnBjIHAxLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIxNTAiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkoA8L3RleHQ+CgogICAgPCEtLSBTbmFwc2hvdCAtLT4KICAgIDxsaW5lIHgxPSIyMDAiIHkxPSItMjAiIHgyPSIyMDAiIHkyPSI3MCIgY2xhc3M9InNuYXBzaG90Ij48L2xpbmU+CiAgPC9zdmc+)

What about the same exact execution, except that process 0 now receives the reply `0,1` from the coordination service?

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDEgLS0+CiAgICA8dGV4dCB4PSIyNSIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjUwIiB4Mj0iMTUwIiB5Mj0iNTAiIGNsYXNzPSJycGMgcDEtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjE1MCIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+SgDwvdGV4dD4KICA8L3N2Zz4=)

Again, this is a valid execution, as witnessed below. Intuitively, the coordination service could have received `jax.live_processes` requests from both processes 0 and 1 and sent the reply `0,1` to both. While this reply was in the network, process 1 failed. Thus, even though process 1 is dead when process 0 receives a reply, the execution is still valid.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjAiIHgyPSIyNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDEgLS0+CiAgICA8dGV4dCB4PSIyNSIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI1MCIgeTE9IjUwIiB4Mj0iMTUwIiB5Mj0iNTAiIGNsYXNzPSJycGMgcDEtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjE1MCIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+SgDwvdGV4dD4KCiAgICA8IS0tIFNuYXBzaG90IC0tPgogICAgPGxpbmUgeDE9IjEwMCIgeTE9Ii0yMCIgeDI9IjEwMCIgeTI9IjcwIiBjbGFzcz0ic25hcHNob3QiPjwvbGluZT4KICA8L3N2Zz4=)

This point bears repeating. If `jax.live_processes` returns a set `P` of processes, it does not mean that all processes in `P` are *currently* alive and all other processes are *currently* dead. It only means that *there existed a point in time* when this was true.

In the following execution, process 1 calls `jax.live_processes` and fails. Later, process 0 starts, calls `jax.live_processes`, and receives `0,1` as a reply.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMTc1IiB5PSIwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iMjAwIiB5MT0iMCIgeDI9IjI1MCIgeTI9IjAiIGNsYXNzPSJycGMgcDAtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI1MCIgeT0iLTE1IiBjbGFzcz0icmVwbHkiPjAsMTwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIxMDAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMTAwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgogIDwvc3ZnPg==)

Using the formal semantics described thus far, this is *not* a valid execution. There is never a point in time where process 0 and 1 are both alive. However, this *should* be a valid execution.

The reason has to do with the unavoidable fact that in a distributed system, it is impossible to detect failures with 100% accuracy. If the coordination service hasn’t received heartbeats from a process in a while, it considers the process dead. But, the coordination service cannot determine with 100% certainty when the process died or if the process is actually dead at all. Maybe the process died a long time ago, or maybe it died very recently, or maybe it is alive but on the other side of a network partition.

Let’s return to the execution above for a concrete example. Imagine the coordination service successfully received process 1’s `live_processes` request. Then, process 1 failed but the coordination service didn’t detect the failure immediately. In the meantime, the coordination service received process 0’s `live_processes` request. At this point, the coordination service thought both processes were alive and saw that both processes were in the barrier, so it naturally returned `0,1` to both processes (though only process 0 received the reply because process 1 was dead).

The coordination service thought process 1 was alive when it was dead. And sometimes the coordination service might think a process is dead when it is alive. Though not ideal, we need to accommodate executions like this because they are unavoidable.

We amend our formal semantics and allow ourselves to move a failure either earlier or later in time, though we cannot move a failure past a different event from the same process. Intuitively, we can move a failure from when it actually happened to the point in time when the coordination service thought it happened. Continuing the example above, we can delay the failure of process 1 to create a moment in time in which both processes 0 and 1 are in the barrier, witnessing the fact that the execution is valid.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMTc1IiB5PSIwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iMjAwIiB5MT0iMCIgeDI9IjI1MCIgeTI9IjAiIGNsYXNzPSJycGMgcDAtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI1MCIgeT0iLTE1IiBjbGFzcz0icmVwbHkiPjAsMTwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIxMDAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+CiAgICAgIDxhbmltYXRlIGF0dHJpYnV0ZW5hbWU9IngyIiB2YWx1ZXM9IjEwMDsyNzU7MTAwIiBkdXI9IjRzIiByZXBlYXRjb3VudD0iaW5kZWZpbml0ZSI+PC9hbmltYXRlPgogICAgPC9saW5lPgogICAgPHRleHQgeD0iMTAwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij4KICAgICAgPGFuaW1hdGUgYXR0cmlidXRlbmFtZT0ieCIgdmFsdWVzPSIxMDA7Mjc1OzEwMCIgZHVyPSI0cyIgcmVwZWF0Y291bnQ9ImluZGVmaW5pdGUiPjwvYW5pbWF0ZT4KICAgICAg8J+SgAogICAgPC90ZXh0PgoKICAgIDwhLS0gU25hcHNob3QgLS0+CiAgICA8bGluZSB4MT0iMjI1IiB5MT0iLTIwIiB4Mj0iMjI1IiB5Mj0iNzAiIGNsYXNzPSJzbmFwc2hvdCI+CiAgICAgIDxhbmltYXRlIGF0dHJpYnV0ZW5hbWU9Im9wYWNpdHkiIHZhbHVlcz0iMDswOzA7MTswOzA7MCIgZHVyPSI0cyIgcmVwZWF0Y291bnQ9ImluZGVmaW5pdGUiPjwvYW5pbWF0ZT4KICAgIDwvbGluZT4KICA8L3N2Zz4=)

Consider a similar execution below.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSIxMDAiIHkxPSIwIiB4Mj0iMjAwIiB5Mj0iMCIgY2xhc3M9InJwYyBwMC1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjAwIiB5PSItMTUiIGNsYXNzPSJyZXBseSI+MDwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIyNTAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgogIDwvc3ZnPg==)

As is, there is no moment in time in which process 0 is alive and process 1 is dead. However, if we move the failure of process 1 leftwards, there is. How might such an execution arise? Imagine process 1 is partitioned from the coordination service. The coordination service doesn’t receive any messages from process 1, including its heartbeats. This leads the coordination service to conclude that process 1 is dead, even though it isn’t. Then, the coordination service receives process 0’s `live_processes` request and responds with `0`.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSIxMDAiIHkxPSIwIiB4Mj0iMjAwIiB5Mj0iMCIgY2xhc3M9InJwYyBwMC1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjAwIiB5PSItMTUiIGNsYXNzPSJyZXBseSI+MDwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIyNTAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+CiAgICAgIDxhbmltYXRlIGF0dHJpYnV0ZW5hbWU9IngyIiB2YWx1ZXM9IjI1MDsxMDA7MjUwIiBkdXI9IjRzIiByZXBlYXRjb3VudD0iaW5kZWZpbml0ZSI+PC9hbmltYXRlPgogICAgPC9saW5lPgogICAgPHRleHQgeD0iMjUwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij4KICAgICAgPGFuaW1hdGUgYXR0cmlidXRlbmFtZT0ieCIgdmFsdWVzPSIyNTA7MTAwOzI1MCIgZHVyPSI0cyIgcmVwZWF0Y291bnQ9ImluZGVmaW5pdGUiPjwvYW5pbWF0ZT4KICAgICAg8J+SgAogICAgPC90ZXh0PgoKICAgIDwhLS0gU25hcHNob3QgLS0+CiAgICA8bGluZSB4MT0iMTUwIiB5MT0iLTIwIiB4Mj0iMTUwIiB5Mj0iNzAiIGNsYXNzPSJzbmFwc2hvdCI+CiAgICAgIDxhbmltYXRlIGF0dHJpYnV0ZW5hbWU9Im9wYWNpdHkiIHZhbHVlcz0iMDswOzA7MTswOzA7MCIgZHVyPSI0cyIgcmVwZWF0Y291bnQ9ImluZGVmaW5pdGUiPjwvYW5pbWF0ZT4KICAgIDwvbGluZT4KICA8L3N2Zz4=)

We cannot move a process failure past the process’ other events, however. For example, the following execution is *invalid* because no matter where we move the failure of process 1, there is never a moment in time where both processes are in the barrier.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxMTAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIGF4ZXMgLS0+CiAgICA8bGluZSB4MT0iMTAiIHkxPSIwIiB4Mj0iMzAwIiB5Mj0iMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iNTAiIHgyPSIzMDAiIHkyPSI1MCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMTc1IiB5PSIwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iMjAwIiB5MT0iMCIgeDI9IjI1MCIgeTI9IjAiIGNsYXNzPSJycGMgcDAtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI1MCIgeT0iLTE1IiBjbGFzcz0icmVwbHkiPjAsMTwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPHRleHQgeD0iMTc1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjUwIiB5MT0iNTAiIHgyPSIxMDAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+CiAgICAgIDxhbmltYXRlIGF0dHJpYnV0ZW5hbWU9IngyIiB2YWx1ZXM9IjEwMDsxNTA7MTAwIiBkdXI9IjJzIiByZXBlYXRjb3VudD0iaW5kZWZpbml0ZSI+PC9hbmltYXRlPgogICAgPC9saW5lPgogICAgPHRleHQgeD0iMTAwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij4KICAgICAgPGFuaW1hdGUgYXR0cmlidXRlbmFtZT0ieCIgdmFsdWVzPSIxMDA7MTUwOzEwMCIgZHVyPSIycyIgcmVwZWF0Y291bnQ9ImluZGVmaW5pdGUiPjwvYW5pbWF0ZT4KICAgICAg8J+SgAogICAgPC90ZXh0PgogIDwvc3ZnPg==)

With these formal semantics, we can make sense of even complex executions. For example, consider the following execution.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI0MCIgeTE9IjAiIHgyPSIxNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIxNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wPC90ZXh0PgogICAgPGxpbmUgeDE9IjE3NSIgeTE9IjAiIHgyPSIyNzUiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNzUiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDI8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDEgLS0+CiAgICA8dGV4dCB4PSI1MCIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI2NSIgeTE9IjUwIiB4Mj0iMTI1IiB5Mj0iNTAiIGNsYXNzPSJycGMgcDEtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjEyNSIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+SgDwvdGV4dD4KICAgIDx0ZXh0IHg9IjE1MCIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSIxNjUiIHkxPSI1MCIgeDI9IjI5MCIgeTI9IjUwIiBjbGFzcz0icnBjIHAxLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyOTAiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkoA8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDIgLS0+CiAgICA8dGV4dCB4PSIxMDAiIHk9IjEwMCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjExNSIgeTE9IjEwMCIgeDI9IjE1MCIgeTI9IjEwMCIgY2xhc3M9InJwYyBwMi1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMTUwIiB5PSIxMDAiIGNsYXNzPSJldmVudCI+8J+SgDwvdGV4dD4KICA8L3N2Zz4=)

After moving some process failures, we see the execution is valid.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI0MCIgeTE9IjAiIHgyPSIxNTAiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIxNTAiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wPC90ZXh0PgogICAgPGxpbmUgeDE9IjE3NSIgeTE9IjAiIHgyPSIyNzUiIHkyPSIwIiBjbGFzcz0icnBjIHAwLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNzUiIHk9Ii0xNSIgY2xhc3M9InJlcGx5Ij4wLDI8L3RleHQ+CgogICAgPCEtLSBQcm9jZXNzIDEgLS0+CiAgICA8dGV4dCB4PSI1MCIgeT0iNTAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSI2NSIgeTE9IjUwIiB4Mj0iNzUiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iNzUiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkoA8L3RleHQ+CiAgICA8dGV4dCB4PSIxNTAiIHk9IjUwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iMTY1IiB5MT0iNTAiIHgyPSIyMDAiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMjAwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAyIC0tPgogICAgPHRleHQgeD0iMTAwIiB5PSIxMDAiIGNsYXNzPSJldmVudCI+8J+RtjwvdGV4dD4KICAgIDxsaW5lIHgxPSIxMTUiIHkxPSIxMDAiIHgyPSIyNzUiIHkyPSIxMDAiIGNsYXNzPSJycGMgcDItY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI3NSIgeT0iMTAwIiBjbGFzcz0iZXZlbnQiPvCfkoA8L3RleHQ+CgogICAgPCEtLSBTbmFwc2hvdCAtLT4KICAgIDxsaW5lIHgxPSI4NyIgeTE9Ii0yMCIgeDI9Ijg3IiB5Mj0iMTIwIiBjbGFzcz0ic25hcHNob3QiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIyMjUiIHkxPSItMjAiIHgyPSIyMjUiIHkyPSIxMjAiIGNsYXNzPSJzbmFwc2hvdCI+PC9saW5lPgogIDwvc3ZnPg==)

The following execution, on the other hand, is invalid.

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSItMTAgLTMwIDMyNSAxNjAiPgogICAgPCEtLSBQcm9jZXNzIG5hbWVzIC0tPgogICAgPHRleHQgeD0iMCIgeT0iMCIgY2xhc3M9InByb2MiPjA8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSI1MCIgY2xhc3M9InByb2MiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIwIiB5PSIxMDAiIGNsYXNzPSJwcm9jIj4yPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyBheGVzIC0tPgogICAgPGxpbmUgeDE9IjEwIiB5MT0iMCIgeDI9IjMwMCIgeTI9IjAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjUwIiB4Mj0iMzAwIiB5Mj0iNTAiIGNsYXNzPSJwcm9jLWF4aXMiPjwvbGluZT4KICAgIDxsaW5lIHgxPSIxMCIgeTE9IjEwMCIgeDI9IjMwMCIgeTI9IjEwMCIgY2xhc3M9InByb2MtYXhpcyI+PC9saW5lPgoKICAgIDwhLS0gUHJvY2VzcyAwIC0tPgogICAgPHRleHQgeD0iMTY1IiB5PSIwIiBjbGFzcz0iZXZlbnQiPvCfkbY8L3RleHQ+CiAgICA8bGluZSB4MT0iMTgwIiB5MT0iMCIgeDI9IjI3NSIgeTI9IjAiIGNsYXNzPSJycGMgcDAtY29sb3IiPjwvbGluZT4KICAgIDx0ZXh0IHg9IjI3NSIgeT0iLTE1IiBjbGFzcz0icmVwbHkiPjAsMjwvdGV4dD4KCiAgICA8IS0tIFByb2Nlc3MgMSAtLT4KICAgIDx0ZXh0IHg9IjI1IiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjQwIiB5MT0iNTAiIHgyPSIxMjUiIHkyPSI1MCIgY2xhc3M9InJwYyBwMS1jb2xvciI+PC9saW5lPgogICAgPHRleHQgeD0iMTI1IiB5PSIzNSIgY2xhc3M9InJlcGx5Ij4xPC90ZXh0PgogICAgPHRleHQgeD0iMTQwIiB5PSI1MCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgoKICAgIDwhLS0gUHJvY2VzcyAyIC0tPgogICAgPHRleHQgeD0iMjUiIHk9IjEwMCIgY2xhc3M9ImV2ZW50Ij7wn5G2PC90ZXh0PgogICAgPGxpbmUgeDE9IjQwIiB5MT0iMTAwIiB4Mj0iMjc1IiB5Mj0iMTAwIiBjbGFzcz0icnBjIHAyLWNvbG9yIj48L2xpbmU+CiAgICA8dGV4dCB4PSIyNzUiIHk9IjEwMCIgY2xhc3M9ImV2ZW50Ij7wn5KAPC90ZXh0PgogIDwvc3ZnPg==)

### Atomicity[\#](#id3 "Link to this heading")

Equipped with `jax.live_processes`, let’s try to write some fault-tolerant multi-controller JAX code.

    step = 0
    while True:
        # Get the devices on all live processes.
        procs = jax.live_processes()
        devices = [d for d in jax.devices() if d.process_index in procs]

        # Shard array x over these devices.
        mesh = jax.make_mesh((len(devices),), ("i",), devices=devices)
        spec = jax.sharding.PartitionSpec("i")
        sharding = jax.sharding.NamedSharding(mesh, spec)
        x = jax.make_array_from_process_local_data(sharding, np.ones(1))

        # Try to perform a jnp.sum.
        try:
            print(jnp.sum(x))
        except:
            # jnp.sum failed.
            pass
        else:
            # jnp.sum succeeded.
            step += 1

The code repeatedly

- calls `jax.live_processes` to learn which processes are alive,

- computes the set of devices on the healthy processes,

- shards an array across these healthy devices,

- performs a `jnp.sum` (i.e. AllReduce) on the array, and

- increments `step` if the `jnp.sum` succeeds.

This code *looks* correct, but it has a very subtle bug. Assume the `jnp.sum` is being performed across a set of processes `P`. If one (or more) of the processes in `P` fails during the execution of the `jnp.sum`, then `jnp.sum` can behave differently on different processes. Some processes in `P` might see `jnp.sum` return the correct result. Other processes might see `jnp.sum` raise an exception. Others might see `jnp.sum` return an incorrect result.

Warning

If a process fails during a collective operation, the operation may behave differently on different processes.

This means that the processes executing the code example above might diverge. Some might increment `step`, and some might not. In the trivial code example above, this divergence is benign, but in a real program, the divergence would likely lead to a crash, a deadlock, or garbage outputs. For example, if a multi-controller JAX program is training a model with data parallelism and starts to diverge, some processes might roll back their model weights to a previous checkpoint while others continue training, leading to a “franken-model” where nobody agrees on what the model weights are supposed to be.

To write fault-tolerant code that does not diverge, we want **atomicity**. When executing a block of code (like the `jnp.sum` above), we either want *every* process to run the code successfully, or *every* process to learn that the code failed to execute successfully. We don’t want some processes succeeding and others failing.

Thankfully, we can achieve atomicity with a very simple trick: call `live_processes` twice, once before a code block and once after. If all the processes that were alive before the block are also alive after the block, then the code block executed successfully on all live processes. On the other hand, if any process died, then all remaining processes can agree the code block failed to execute properly. Here’s a sketch of what that might look like:

    # Get the set of live processes before the code block.
    procs_before = jax.live_processes()

    # Execute the code block.
    ...

    # Get the set of live processes after the code block
    procs_after = jax.live_processes()
    if procs_before == procs_after:
        # The code block executed successfully on all processes in
        # procs_before.
        pass
    else:
        # The code block did not execute successfully. All processes will
        # agree it failed.
        pass

The code above should give you a rough idea of how to use two calls to `live_processes` to achieve atomicity, but there are still a handful of small issues we need to address before it is fully correct. For example,

- What if the code block throws an exception? We need to catch the exception and still call `live_processess` the second time and then re-raise the exception.

- What if a process fails after the first call to `live_processes` and recovers before the second call? Wouldn’t the code block fail but the processes before and after be the same? Every time a process starts, it generates a random **incarnation id**. In addition to checking that the set of processes hasn’t changed, we also check that their incarnation ids haven’t changed.

- What if a process recovers and its first call to `live_processes` matches up with a different process’ second call to `live_processes`? Couldn’t this lead to a deadlock? Yes. We can avoid the problem by only calling `live_processes` at a single program point. We can be clever and use a single call to `live_processes` for two purposes. It can be used to check that the set of processes hasn’t changed since the previous call to `live_processes`, and it can be used to generate the set of live processes that should be used the next time the atomic code block is executed.

All these details are handled and abstracted away by the `live_devices` API introduced in [Part 1: Fault Tolerance Basics](#part1). `live_devices` is a context manager that guarantees the atomic execution of a block of code. In the code snippet below, `devices` is a list of the devices on all live processes. The code block `A` will execute atomically across these processes. That is, either every process will see the code raise an exception (branch `B`) or every process will see the code succeed (branch `C`).

    try:
      with live_devices() as devices:
        pass # A
    except Exception as e:
      pass # B
    else:
      pass # C

### Cancelling Collectives[\#](#id4 "Link to this heading")

As mentioned in [Cancelling Collectives](#canceling-collectives), if a process participating in a collective fails, then the other participating processes get stuck forever. We need to explicitly cancel these collectives to allow the alive participants to make progress. While the `live_devices` API is supported on all JAX backends (i.e. CPU, GPU, TPU), cancelling collectives is only supported by the GPU backend. Here, we briefly explain some of the implementation details behind collective cancelling.

The GPU backend implements collectives using [NCCL](https://developer.nvidia.com/nccl), NVIDIA’s collective communication library. When a set of processes wants to perform a collective, they form a **NCCL communicator**. Processes can then repeatedly perform collectives using this communicator. Creating a communicator is expensive—it requires network communication—so the JAX backend caches communicators keyed by the set of participating processes and their incarnation ids.

Internally, a JAX client polls the coordination service for the current status of every process. If a client ever detects that a process is dead or has restarted with a new incarnation id, then the client aborts all communicators with the failed incarnation id in its cache key.

[](multi_process.html "previous page")

previous

Introduction to multi-controller JAX (aka multi-process/multi-host JAX)

[](distributed_data_loading.html "next page")

next

Distributed data loading

Contents

- [Part 1: Fault Tolerance Basics](#part-1-fault-tolerance-basics)
  - [Fault Intolerant By Default](#fault-intolerant-by-default)
  - [Surviving Faults](#surviving-faults)
  - [Getting Stuck in Collectives](#getting-stuck-in-collectives)
  - [Cancelling Collectives](#cancelling-collectives)
  - [Knowing Who’s Alive](#knowing-who-s-alive)
  - [Barrier Semantics](#barrier-semantics)
  - [Atomicity](#atomicity)
- [Part 2: Examples](#part-2-examples)
  - [Example 1: Fault Tolerant Data Parallel Training](#example-1-fault-tolerant-data-parallel-training)
  - [Example 2: Fault Tolerant Data Parallel Training With Recovery](#example-2-fault-tolerant-data-parallel-training-with-recovery)
- [Part 3: Implementation Details](#part-3-implementation-details)
  - [The Coordination Service](#the-coordination-service)
  - [Live Processes](#live-processes)
  - [Barrier Semantics](#id2)
  - [Formal Semantics](#formal-semantics)
  - [Atomicity](#id3)
  - [Cancelling Collectives](#id4)

By The JAX authors

© Copyright 2024, The JAX Authors.\
