- [](../index.html)
- [API Reference](../jax.html)
- [`jax.distributed` module](../jax.distributed.html)
- jax.distributed.initialize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.distributed.initialize.rst "Download source file")
-  .pdf

# jax.distributed.initialize

## Contents

- [`initialize()`](#jax.distributed.initialize)

# jax.distributed.initialize[\#](#jax-distributed-initialize "Link to this heading")

jax.distributed.initialize(*coordinator_address=None*, *num_processes=None*, *process_id=None*, *local_device_ids=None*, *cluster_detection_method=None*, *initialization_timeout=300*, *heartbeat_timeout_seconds=100*, *shutdown_timeout_seconds=300*, *coordinator_bind_address=None*, *slice_index=None*, *partition_index=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/distributed.py#L222-L338)[\#](#jax.distributed.initialize "Link to this definition")  
Initializes the JAX distributed system.

Calling [`initialize()`](#jax.distributed.initialize "jax.distributed.initialize") prepares JAX for execution on multi-host GPU and Cloud TPU. [`initialize()`](#jax.distributed.initialize "jax.distributed.initialize") must be called before performing any JAX computations.

The JAX distributed system serves a number of roles:

> - It allows JAX processes to discover each other and share topology information,
>
> - It performs health checking, ensuring that all processes shut down if any process dies, and
>
> - It is used for distributed checkpointing.

If you are using TPU, Slurm, or Open MPI, all arguments are optional: if omitted, they will be chosen automatically.

The `cluster_detection_method` may be used to choose a specific method for detecting those distributed arguments. You may pass any of the automatic `spec_detect_methods` to this argument though it is not necessary in the TPU, Slurm, or Open MPI cases. For other MPI installations, if you have a functional `mpi4py` installed, you may pass `cluster_detection_method="mpi4py"` to bootstrap the required arguments.

Otherwise, you must provide the `coordinator_address`, `num_processes`, `process_id`, and `local_device_ids` arguments to [`initialize()`](#jax.distributed.initialize "jax.distributed.initialize"). When all four arguments are provided, cluster environment auto detection will be skipped.

Please note: on some systems, particularly HPC clusters that only access external networks through proxy variables such as HTTP_PROXY, HTTPS_PROXY, etc., the call to [`initialize()`](#jax.distributed.initialize "jax.distributed.initialize") may timeout. You may need to unset these variables prior to application launch.

Parameters:  
- **coordinator_address** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – the IP address of process 0 and a port on which that process should launch a coordinator service. The choice of port does not matter, so long as the port is available on the coordinator and all processes agree on the port. May be `None` only on supported environments, in which case it will be chosen automatically. Note that special addresses like `localhost` or `127.0.0.1` usually mean that the program will bind to a local interface and are not suitable when running in a multi-host environment.

- **num_processes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Number of processes. May be `None` only on supported environments, in which case it will be chosen automatically.

- **process_id** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The ID number of the current process. The `process_id` values across the cluster must be a dense range `0`, `1`, …, `num_processes`` ``-`` ``1`. May be `None` only on supported environments; if `None` it will be chosen automatically.

- **local_device_ids** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – Restricts the visible devices of the current process to `local_device_ids`. If `None`, defaults to all local devices being visible to the process except when processes are launched via Slurm and Open MPI on GPUs. In that case, it will default to a single device per process.

- **cluster_detection_method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – An optional string to attempt to autodetect the configuration of the distributed run. Note that “mpi4py” method requires you to have a working `mpi4py` install in your environment, and launch the applicatoin with an MPI-compatible job launcher such as `mpiexec` or `mpirun`. Legacy auto-detect options “ompi” (OMPI) and “slurm” (Slurm) remain enabled. “deactivate” bypasses automatic cluster detection.

- **initialization_timeout** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Time period (in seconds) for which connection will be retried. If the initialization takes more than the timeout specified, the initialization will error. Defaults to 300 secs i.e. 5 mins.

- **heartbeat_timeout_seconds** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The time (in seconds) after which a process is considered dead if it hasn’t successfully sent any heartbeats. Defaults to 100 seconds.

- **shutdown_timeout_seconds** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The time (in seconds) a terminating process will wait for all other processes to also terminate. Defaults to 300 seconds.

- **coordinator_bind_address** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – the address and port to which the coordinator service on process 0 should bind. If this is not specified, the default is to bind to all available addresses on the same port as `coordinator_address`. On systems that have multiple network interfaces per node it may be insufficient to only have the coordinator service listen on one address/interface.

- **slice_index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – DEPRECATED: Use `partition_index` instead.

- **partition_index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The partition index assigned to this process’ local devices. If any process sets `partition_index`, then all processes must do so. If `None` the partition indices will be chosen automatically.

Raises:  
[**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.14)") – If [`initialize()`](#jax.distributed.initialize "jax.distributed.initialize") is called more than once or if called after the backend is already initialized.

Examples:

Suppose there are two GPU processes, and process 0 is the designated coordinator with address `10.0.0.1:1234`. To initialize the GPU cluster, run the following commands before anything else.

On process 0:

    >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=0)  

On process 1:

    >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=1)  

[](../jax.distributed.html "previous page")

previous

`jax.distributed` module

[](jax.distributed.shutdown.html "next page")

next

jax.distributed.shutdown

Contents

- [`initialize()`](#jax.distributed.initialize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
