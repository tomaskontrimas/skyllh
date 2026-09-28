import multiprocessing as mp
import os
import queue
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from logging.handlers import (
    QueueHandler,
)
from typing import Literal, cast

import numpy as np

from skyllh.core.config import Config, HasConfig
from skyllh.core.logging import (
    get_logger,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
)
from skyllh.core.random import (
    RandomStateService,
)
from skyllh.core.timing import (
    TimeLord,
)

# The type of a number of CPUs setting. The value 'auto' means the number of
# CPUs available to the current process.
NCpuSetting = int | Literal['auto']

# The multiprocessing start method used by SkyLLH. The "fork" start method lets
# the worker processes share the (read-only) memory of the main process via
# copy-on-write, which avoids pickling and duplicating the analysis data.
_MP_START_METHOD = 'fork'

# Flag if the current process is a worker process spawned by ``parallelize``.
# Nested calls to ``parallelize`` within a worker process are executed
# serially to avoid over-subscribing the CPUs.
_IS_WORKER_PROCESS = False

# The time in seconds to wait for data from the worker processes before
# checking their health.
_POLL_INTERVAL = 0.1

# The time in seconds to wait for the result of a worker process, which exited
# successfully, before giving up.
_RESULT_TIMEOUT = 10


def get_available_ncpu() -> int:
    """Determines the number of CPUs the current process is allowed to use.

    In contrast to :func:`os.cpu_count`, which returns the number of CPUs of the
    entire machine, this function respects the CPU affinity of the process,
    which is set for example by batch systems like Slurm or HTCondor, or by
    tools like ``taskset``.

    Returns
    -------
    ncpu
        The number of CPUs available to the current process.
    """
    process_cpu_count = getattr(os, 'process_cpu_count', None)
    if process_cpu_count is not None:
        # Python >= 3.13
        ncpu = process_cpu_count()
    elif hasattr(os, 'sched_getaffinity'):
        ncpu = len(os.sched_getaffinity(0))
    else:
        ncpu = os.cpu_count()

    if ncpu is None or ncpu < 1:
        ncpu = 1

    return ncpu


def get_ncpu(
    cfg: Config,
    local_ncpu: NCpuSetting | None,
) -> int:
    """Determines the number of CPUs to use for functions that support
    multi-processing.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    local_ncpu
        The local setting of the number of CPUs to use. If set to ``'auto'``,
        the number of CPUs available to the current process is used, see
        :func:`get_available_ncpu`.

    Returns
    -------
    ncpu
        The number of CPUs to use by functions that allow multi-processing.
        If ``local_ncpu`` is set to None, the global NCPU setting is returned.
        If the global NCPU setting is None as well, the default value 1 is
        returned.
    """
    ncpu = local_ncpu
    if ncpu is None:
        ncpu = cfg['multiproc']['ncpu']
    if ncpu is None:
        ncpu = 1
    if ncpu == 'auto':
        ncpu = get_available_ncpu()

    if not isinstance(ncpu, int):
        raise TypeError("The ncpu setting must be of type int or 'auto'!")

    if ncpu < 1:
        raise ValueError('The ncpu setting must be >= 1!')

    return ncpu


def _limit_blas_threads(n_threads: int | None) -> AbstractContextManager:
    """Creates a context manager limiting the number of threads used by the
    BLAS and OpenMP libraries, e.g. used by numpy and scipy.

    Parameters
    ----------
    n_threads
        The maximum number of threads. If set to ``None``, no limit is applied.

    Returns
    -------
    ctx
        The context manager applying the limit. If the ``threadpoolctl`` package
        is not available, no limit is applied.
    """
    if n_threads is None:
        return nullcontext()

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()

    return threadpool_limits(limits=n_threads)


def _run_tasks(
    func: Callable,
    sub_args_list: list[tuple],
    rss: RandomStateService | None,
    tl: TimeLord | None,
    task_done_callback: Callable[[int], None] | None = None,
) -> list:
    """Evaluates ``func`` for all the arguments given by ``sub_args_list``.

    Parameters
    ----------
    func
        The function to call.
    sub_args_list
        The list of 2-element tuples holding the arguments and keyword arguments
        of ``func`` for each task.
    rss
        The RandomStateService instance that should be passed to ``func`` via
        the ``rss`` keyword argument. If None, no ``rss`` argument is passed.
    tl
        The TimeLord instance that should be passed to ``func`` via the ``tl``
        keyword argument. If None, no ``tl`` argument is passed.
    task_done_callback
        The optional callable that is called with the task index after each
        finished task.

    Returns
    -------
    result_list
        The list of the results of ``func`` for each task.
    """
    result_list = []
    for task_idx, (args, kwargs) in enumerate(sub_args_list):
        kwargs = dict(kwargs)
        if rss is not None:
            kwargs['rss'] = rss
        if tl is not None:
            kwargs['tl'] = tl
        result_list.append(func(*args, **kwargs))

        if task_done_callback is not None:
            task_done_callback(task_idx)

    return result_list


def _worker_main(
    func: Callable,
    sub_args_list: list[tuple],
    pid: int,
    rqueue,
    lqueue,
    squeue=None,
    rss: RandomStateService | None = None,
    tl: TimeLord | None = None,
    blas_threads: int | None = None,
):
    """The main function of a worker process. It evaluates ``func`` for the
    subset ``sub_args_list`` of all the arguments.

    The worker process inherits the ``QueueHandler`` of the skyllh logger,
    which sends the log records to the main process via the ``lqueue``.

    Parameters
    ----------
    func
        The function which should be called with different arguments.
    sub_args_list
        The list of the different arguments for function ``func``.
    pid
        The process ID that identifies the process in order to sort the
        results to the initial order of the function arguments.
    rqueue
        The Queue instance where to put the function results in. The result is
        a 4-element tuple ``(pid, result_list, tl, error)``, where ``error`` is
        the formatted traceback string of a raised exception, or ``None``.
    lqueue
        The Queue instance for the log records. When the worker is finished,
        ``None`` is put into the queue to mark the end of its log records.
    squeue
        The Queue instance where to put in status information about finished
        tasks. Can be None to skip sending status information.
    rss
        The RandomStateService instance to use for generating random numbers.
    tl
        The instance of TimeLord that should be used to time individual tasks.
    blas_threads
        The maximum number of BLAS / OpenMP threads for this worker process.
    """
    global _IS_WORKER_PROCESS
    _IS_WORKER_PROCESS = True

    def task_done_callback(task_idx):
        if squeue is not None:
            squeue.put((pid, task_idx))

    try:
        with _limit_blas_threads(blas_threads):
            result_list = _run_tasks(func, sub_args_list, rss, tl, task_done_callback)
        rqueue.put((pid, result_list, tl, None))
    except Exception:  # noqa: BLE001 - Forward any error to the main process.
        rqueue.put((pid, None, None, traceback.format_exc()))
    finally:
        # Mark the end of the log records of this worker process.
        lqueue.put(None)


def parallelize(
    func: Callable,
    args_list: list[tuple],
    ncpu: int,
    rss: RandomStateService | None = None,
    tl: TimeLord | None = None,
    ppbar: ProgressBar | None = None,
    blas_threads: int | None = 1,
) -> list:
    """Parallelizes the execution of the given function for different arguments.

    The main process is used as one of the ``ncpu`` worker processes. The worker
    processes are created using the "fork" start method, hence they share the
    memory of the main process via copy-on-write and the function and its
    arguments do not need to be pickled. Only the results are transferred back
    to the main process.

    Parameters
    ----------
    func
        The function which should be called with different arguments, which are
        given through the args_list argument. If the `rss` argument is not None,
        `func` requires an argument named `rss`.
    args_list
        The list of the different arguments for function ``func``. Each element
        of that list must be a 2-element tuple, where the first element is a
        tuple of the arguments of ``func``, and the second element is a
        dictionary with the keyword arguments of ``func``. If the `rss` argument
        is not None, `func` argument `rss` has to be omitted.
    ncpu
        The number of CPUs to use, i.e. the number of processes (including the
        main process) to use. If this function is called within a worker
        process of another ``parallelize`` call, the tasks are executed
        serially.
    rss
        The RandomStateService instance to use for generating random numbers.
    tl
        The instance of TimeLord that should be used to time individual tasks.
    ppbar
        The possible parent ProgressBar instance.
    blas_threads
        The maximum number of threads each process may use for BLAS and OpenMP
        operations, e.g. within numpy and scipy, while running the tasks in
        parallel. This avoids over-subscribing the CPUs, i.e. using
        ``ncpu`` times the number of BLAS threads. It requires the
        ``threadpoolctl`` package. If set to None, no limit is applied.
        The limit is not applied if ``ncpu`` is 1.

    Returns
    -------
    result_list
        The list of the result values of ``func``, where each element of that
        list corresponds to the arguments element in ``args_list``.

    Raises
    ------
    RuntimeError
        If a worker process raised an exception, died, or did not return its
        result.
    """
    global _IS_WORKER_PROCESS

    if _IS_WORKER_PROCESS:
        ncpu = 1

    # Create the progress bar if we are in an interactive session.
    pbar = ProgressBar(maxval=len(args_list), parent=ppbar).start()

    # Return result list if only one CPU is used.
    if ncpu == 1:

        def update_pbar(task_idx):
            if pbar.is_shown:
                pbar.update(task_idx + 1)

        result_list = _run_tasks(func, args_list, rss, tl, update_pbar)

        pbar.finish()

        return result_list

    try:
        mp_ctx = mp.get_context(_MP_START_METHOD)
    except ValueError as exc:
        raise RuntimeError(
            f'The multiprocessing start method "{_MP_START_METHOD}" is not available on this platform! Use ncpu=1.'
        ) from exc

    # Multiple CPUs are used. Split the work across multiple processes.
    # We will use our own process (pid = 0) as a worker too.
    sub_args_list_list = np.array_split(np.array(args_list, dtype=object), ncpu)

    # Create a list of RandomStateService for each process if rss argument is
    # set.
    rss_list: list[RandomStateService | None] = [rss]
    if rss is None:
        rss_list += [None] * (ncpu - 1)
    else:
        if not isinstance(rss, RandomStateService):
            raise TypeError('The rss argument must be an instance of RandomStateService!')
        rss_list.extend([RandomStateService(seed=rss.random.randint(0, 2**32)) for i in range(1, ncpu)])

    # Create a list of TimeLord instances, one for each process if tl argument
    # is set.
    tl_list: list[TimeLord | None] = [tl]
    if tl is None:
        tl_list += [None] * (ncpu - 1)
    else:
        if not isinstance(tl, TimeLord):
            raise TypeError('The tl argument must be an instance of TimeLord!')
        tl_list.extend([TimeLord() for i in range(1, ncpu)])

    rqueue = mp_ctx.Queue()
    lqueue = mp_ctx.Queue()
    squeue = mp_ctx.Queue() if pbar.is_shown else None

    # Replace all existing main process handlers with a `QueueHandler`, which
    # will be inherited by the worker processes. This sends all the log records
    # generated by worker processes to the main process, where they are handled
    # while the tasks are running. After creating the worker processes revert
    # the handlers to the initial state.
    skyllh_logger = get_logger('skyllh')
    orig_handlers = list(skyllh_logger.handlers)
    for orig_handler in orig_handlers:
        skyllh_logger.removeHandler(orig_handler)
    queue_handler = QueueHandler(lqueue)
    skyllh_logger.addHandler(queue_handler)

    # Create worker processes only for non-empty chunks of work.
    processes = {
        pid: mp_ctx.Process(
            target=_worker_main,
            args=(func, list(sub_args_list), pid, rqueue, lqueue),
            kwargs={'squeue': squeue, 'rss': rss_list[pid], 'tl': tl_list[pid], 'blas_threads': blas_threads},
        )
        for (pid, sub_args_list) in enumerate(sub_args_list_list)
        if pid > 0 and len(sub_args_list) > 0
    }

    try:
        for proc in processes.values():
            proc.start()
    finally:
        # Revert main process handlers to the initial state.
        skyllh_logger.removeHandler(queue_handler)
        for orig_handler in orig_handlers:
            skyllh_logger.addHandler(orig_handler)

    logger = get_logger(__name__)

    sarr = np.zeros((ncpu,), dtype=[('n_finished_tasks', np.int64)])
    n_finished_log_streams = 0

    def poll():
        """Handles the log records and status information sent by the worker
        processes and updates the progress bar.
        """
        nonlocal n_finished_log_streams

        while True:
            try:
                record = lqueue.get_nowait()
            except queue.Empty:
                break
            if record is None:
                n_finished_log_streams += 1
            else:
                get_logger(record.name).handle(record)

        if squeue is not None:
            while True:
                try:
                    (pid, worker_task_idx) = squeue.get_nowait()
                except queue.Empty:
                    break
                sarr[pid]['n_finished_tasks'] = worker_task_idx + 1

        if pbar.is_shown:
            pbar.update(np.sum(sarr['n_finished_tasks']))

    def master_task_done_callback(task_idx):
        sarr[0]['n_finished_tasks'] = task_idx + 1
        poll()

    def terminate_workers():
        for proc in processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in processes.values():
            proc.join()

    try:
        # Compute the first chunk in the main process. While doing so, the main
        # process acts as a worker process as well.
        _IS_WORKER_PROCESS = True
        try:
            with _limit_blas_threads(blas_threads):
                result_list_0 = _run_tasks(
                    func, list(sub_args_list_list[0]), rss_list[0], tl_list[0], master_task_done_callback
                )
        finally:
            _IS_WORKER_PROCESS = False

        # Gather the results of the worker processes and join their TimeLord
        # instances with the main TimeLord instance.
        pid_result_list_map = {0: result_list_0}
        pending_pids = set(processes.keys())
        exit_times: dict[int, float] = {}
        while len(pending_pids) > 0:
            try:
                (pid, result_list, proc_tl, error) = rqueue.get(timeout=_POLL_INTERVAL)
            except queue.Empty:
                poll()
                for pid in pending_pids:
                    proc = processes[pid]
                    if proc.exitcode is None:
                        continue
                    if proc.exitcode != 0:
                        raise RuntimeError(
                            f'Worker process {proc.pid} died with exit code {proc.exitcode}! A negative exit code '
                            'means the process was killed by a signal, e.g. -9 (SIGKILL) by the out-of-memory killer.'
                        ) from None
                    # The process exited successfully, but its result was not
                    # received (yet).
                    exit_time = exit_times.setdefault(pid, time.monotonic())
                    if time.monotonic() - exit_time > _RESULT_TIMEOUT:
                        raise RuntimeError(
                            f'Worker process {proc.pid} finished without returning its result! '
                            'Probably the result of the function could not be pickled.'
                        ) from None
                continue

            if error is not None:
                raise RuntimeError(f'Worker process {processes[pid].pid} raised an exception:\n{error}')

            pending_pids.remove(pid)
            pid_result_list_map[pid] = result_list
            if tl is not None and proc_tl is not None:
                tl.join(proc_tl)
    except BaseException:
        terminate_workers()
        raise

    # Wait for the remaining log records of the worker processes.
    last_record_time = time.monotonic()
    while n_finished_log_streams < len(processes):
        n_before = n_finished_log_streams
        poll()
        if n_finished_log_streams == n_before:
            if time.monotonic() - last_record_time > _RESULT_TIMEOUT:
                logger.warning('Not all log records of the worker processes could be received!')
                break
            time.sleep(0.01)
        else:
            last_record_time = time.monotonic()

    # Join all the processes.
    for proc in processes.values():
        proc.join()

    # Order the result lists.
    result_list = []
    for pid in sorted(pid_result_list_map.keys()):
        result_list += pid_result_list_map[pid]

    pbar.finish()

    return result_list


class IsParallelizable:
    """Classifier class defining the ncpu property. Classes that derive from
    this class indicate, that they can make use of multi-processing on several
    CPUs at the same time.
    """

    def __init__(
        self,
        *args,
        ncpu: NCpuSetting | None = None,
        **kwargs,
    ):
        """Creates a new instance of IsParallelizable.

        Parameters
        ----------
        ncpu
            The number of CPUs to utilize. If set to ``None``, the global
            setting will be used. If set to ``'auto'``, the number of CPUs
            available to the current process will be used.
        """
        super().__init__(*args, **kwargs)

        if not isinstance(self, HasConfig):
            raise TypeError(f'The class "{classname(self)}" is not derived from skyllh.core.config.HasConfig!')

        self.ncpu = ncpu

    @property
    def ncpu(self) -> int:
        """The number (int) of CPUs to utilize. It calls the ``get_ncpu``
        utility function with this property as argument. Hence, if this property
        is set to None, the global NCPU setting will take precedence.
        """
        return get_ncpu(cast(HasConfig, self).cfg, self._ncpu)

    @ncpu.setter
    def ncpu(self, n):
        if n is not None and n != 'auto':
            if not isinstance(n, int):
                raise TypeError("The ncpu property must be of type int or 'auto'!")
            if n < 1:
                raise ValueError('The ncpu property must be >= 1!')
        self._ncpu: NCpuSetting | None = n
