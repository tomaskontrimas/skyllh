import logging
import os
import signal
import unittest
from unittest.mock import patch

from skyllh.core import multiproc
from skyllh.core.config import Config
from skyllh.core.logging import get_logger
from skyllh.core.multiproc import (
    get_available_ncpu,
    get_ncpu,
    parallelize,
)
from skyllh.core.random import RandomStateService


def square(x):
    return x**2


def uniform(x, rss):
    return rss.random.uniform()


def raise_for_last(x, n):
    if x == n - 1:
        raise ValueError(f'Bad value {x}!')
    return x


def kill_for_last(x, n):
    if x == n - 1:
        os.kill(os.getpid(), signal.SIGKILL)
    return x


def unpicklable_for_last(x, n):
    if x == n - 1:
        return lambda: x
    return x


def log_pid(x):
    get_logger('skyllh.test_multiproc').info('Task %d on pid %d', x, os.getpid())
    return os.getpid()


def nested_pids(x):
    return parallelize(func=get_pid, args_list=[((i,), {}) for i in range(4)], ncpu=4)


def get_pid(x):
    return os.getpid()


def get_max_blas_threads(x):
    from threadpoolctl import threadpool_info

    return max((info['num_threads'] for info in threadpool_info()), default=None)


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class ParallelizeTestCase(unittest.TestCase):
    def test_results_are_ordered(self):
        n = 11
        args_list = [((x,), {}) for x in range(n)]
        for ncpu in (1, 2, 3, 4, 16):
            with self.subTest(ncpu=ncpu):
                result = parallelize(func=square, args_list=args_list, ncpu=ncpu)
                self.assertEqual(result, [x**2 for x in range(n)])

    def test_seeded_results_are_unchanged(self):
        """The random numbers for a given seed and ncpu must be the same as for
        the previous implementation of parallelize, to keep trial results
        reproducible.
        """
        args_list = [((x,), {}) for x in range(10)]
        expected = {
            1: [0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366],
            3: [0.9507143064099162, 0.7319939418114051, 0.5986584841970366, 0.15601864044243652],
        }
        for ncpu, values in expected.items():
            with self.subTest(ncpu=ncpu):
                result = parallelize(func=uniform, args_list=args_list, ncpu=ncpu, rss=RandomStateService(seed=42))
                self.assertEqual(result[: len(values)], values)

    def test_does_not_modify_kwargs(self):
        kwargs = {}
        parallelize(func=uniform, args_list=[((1,), kwargs)], ncpu=1, rss=RandomStateService(seed=1))
        self.assertEqual(kwargs, {})

    def test_worker_exception_is_raised_with_traceback(self):
        n = 6
        args_list = [((x,), {'n': n}) for x in range(n)]
        with self.assertRaisesRegex(RuntimeError, r'(?s)raised an exception.*ValueError: Bad value 5!'):
            parallelize(func=raise_for_last, args_list=args_list, ncpu=3)

    def test_killed_worker_is_detected(self):
        n = 6
        args_list = [((x,), {'n': n}) for x in range(n)]
        with self.assertRaisesRegex(RuntimeError, 'exit code -9'):
            parallelize(func=kill_for_last, args_list=args_list, ncpu=3)

    def test_unpicklable_result_is_detected(self):
        n = 6
        args_list = [((x,), {'n': n}) for x in range(n)]
        with patch.object(multiproc, '_RESULT_TIMEOUT', 1), self.assertRaisesRegex(RuntimeError, 'pickled'):
            parallelize(func=unpicklable_for_last, args_list=args_list, ncpu=3)

    def test_worker_log_records_are_forwarded(self):
        logger = get_logger('skyllh')
        handler = ListHandler()
        orig_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            n = 8
            pids = parallelize(func=log_pid, args_list=[((x,), {}) for x in range(n)], ncpu=4)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(orig_level)

        messages = sorted(r.getMessage() for r in handler.records if r.name == 'skyllh.test_multiproc')
        self.assertEqual(messages, sorted(f'Task {x} on pid {pid}' for (x, pid) in enumerate(pids)))
        self.assertEqual(len(set(pids)), 4)
        # The original handlers must be restored.
        self.assertNotIn('QueueHandler', [type(h).__name__ for h in logger.handlers])

    def test_nested_parallelize_runs_serially(self):
        result = parallelize(func=nested_pids, args_list=[((x,), {}) for x in range(2)], ncpu=2)
        for pids in result:
            self.assertEqual(len(set(pids)), 1)

    def test_blas_threads_are_limited(self):
        try:
            import threadpoolctl  # noqa: F401
        except ImportError:
            self.skipTest('threadpoolctl is not installed.')

        n_threads = parallelize(func=get_max_blas_threads, args_list=[((x,), {}) for x in range(4)], ncpu=2)
        if n_threads[0] is None:
            self.skipTest('No BLAS / OpenMP thread pool found.')
        self.assertEqual(n_threads, [1, 1, 1, 1])


class NCpuTestCase(unittest.TestCase):
    def test_get_available_ncpu(self):
        ncpu = get_available_ncpu()
        self.assertGreaterEqual(ncpu, 1)
        if hasattr(os, 'sched_getaffinity'):
            self.assertEqual(ncpu, len(os.sched_getaffinity(0)))

    def test_get_ncpu(self):
        cfg = Config()
        self.assertEqual(get_ncpu(cfg, None), 1)
        self.assertEqual(get_ncpu(cfg, 3), 3)
        self.assertEqual(get_ncpu(cfg, 'auto'), get_available_ncpu())

        cfg.set_ncpu('auto')
        self.assertEqual(get_ncpu(cfg, None), get_available_ncpu())
        self.assertEqual(get_ncpu(cfg, 2), 2)

    def test_get_ncpu_invalid(self):
        cfg = Config()
        with self.assertRaises(ValueError):
            get_ncpu(cfg, 0)
        with self.assertRaises(TypeError):
            get_ncpu(cfg, 'all')  # type: ignore[arg-type]


if __name__ == '__main__':
    unittest.main()
