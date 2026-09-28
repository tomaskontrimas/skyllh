import unittest

import skyllh
from skyllh.core import config as config_module
from skyllh.core.config import (
    Config,
    get_current_config,
    get_default_config,
    resolve_config,
    set_default_config,
    use_config,
)
from skyllh.core.minimizer import LBFGSMinimizerImpl
from skyllh.core.multiproc import get_ncpu, parallelize


def create_marked_config(marker):
    cfg = Config()
    cfg['marker'] = marker
    return cfg


def get_current_marker(x):
    return get_current_config().get('marker')


def get_marker_of_new_object(x):
    return LBFGSMinimizerImpl().cfg.get('marker')


def get_marker_of_object(obj):
    return obj.cfg.get('marker')


class ConfigContextTestCase(unittest.TestCase):
    def setUp(self):
        # Make sure the tests do not alter the process-wide default config.
        self._orig_default_config = config_module._DEFAULT_CONFIG
        config_module._DEFAULT_CONFIG = None

    def tearDown(self):
        config_module._DEFAULT_CONFIG = self._orig_default_config

    def test_default_config(self):
        cfg = get_default_config()
        self.assertIsInstance(cfg, Config)
        self.assertIs(get_default_config(), cfg)
        self.assertIs(get_current_config(), cfg)
        self.assertIs(resolve_config(), cfg)

    def test_set_default_config(self):
        cfg = create_marked_config('default')
        set_default_config(cfg)
        self.assertIs(get_current_config(), cfg)
        self.assertIs(LBFGSMinimizerImpl().cfg, cfg)

        with self.assertRaises(TypeError):
            set_default_config({})  # type: ignore[arg-type]

    def test_use_config(self):
        cfg1 = create_marked_config(1)
        cfg2 = create_marked_config(2)
        default_cfg = get_default_config()

        with use_config(cfg1) as cfg:
            self.assertIs(cfg, cfg1)
            self.assertIs(get_current_config(), cfg1)
            with use_config(cfg2):
                self.assertIs(get_current_config(), cfg2)
            self.assertIs(get_current_config(), cfg1)
        self.assertIs(get_current_config(), default_cfg)

        with self.assertRaises(TypeError), use_config({}):  # type: ignore[arg-type]
            pass

    def test_use_config_is_reset_on_exception(self):
        cfg = create_marked_config(1)
        default_cfg = get_default_config()
        with self.assertRaises(RuntimeError), use_config(cfg):
            raise RuntimeError()
        self.assertIs(get_current_config(), default_cfg)

    def test_resolve_config_precedence(self):
        explicit_cfg = create_marked_config('explicit')
        obj_cfg = create_marked_config('obj')
        context_cfg = create_marked_config('context')
        obj = LBFGSMinimizerImpl(cfg=obj_cfg)

        with use_config(context_cfg):
            self.assertIs(resolve_config(explicit_cfg, objs=[obj]), explicit_cfg)
            self.assertIs(resolve_config(None, objs=['not a HasConfig', obj]), obj_cfg)
            self.assertIs(resolve_config(None, objs=[]), context_cfg)
            self.assertIs(resolve_config(), context_cfg)

        with self.assertRaises(TypeError):
            resolve_config({})  # type: ignore[arg-type]

    def test_has_config_without_cfg(self):
        cfg = create_marked_config('context')
        with use_config(cfg):
            self.assertIs(LBFGSMinimizerImpl().cfg, cfg)

        # An explicit Config instance takes precedence.
        explicit_cfg = create_marked_config('explicit')
        with use_config(cfg):
            self.assertIs(LBFGSMinimizerImpl(cfg=explicit_cfg).cfg, explicit_cfg)

    def test_get_ncpu(self):
        cfg = create_marked_config('context')
        cfg.set_ncpu(3)
        with use_config(cfg):
            self.assertEqual(get_ncpu(), 3)
            self.assertEqual(get_ncpu(local_ncpu=2), 2)

    def test_create_datasets(self):
        cfg = create_marked_config('context')
        with use_config(cfg):
            datasets = skyllh.create_datasets('TestData')
        self.assertGreater(len(datasets), 0)
        for ds in datasets:
            self.assertIs(ds.cfg, cfg)

        explicit_cfg = create_marked_config('explicit')
        datasets = skyllh.create_datasets('TestData', cfg=explicit_cfg)
        for ds in datasets:
            self.assertIs(ds.cfg, explicit_cfg)

    def test_create_analysis_requires_datasets_and_source(self):
        from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis

        with self.assertRaisesRegex(TypeError, 'datasets, source'):
            create_analysis()


class ConfigContextMultiprocessingTestCase(unittest.TestCase):
    """The Config instance must be available within worker processes of
    parallelize.
    """

    def test_worker_processes_inherit_current_config(self):
        cfg = create_marked_config('context')
        args_list = [((x,), {}) for x in range(4)]
        with use_config(cfg):
            markers = parallelize(func=get_current_marker, args_list=args_list, ncpu=2)
            new_object_markers = parallelize(func=get_marker_of_new_object, args_list=args_list, ncpu=2)
        self.assertEqual(markers, ['context'] * 4)
        self.assertEqual(new_object_markers, ['context'] * 4)

    def test_objects_keep_their_config_in_worker_processes(self):
        cfg = create_marked_config('object')
        with use_config(cfg):
            obj = LBFGSMinimizerImpl()
        # The object keeps its Config instance, even outside the context.
        markers = parallelize(func=get_marker_of_object, args_list=[((obj,), {}) for _ in range(4)], ncpu=2)
        self.assertEqual(markers, ['object'] * 4)


if __name__ == '__main__':
    unittest.main()
