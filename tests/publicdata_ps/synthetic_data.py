"""Generator of small synthetic datasets in the file format of the IceCube
public point-source track data releases (IceTracks-DR1/DR2).

The synthetic datasets are used to test analyses based on the public data
without the need to download the actual data. They contain background-like
events, a good-run-list with detector down-time, a smooth effective area, and a
smearing matrix. The physics content is not realistic, but the file formats,
binnings, and data handling are identical to the real data.
"""

import itertools
import os
from dataclasses import dataclass

import numpy as np

from skyllh.core.config import Config
from skyllh.core.dataset import DatasetCollection
from skyllh.i3.dataset import I3Dataset

LOG10_TRUE_E_EDGES = np.arange(2.0, 8.0 + 0.01, 0.5)
TRUE_DEC_EDGES_DEG = np.arange(-90.0, 90.0 + 0.01, 30.0)
AEFF_LOG10_E_EDGES = np.arange(2.0, 8.0 + 0.01, 0.25)
AEFF_DEC_EDGES_DEG = np.arange(-90.0, 90.0 + 0.01, 10.0)
PSI_EDGES_DEG = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0])
ANG_ERR_EDGES_DEG = np.array([0.2, 0.3, 0.45, 0.65, 0.9, 1.3, 2.0, 3.0])
N_RECO_E = 10


@dataclass
class SyntheticSeason:
    """Definition of a synthetic dataset (season).

    Attributes
    ----------
    name
        The name of the dataset.
    t_start
        The MJD start time of the dataset.
    t_stop
        The MJD stop time of the dataset.
    n_events
        The number of background events.
    aeff_scale
        The scaling factor of the effective area.
    seed
        The seed of the random number generator for the events.
    """

    name: str
    t_start: float
    t_stop: float
    n_events: int = 20000
    aeff_scale: float = 1.0
    seed: int = 1


def _aeff(log10_e: float, dec_deg: float) -> float:
    """A smooth effective area in cm^2 as function of log10(E/GeV) and dec."""
    return (
        1e2
        * 10 ** (0.8 * (log10_e - 3.0))
        / (1 + 10 ** (0.6 * (log10_e - 6.0)))
        * (1.5 + 0.5 * np.sin(np.radians(dec_deg)))
    )


def write_effective_area_file(pathfilename: str, aeff_scale: float = 1.0):
    """Writes the effective area file."""
    lines = ['# log10(E_nu/GeV)_min log10(E_nu/GeV)_max Dec_nu_min[deg] Dec_nu_max[deg] A_Eff[cm^2]']
    for d_lo, d_hi in itertools.pairwise(AEFF_DEC_EDGES_DEG):
        for e_lo, e_hi in itertools.pairwise(AEFF_LOG10_E_EDGES):
            a = aeff_scale * _aeff(0.5 * (e_lo + e_hi), 0.5 * (d_lo + d_hi))
            lines.append(f'{e_lo:.3f} {e_hi:.3f} {d_lo:.1f} {d_hi:.1f} {a:.6e}')
    with open(pathfilename, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_smearing_file(pathfilename: str):
    """Writes the smearing matrix file. The rows are ordered by true energy,
    true declination, reco energy, psi, and angular error. The fractional
    counts are normalized to unity for each (true energy, true declination)
    bin. The psi distribution follows a Rayleigh distribution with the angular
    error as width, i.e. the angular errors are well calibrated.
    """
    ang_err_c = 0.5 * (ANG_ERR_EDGES_DEG[1:] + ANG_ERR_EDGES_DEG[:-1])

    # The binned Rayleigh distribution of psi for each angular error bin.
    # p_psi_ae has the shape (n_psi, n_ang_err).
    cdf = 1 - np.exp(-0.5 * (PSI_EDGES_DEG[:, None] / ang_err_c[None, :]) ** 2)
    p_psi_ae = np.diff(cdf, axis=0)

    lines = [
        (
            '# log10(E_nu/GeV)_min log10(E_nu/GeV)_max Dec_nu_min[deg] Dec_nu_max[deg] log10(E/GeV)_min '
            'log10(E/GeV)_max PSF_min[deg] PSF_max[deg] AngErr_min[deg] AngErr_max[deg] Fractional_Counts'
        )
    ]
    for te_lo, te_hi in itertools.pairwise(LOG10_TRUE_E_EDGES):
        te_c = 0.5 * (te_lo + te_hi)
        # The reco energy bins depend on the true energy bin.
        reco_e_edges = np.linspace(te_c - 1.75, te_c + 0.75, N_RECO_E + 1)
        reco_e_c = 0.5 * (reco_e_edges[1:] + reco_e_edges[:-1])
        p_reco_e = np.exp(-0.5 * ((reco_e_c - (te_c - 0.4)) / 0.4) ** 2)
        # The angular error gets smaller with energy.
        ang_err_mean = 2.0 / (1 + (te_c - 2.0))
        p_ang_err = np.exp(-0.5 * ((ang_err_c - ang_err_mean) / 0.5) ** 2)
        for td_lo, td_hi in itertools.pairwise(TRUE_DEC_EDGES_DEG):
            p = p_reco_e[:, None, None] * p_psi_ae[None, :, :] * p_ang_err[None, None, :]
            p /= np.sum(p)
            for i_e in range(N_RECO_E):
                for i_psi in range(len(PSI_EDGES_DEG) - 1):
                    for i_ae in range(len(ang_err_c)):
                        lines.append(
                            f'{te_lo:.2f} {te_hi:.2f} {td_lo:.1f} {td_hi:.1f} '
                            f'{reco_e_edges[i_e]:.4f} {reco_e_edges[i_e + 1]:.4f} '
                            f'{PSI_EDGES_DEG[i_psi]:.2f} {PSI_EDGES_DEG[i_psi + 1]:.2f} '
                            f'{ANG_ERR_EDGES_DEG[i_ae]:.2f} {ANG_ERR_EDGES_DEG[i_ae + 1]:.2f} '
                            f'{p[i_e, i_psi, i_ae]:.6e}'
                        )
    with open(pathfilename, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def create_grl(t_start: float, t_stop: float, run_length: float = 1 / 3, gap_fraction: float = 0.1) -> np.ndarray:
    """Creates the good-run-list as (N_runs,2)-shaped array of start and stop
    times. Each run has a length of ``run_length`` days and is followed by a
    detector down-time of ``gap_fraction`` times the run length.
    """
    starts = np.arange(t_start, t_stop, run_length * (1 + gap_fraction))
    stops = np.minimum(starts + run_length, t_stop)
    m = stops > starts
    return np.vstack((starts[m], stops[m])).T


def write_season_files(base_path: str, season: SyntheticSeason, grl: np.ndarray | None = None):
    """Writes the event and uptime files of the given synthetic season.

    Parameters
    ----------
    base_path
        The directory where to write the files.
    season
        The definition of the season.
    grl
        The optional (N_runs,2)-shaped array of start and stop times. If
        ``None``, the GRL is created via :func:`create_grl`.
    """
    if grl is None:
        grl = create_grl(season.t_start, season.t_stop)

    rng = np.random.default_rng(season.seed)
    n = season.n_events

    # Draw the event times uniformly during the detector on-time.
    run_lengths = grl[:, 1] - grl[:, 0]
    run_idxs = rng.choice(len(grl), size=n, p=run_lengths / np.sum(run_lengths))
    times = np.sort(rng.uniform(grl[run_idxs, 0], grl[run_idxs, 1]))
    run_idxs = np.searchsorted(grl[:, 0], times, side='right') - 1

    sin_dec = rng.uniform(-0.99, 0.99, n)
    dec = np.degrees(np.arcsin(sin_dec))
    ra = rng.uniform(0, 360, n)
    log10_e = np.clip(2.0 + rng.exponential(0.5, n) + 0.3 * (1 - sin_dec), 2.0, 7.9)
    ang_err = np.clip(rng.lognormal(np.log(0.7), 0.5, n), 0.2, 15.0)
    azi = rng.uniform(0, 360, n)
    zen = dec + 90.0

    os.makedirs(os.path.join(base_path, 'events'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'uptime'), exist_ok=True)

    lines = ['# run event subevent MJD[days] log10(E/GeV) AngErr[deg] RA[deg] Dec[deg] Azimuth[deg] Zenith[deg]']
    for i in range(n):
        lines.append(
            f'{100000 + run_idxs[i]} {i} 0 {times[i]:.8f} {log10_e[i]:.4f} {ang_err[i]:.4f} '
            f'{ra[i]:.5f} {dec[i]:.5f} {azi[i]:.5f} {zen[i]:.5f}'
        )
    with open(os.path.join(base_path, 'events', f'{season.name}_exp.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    lines = ['# run MJD_start[days] MJD_stop[days] livetime[days] n_events']
    for i, (start, stop) in enumerate(grl):
        n_events = np.count_nonzero(run_idxs == i)
        lines.append(f'{100000 + i} {start:.8f} {stop:.8f} {stop - start:.8f} {n_events}')
    with open(os.path.join(base_path, 'uptime', f'{season.name}_exp.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_synthetic_data(base_path: str, seasons: list[SyntheticSeason], grls: dict | None = None):
    """Writes all the files of the given synthetic seasons into the given base
    path.

    Parameters
    ----------
    base_path
        The directory where to write the files.
    seasons
        The list of season definitions.
    grls
        The optional dictionary mapping season names to (N_runs,2)-shaped GRL
        arrays.
    """
    if grls is None:
        grls = {}

    os.makedirs(os.path.join(base_path, 'irfs'), exist_ok=True)
    write_smearing_file(os.path.join(base_path, 'irfs', 'smearing.csv'))
    for season in seasons:
        write_effective_area_file(
            os.path.join(base_path, 'irfs', f'{season.name}_effectiveArea.csv'), aeff_scale=season.aeff_scale
        )
        write_season_files(base_path=base_path, season=season, grl=grls.get(season.name))


def create_dataset_collection(cfg: Config, base_path: str, seasons: list[SyntheticSeason]) -> DatasetCollection:
    """Creates the dataset collection for the synthetic seasons, which were
    written into the given base path via :func:`write_synthetic_data`.

    Parameters
    ----------
    cfg
        The instance of Config holding the local configuration.
    base_path
        The directory holding the synthetic data files.
    seasons
        The list of season definitions.

    Returns
    -------
    dsc
        The instance of DatasetCollection.
    """
    dsc = DatasetCollection('Synthetic public point-source data')
    dsc.description = 'Synthetic data in the public point-source track data format for testing.'

    datasets = []
    for season in seasons:
        ds = I3Dataset(
            cfg=cfg,
            name=season.name,
            exp_pathfilenames=f'events/{season.name}_exp.csv',
            mc_pathfilenames=None,
            grl_pathfilenames=f'uptime/{season.name}_exp.csv',
            version=1,
            verqualifiers={'p': 0},
            base_path=base_path,
            default_sub_path_fmt='',
            sub_path_fmt='',
        )
        ds.grl_field_name_renaming_dict = {
            'MJD_start[days]': 'start',
            'MJD_stop[days]': 'stop',
            'livetime[days]': 'livetime',
        }
        ds.add_aux_data_definition('eff_area_datafile', f'irfs/{season.name}_effectiveArea.csv')
        ds.add_aux_data_definition('smearing_datafile', 'irfs/smearing.csv')
        ds.define_binning('sin_dec', np.linspace(-1, 1, 21))
        ds.define_binning('log_energy', np.arange(1.0, 9.0 + 0.01, 0.25))
        ds.add_aux_data('spline_smoothing', 0.5)
        ds.add_aux_data('cumulative_threshold', 1e-3)
        datasets.append(ds)

    dsc.add_datasets(tuple(datasets))

    dsc.set_exp_field_name_renaming_dict(
        {
            'MJD[days]': 'time',
            'log10(E/GeV)': 'log_energy',
            'AngErr[deg]': 'ang_err',
            'RA[deg]': 'ra',
            'Dec[deg]': 'dec',
            'Azimuth[deg]': 'azi',
            'Zenith[deg]': 'zen',
        }
    )

    def convert_deg2rad(data):
        exp = data.exp
        for name in ('ang_err', 'ra', 'dec', 'azi', 'zen'):
            if name in exp:
                exp[name] = np.deg2rad(exp[name])

    dsc.add_data_preparation(convert_deg2rad)

    return dsc
