import numpy as np
from xtrack import TwissTable
from typing import Sequence


def _propagate_error_phase(errb0, erra0, dphi, bet0, alf0):
    return np.sqrt(
        (
            (
                (1 / 2.0 * np.cos(4 * np.pi * dphi) * alf0 / bet0)
                - (1 / 2.0 * np.sin(4 * np.pi * dphi) / bet0)
                - (1 / 2.0 * alf0 / bet0)
            )
            * errb0
        )
        ** 2
        + ((-(1 / 2.0 * np.cos(4 * np.pi * dphi)) + (1 / 2.0)) * erra0) ** 2
    ) / (2 * np.pi)


def _propagate_error_phase2(errb0, erra0, dphi, bet0, alf0):
    # NOTE: Written as in OMC3 SBS
    sin2phi = np.sin(4 * np.pi * dphi)
    cos2phi = np.cos(4 * np.pi * dphi)

    res = np.sqrt(
        (0.5 * (((cos2phi - 1) * alf0) - sin2phi) * errb0 / bet0) ** 2
        + (0.5 * (cos2phi - 1) * erra0) ** 2
    ) / (2 * np.pi)
    return res


def _propagate_error_dispersion(std_D0, bet0, bets, dphi, alf0):
    return np.abs(
        std_D0
        * np.sqrt(bets / bet0)
        * (np.cos(2 * np.pi * dphi) + alf0 * np.sin(2 * np.pi * dphi))
    )


def _get_tw_phase(tw, loc0, loc, plane):
    ph = (
        getattr(tw.rows[loc[0]], f"mu{plane}")
        - getattr(tw.rows[loc0[0]], f"mu{plane}")[0]
    ) % 1
    return ph


def _get_mes_phase(mes_all, loc0, loc, plane):
    mes = (
        getattr(mes_all.loc[loc[1]], f"PHASE{plane.upper()}").values
        - getattr(mes_all.loc[loc0[1]], f"PHASE{plane.upper()}")
    ) % 1
    return mes


def _phase_difference(mu1, mu2):
    mu_diff = (mu1 - mu2) % 1
    mu_diff = np.where(mu_diff > 0.5, mu_diff - 1, mu_diff)
    return mu_diff


def _get_mu_diff(tw, mes_all, loc0, loc, plane):
    mdl_mu = _get_tw_phase(tw, loc0, loc, plane)
    mes_mu = _get_mes_phase(mes_all, loc0, loc, plane)
    mu_diff = (mes_mu - mdl_mu) % 1
    mu_diff = np.where(mu_diff > 0.5, mu_diff - 1, mu_diff)
    return mu_diff


def merge_tw(
    tw1: TwissTable,
    tw2: TwissTable,
    var_names1: Sequence[str],
    var_names2: Sequence[str],
):
    res = {}
    res["name"] = tw1.name
    res["s"] = tw1.s

    for nv in var_names1:
        res[nv] = tw1[nv]
    for nv in var_names2:
        res[nv] = tw2[nv]

    return TwissTable(res)


def coupling_rdts(sbs):
    """
    Propagate f1001 and f1010 from bpm to bpm
    """
    bpms = sbs.get_common_bpms()
    tw_bpms = sbs.twiss_sbs().rows[bpms[0]]

    sbetx, sbety = np.sqrt(tw_bpms.betx), np.sqrt(tw_bpms.bety)
    alfx, alfy = tw_bpms.alfx, tw_bpms.alfy

    N = bpms.shape[1]
    f1001 = np.zeros(N, dtype=np.complex128)
    f1010 = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        try:
            _, rmat = sbs.get_tw_init(at_ele=bpms[0, i])
            r11 = rmat[0, 0]
            r12 = rmat[0, 1]
            r21 = rmat[1, 0]
            r22 = rmat[1, 1]

            r = np.array([(r22, -r12), (-r21, r11)])
            ga = np.array([(1 / sbetx[i], 0), (alfx[i] / sbetx[i], sbetx[i])])
            gb = np.array([(sbety[i], 0), (-alfy[i] / sbety[i], 1 / sbety[i])])
            c = r / np.sqrt(1 + np.linalg.det(r))
            cbar = np.matmul(ga, np.matmul(c, gb))
            cb = 0.25 / np.sqrt(1 - np.linalg.det(cbar)) * cbar

            f1001[i] = cb[0, 1] - cb[1, 0] + 1j * (cb[0, 0] + cb[1, 1])
            f1010[i] = -cb[0, 1] - cb[1, 0] + 1j * (cb[0, 0] - cb[1, 1])
        except:
            # NOTE: HACKING AROUND
            f1001[i] = 0
            f1010[i] = 0

    res = {"name": tw_bpms.name, "s": tw_bpms.s, "f1001": f1001, "f1010": f1010}

    return TwissTable(res)
