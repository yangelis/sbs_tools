import numpy as np
from xtrack import TwissTable


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


def _get_mu_diff(tw, mes_all, loc0, loc, plane):
    mdl_mu = _get_tw_phase(tw, loc0, loc, plane)
    mes_mu = _get_mes_phase(mes_all, loc0, loc, plane)
    mu_diff = (mes_mu - mdl_mu) % 1
    mu_diff = np.where(mu_diff > 0.5, mu_diff - 1, mu_diff)
    return mu_diff


def merge_tw(tw1, tw2, var_names1, var_names2):
    res = {}
    res["name"] = tw1.name
    res["s"] = tw1.s

    for nv in var_names1:
        res[nv] = tw1[nv]
    for nv in var_names2:
        res[nv] = tw2[nv]

    return TwissTable(res)
