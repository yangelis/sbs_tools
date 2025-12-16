from xtrack import TwissTable
import numpy as np
from typing import Sequence


from .segment import Segment
from .utils import coupling_rdts


def _create_res_array(n1: int, n2: int):
    betax = np.zeros(shape=(n1, n2), dtype=np.float64)
    betay = np.zeros(shape=(n1, n2), dtype=np.float64)

    betabeatx = np.zeros(shape=(n1, n2), dtype=np.float64)
    betabeaty = np.zeros(shape=(n1, n2), dtype=np.float64)

    dxs = np.zeros(shape=(n1, n2), dtype=np.float64)
    dys = np.zeros(shape=(n1, n2), dtype=np.float64)

    ndxs = np.zeros(shape=(n1, n2), dtype=np.float64)
    ndys = np.zeros(shape=(n1, n2), dtype=np.float64)

    dmuxs = np.zeros(shape=(n1, n2), dtype=np.float64)
    dmuys = np.zeros(shape=(n1, n2), dtype=np.float64)

    return betax, betay, betabeatx, betabeaty, dxs, dys, ndxs, ndys, dmuxs, dmuys


def _fill_res_array(
    i: int,
    twk: TwissTable,
    twm: TwissTable,
    deltak: float,
    betax: np.ndarray,
    betay: np.ndarray,
    betabeatx: np.ndarray,
    betabeaty: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
    ndxs: np.ndarray,
    ndys: np.ndarray,
    dmuxs: np.ndarray,
    dmuys: np.ndarray,
):
    betax[:, i] = twk.betx
    betay[:, i] = twk.bety

    betabeatx[:, i] = ((betax[:, i] - twm.betx) / twm.betx) / deltak
    betabeaty[:, i] = ((betay[:, i] / twm.bety) / twm.bety) / deltak

    dxs[:, i] = (twk.dx - twm.dx) / deltak
    dys[:, i] = (twk.dy - twm.dy) / deltak

    ndxs[:, i] = (twk.dx / np.sqrt(twk.betx) - twm.dx / np.sqrt(twm.betx)) / deltak
    ndys[:, i] = (twk.dy / np.sqrt(twk.bety) - twm.dy / np.sqrt(twm.bety)) / deltak

    dmuxs[:, i] = (twk.mux - twm.mux) / deltak
    dmuys[:, i] = (twk.muy - twm.muy) / deltak
    return


def create_response(
    segment: Segment,
    magnet_names: Sequence[str],
    bpms: np.ndarray,
    attr: str = "k1",
    deltak: float = 2e-5,
    mode: str = "front",
) -> dict[str, np.ndarray]:

    nknobs = len(magnet_names)

    betax, betay, betabeatx, betabeaty, dxs, dys, ndxs, ndys, dmuxs, dmuys = (
        _create_res_array(bpms.shape[1], nknobs)
    )

    xs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    ys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    df1001 = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.complex128)
    df1010 = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.complex128)

    tw_sbs = segment.twiss(mode=mode)
    tw_mdl_cpl = coupling_rdts(segment)

    for i, mname in enumerate(magnet_names):
        original_val = getattr(segment.line.element_dict[mname], attr)
        setattr(segment.line.element_dict[mname], attr, original_val + deltak)

        tw_dk = segment.twiss(mode=mode)
        tw_cpl = coupling_rdts(segment)

        setattr(segment.line.element_dict[mname], attr, original_val)

        _fill_res_array(
            i,
            tw_dk.rows[bpms[0, :]],
            tw_sbs.rows[bpms[0, :]],
            deltak,
            betax,
            betay,
            betabeatx,
            betabeaty,
            dxs,
            dys,
            ndxs,
            ndys,
            dmuxs,
            dmuys,
        )
        xs[:, i] = tw_dk.rows[bpms[0, :]].x
        ys[:, i] = tw_dk.rows[bpms[0, :]].y
        df1001[:, i] = tw_cpl.f1001 - tw_mdl_cpl.f1001
        df1010[:, i] = tw_cpl.f1010 - tw_mdl_cpl.f1010

    return {
        "X": xs,
        "Y": ys,
        "BETX": betax,
        "BETY": betay,
        "DBETX": betabeatx,
        "DBETY": betabeaty,
        "DX": dxs,
        "DY": dys,
        "NDX": ndxs,
        "NDY": ndys,
        "DMUX": dmuxs,
        "DMUY": dmuys,
        "F1001R": np.real(df1001),
        "F1010R": np.real(df1010),
        "F1001I": np.imag(df1001),
        "F1010I": np.imag(df1010),
    }


def create_knob_response(
    segment: Segment,
    knob_names: Sequence[str],
    bpms: np.ndarray,
    deltak: float = 2e-5,
    mode: str = "front",
) -> dict[str, np.ndarray]:

    if mode not in ["front", "back"]:
        print("Available modes are 'front' or 'back' ")
    nknobs = len(knob_names)

    betax, betay, betabeatx, betabeaty, dxs, dys, ndxs, ndys, dmuxs, dmuys = (
        _create_res_array(bpms.shape[1], nknobs)
    )

    tw_sbs = segment.twiss(mode=mode)

    for i, kname in enumerate(knob_names):
        # NOTE: str(kname) instead of kname to avoid np.str_
        original_val = segment.line.varval[str(kname)]
        segment.line.vars[str(kname)] += deltak

        tw_dk = segment.twiss(mode=mode)

        segment.line.vars[str(kname)] = original_val

        _fill_res_array(
            i,
            tw_dk.rows[bpms[0, :]],
            tw_sbs.rows[bpms[0, :]],
            deltak,
            betax,
            betay,
            betabeatx,
            betabeaty,
            dxs,
            dys,
            ndxs,
            ndys,
            dmuxs,
            dmuys,
        )

    return {
        "DBETX": betabeatx,
        "DBETY": betabeaty,
        "DX": dxs,
        "DY": dys,
        "NDX": ndxs,
        "NDY": ndys,
        "DMUX": dmuxs,
        "DMUY": dmuys,
    }


def tw_strengths_deltak(
    sbs: Segment,
    magnet_names: Sequence[str],
    dks: Sequence[float],
    mode: str = "front",
    attr: str = "k1",
) -> TwissTable:

    original_values = []
    for i, imq in enumerate(magnet_names):
        old_val = getattr(sbs.line.element_dict[imq], attr)
        original_values.append(old_val)
        setattr(sbs.line.element_dict[imq], attr, old_val + dks[i])

    tw_dk = sbs.twiss(mode=mode)

    for i, imq in enumerate(magnet_names):
        setattr(sbs.line.element_dict[imq], attr, original_values[i])

    return tw_dk


def invert_response(tw, response, var_names_res, var_names_tw, rcond=0.03):
    vars_response = np.concatenate([response[vr] for vr in var_names_res])
    vars_twiss = np.concatenate([tw[vr] for vr in var_names_tw])
    dk = np.dot(np.linalg.pinv(vars_response, rcond=rcond), vars_twiss)

    return dk
