import xtrack as xt
import numpy as np
from typing import Sequence


from .segment import Segment


def create_response(
    segment: Segment,
    magnet_names: Sequence[str],
    bpms: np.ndarray,
    attr: str = "k1",
    delta_k: float = 2e-5,
    mode="front",
) -> dict[str, np.ndarray]:

    nknobs = len(magnet_names)

    betax = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betay = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    betabeatx = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betabeaty = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dmuxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dmuys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    tw_sbs = segment.twiss(mode=mode)

    for i, mname in enumerate(magnet_names):
        original_val = getattr(segment.line.element_dict[mname], attr)
        setattr(segment.line.element_dict[mname], attr, original_val + delta_k)

        tw_dk = segment.twiss(mode=mode)

        setattr(segment.line.element_dict[mname], attr, original_val)

        betax[:, i] = tw_dk.rows[bpms[0, :]].betx
        betay[:, i] = tw_dk.rows[bpms[0, :]].bety

        dxs[:, i] = (tw_dk.rows[bpms[0, :]].dx - tw_sbs.rows[bpms[0, :]].dx) / delta_k
        dys[:, i] = (tw_dk.rows[bpms[0, :]].dy - tw_sbs.rows[bpms[0, :]].dy) / delta_k

        dmuxs[:, i] = (
            tw_dk.rows[bpms[0, :]].mux - tw_sbs.rows[bpms[0, :]].mux
        ) / delta_k
        dmuys[:, i] = (
            tw_dk.rows[bpms[0, :]].muy - tw_sbs.rows[bpms[0, :]].muy
        ) / delta_k

    return {
        "DX": dxs,
        "DY": dys,
        "DMUX": dmuxs,
        "DMUY": dmuys,
    }


def create_knob_response(
    segment: Segment,
    knob_names: Sequence[str],
    bpms: np.ndarray,
    delta: float = 2e-5,
    mode="front",
) -> dict[str, np.ndarray]:

    nknobs = len(knob_names)

    betax = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betay = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    betabeatx = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betabeaty = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dmuxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dmuys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    tw_sbs = segment.twiss(mode=mode)

    for i, kname in enumerate(knob_names):
        # NOTE: str(kname) instead of kname to avoid np.str_
        original_val = segment.line.varval[str(kname)]
        segment.line.vars[str(kname)] += delta

        tw_dk = segment.twiss(mode=mode)

        segment.line.vars[str(kname)] -= delta

        betax[:, i] = tw_dk.rows[bpms[0, :]].betx
        betay[:, i] = tw_dk.rows[bpms[0, :]].bety

        dxs[:, i] = (tw_dk.rows[bpms[0, :]].dx - tw_sbs.rows[bpms[0, :]].dx) / delta
        dys[:, i] = (tw_dk.rows[bpms[0, :]].dy - tw_sbs.rows[bpms[0, :]].dy) / delta

        dmuxs[:, i] = (tw_dk.rows[bpms[0, :]].mux - tw_sbs.rows[bpms[0, :]].mux) / delta
        dmuys[:, i] = (tw_dk.rows[bpms[0, :]].muy - tw_sbs.rows[bpms[0, :]].muy) / delta

    return {
        "DX": dxs,
        "DY": dys,
        "DMUX": dmuxs,
        "DMUY": dmuys,
    }


def tw_strengths_deltak(
    sbs: Segment,
    magnet_names: Sequence[str],
    dks: Sequence[float],
    mode="front",
    attr: str = "k1",
) -> xt.TwissTable:

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
