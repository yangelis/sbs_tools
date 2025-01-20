import xtrack as xt
import numpy as np
from typing import Sequence


from .segment import Segment


def create_response(
    segment: Segment,
    magnet_names: Sequence[str],
    bpms: np.ndarray,
    nknobs: int,
    attr: str = "k1",
    delta_k: float = 2e-5,
) -> dict[str, np.ndarray]:

    betax = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betay = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    betabeatx = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    betabeaty = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dmuxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dmuys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    tw_sbs = segment.twiss_sbs()

    for i, mname in enumerate(magnet_names):
        original_val = getattr(segment.line.element_dict[mname], attr)
        setattr(segment.line.element_dict[mname], attr, original_val + delta_k)

        tw_dk = segment.twiss_sbs()

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


def tw_strengths_deltak(
    sbs: Segment, magnet_names: Sequence[str], dks: Sequence[float], attr: str = "k1"
) -> xt.TwissTable:

    original_values = []
    for i, imq in enumerate(magnet_names):
        old_val = getattr(sbs.line.element_dict[imq], attr)
        original_values.append(old_val)
        setattr(sbs.line.element_dict[imq], attr, old_val + dks[i])

    tw_dk = sbs.twiss_sbs()

    for i, imq in enumerate(magnet_names):
        setattr(sbs.line.element_dict[imq], attr, original_values[i])

    return tw_dk
