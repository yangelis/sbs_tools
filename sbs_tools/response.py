import xtrack as xt
import numpy as np
from typing import Sequence


from .segment import Segment


def create_response(
    segment: Segment,
    magnet_names: Sequence[str],
    bpms: np.ndarray,
    nknobs: int,
    attr="k1",
    delta_k=2e-5,
) -> dict[str, np.ndarray]:
    dxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    dmuxs = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)
    dmuys = np.zeros(shape=(bpms.shape[1], nknobs), dtype=np.float64)

    tw_sbs = segment.twiss_sbs()

    for i, mname in enumerate(magnet_names):
        old_val = getattr(segment.line.element_dict[mname], attr)
        setattr(segment.line.element_dict[mname], attr, old_val + delta_k)

        tw_dk = segment.twiss_sbs()

        old_val = getattr(segment.line.element_dict[mname], attr)
        setattr(segment.line.element_dict[mname], attr, old_val - delta_k)

        dxs[:, i] = (tw_dk.rows[bpms[0]].dx - tw_sbs.rows[bpms[0]].dx) / delta_k
        dys[:, i] = (tw_dk.rows[bpms[0]].dy - tw_sbs.rows[bpms[0]].dy) / delta_k

        dmuxs[:, i] = (tw_dk.rows[bpms[0]].mux - tw_sbs.rows[bpms[0]].mux) / delta_k
        dmuys[:, i] = (tw_dk.rows[bpms[0]].muy - tw_sbs.rows[bpms[0]].muy) / delta_k

    return {
        "DX": dxs,
        "DY": dys,
        "DMUX": dmuxs,
        "DMUY": dmuys,
    }


def tw_strengths_deltak(
    sbs: Segment, magnet_names: Sequence[str], dks: Sequence[float], attr="k1"
) -> xt.TwissTable:
    for i, imq in enumerate(magnet_names):
        old_val = getattr(sbs.line.element_dict[imq], attr)
        setattr(sbs.line.element_dict[imq], attr, old_val + dks[i])

    tw_dk = sbs.twiss_sbs()

    for i, imq in enumerate(magnet_names):
        old_val = getattr(sbs.line.element_dict[imq], attr)
        setattr(sbs.line.element_dict[imq], attr, old_val - dks[i])

    return tw_dk
