import xtrack as xt
import numpy as np
from .readers import OptData
from .utils import (
    _get_mu_diff,
    _propagate_error_phase,
    _propagate_error_phase2,
    _propagate_error_dispersion,
    _phase_difference,
    mdl2twiss,
    get_R_terms,
)

from typing import Sequence


class SegmentFromTwiss:
    def __init__(
        self,
        mdl_twiss,
        start_bpm: str,
        end_bpm: str,
        mes: OptData,
    ) -> None:
        self.mdl = mdl_twiss
        self.mes = mes
        self.start_bpm = start_bpm
        self.end_bpm = end_bpm

        self.tw_sbs = mdl2twiss(mdl_twiss)

        init_start, rmat_start = self.get_tw_init(at_ele=self.start_bpm)
        init_end, rmat_end = self.get_tw_init(at_ele=self.end_bpm)
        self.init_start = init_start
        self.init_end = init_end
        self.rmat_start = rmat_start
        self.rmat_end = rmat_end

    def get_common_bpms(self, attr: str = "total_phase_"):
        tw_sbs_bpms = self.tw_sbs.rows["bpm.*"]
        bpms_names = np.array([[b, b.upper()] for b in tw_sbs_bpms.name]).T

        common_bpm_mes = getattr(self.mes, f"{attr}x").index.intersection(
            getattr(self.mes, f"{attr}y").index
        )
        common_bpms = bpms_names[1][np.isin(bpms_names[1], common_bpm_mes)]

        bpms_names = np.array([[b.lower(), b] for b in common_bpms]).T
        return bpms_names

    def get_s_and_bpms(self, attr: str = "total_phase_"):
        tw_sbs_bpms = self.tw_sbs.rows["bpm.*"]
        bpms_names = np.array([[b, b.upper()] for b in tw_sbs_bpms.name]).T

        common_bpms_x = bpms_names[
            np.isin(bpms_names, getattr(self.mes, f"{attr}x").index)
        ]
        common_bpms_y = bpms_names[
            np.isin(bpms_names, getattr(self.mes, f"{attr}y").index)
        ]

        bpms_x = np.array([[b.lower(), b] for b in common_bpms_x]).T
        bpms_y = np.array([[b.lower(), b] for b in common_bpms_y]).T

        mes_sx = getattr(self.mes, f"{attr}x").loc[bpms_x[1]].S
        mes_sy = getattr(self.mes, f"{attr}y").loc[bpms_y[1]].S

        res = {
            "bpms_x": bpms_x,
            "bpms_y": bpms_y,
            "mes_sx": mes_sx,
            "mes_sy": mes_sy,
        }
        return res

    def get_tw_init(self, at_ele: str) -> tuple[xt.TwissInit, np.ndarray]:
        """
        Return the twiss init at 'at_ele' element
        and the coupling matrix, derived from the measurement
        """
        BPM = at_ele.upper()
        dpx_ini = 0
        dpy_ini = 0
        dx_ini = 0
        dy_ini = 0

        phix_ini = 0.0
        phiy_ini = 0.0
        wx_ini = 0
        wy_ini = 0

        f_ini_bbs = {}

        mes_betx = self.mes.betx.loc[BPM]
        mes_bety = self.mes.bety.loc[BPM]
        mes_f1001 = self.mes.f1001.loc[BPM]
        mes_f1010 = self.mes.f1010.loc[BPM]

        betx_ini = mes_betx.BETX
        bety_ini = mes_bety.BETY
        alfx_ini = mes_betx.ALFX
        alfy_ini = mes_bety.ALFY

        f_ini_bbs["f1001r"] = mes_f1001.REAL
        f_ini_bbs["f1001i"] = mes_f1001.IMAG
        f_ini_bbs["f1010r"] = mes_f1010.REAL
        f_ini_bbs["f1010i"] = mes_f1010.IMAG

        if self.mes.has_dispersion:
            mes_dx = self.mes.data_dx.loc[BPM]
            mes_dy = self.mes.data_dy.loc[BPM]
            dx_ini = mes_dx.DX
            dy_ini = mes_dy.DY
            dpx_ini = mes_dx.DPX
            dpy_ini = mes_dy.DPY

        # ax_chrom = wx_ini * np.cos(phix_ini)
        # bx_chrom = wx_ini * np.sin(phix_ini)
        # ay_chrom = wy_ini * np.cos(phiy_ini)
        # by_chrom = wy_ini * np.sin(phiy_ini)

        ini_r11, ini_r12, ini_r21, ini_r22 = get_R_terms(
            betx=betx_ini,
            bety=bety_ini,
            alfx=alfx_ini,
            alfy=alfy_ini,
            f1001r=f_ini_bbs["f1001r"],
            f1001i=f_ini_bbs["f1001i"],
            f1010r=f_ini_bbs["f1010r"],
            f1010i=f_ini_bbs["f1010i"],
        )

        R_mat = np.array([(ini_r11, ini_r12), (ini_r21, ini_r22)])

        init = xt.twiss.TwissInit(
            element_name=at_ele,
            betx=betx_ini,
            bety=bety_ini,
            alfx=alfx_ini,
            alfy=alfy_ini,
            dx=dx_ini,
            dy=dy_ini,
            dpx=dpx_ini,
            dpy=dpy_ini,
            mux=0.0,
            muy=0.0,
        )
        return init, R_mat

    def phase_diffs(
        self, bpms_x: np.ndarray | None = None, bpms_y: np.ndarray | None = None
    ) -> tuple[xt.TwissTable, xt.TwissTable]:
        if bpms_x is None and bpms_y is None:
            s_bpms = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = s_bpms["bpms_x"]
            bpms_y = s_bpms["bpms_y"]

        mux_diff_front = _get_mu_diff(
            self.tw_sbs,
            self.mes.total_phase_x,
            loc0=bpms_x[:, 0],
            loc=bpms_x,
            plane="x",
        )
        muy_diff_front = _get_mu_diff(
            self.tw_sbs,
            self.mes.total_phase_y,
            loc0=bpms_y[:, 0],
            loc=bpms_y,
            plane="y",
        )

        # mux_diff_back = _get_mu_diff(
        #     tw_back,
        #     self.mes.total_phase_x,
        #     loc0=bpms_x[:, -1],
        #     loc=bpms_x,
        #     plane="x",
        # )
        # muy_diff_back = _get_mu_diff(
        #     tw_back,
        #     self.mes.total_phase_y,
        #     loc0=bpms_y[:, -1],
        #     loc=bpms_y,
        #     plane="y",
        # )

        betx_ini = self.mes.betx.loc[bpms_x[1, 0]].BETX
        bety_ini = self.mes.bety.loc[bpms_y[1, 0]].BETY
        betx_ini_err = self.mes.betx.loc[bpms_x[1, 0]].ERRBETX
        bety_ini_err = self.mes.bety.loc[bpms_y[1, 0]].ERRBETY

        alfx_ini = self.mes.betx.loc[bpms_x[1, 0]].ALFX
        alfy_ini = self.mes.bety.loc[bpms_y[1, 0]].ALFY
        alfx_ini_err = self.mes.betx.loc[bpms_x[1, 0]].ERRALFX
        alfy_ini_err = self.mes.bety.loc[bpms_y[1, 0]].ERRALFY

        betx_end = self.mes.betx.loc[bpms_x[1, -1]].BETX
        bety_end = self.mes.bety.loc[bpms_y[1, -1]].BETY
        betx_end_err = self.mes.betx.loc[bpms_x[1, -1]].ERRBETX
        bety_end_err = self.mes.bety.loc[bpms_y[1, -1]].ERRBETY

        alfx_end = self.mes.betx.loc[bpms_x[1, -1]].ALFX
        alfy_end = self.mes.bety.loc[bpms_y[1, -1]].ALFY
        alfx_end_err = self.mes.betx.loc[bpms_x[1, -1]].ERRALFX
        alfy_end_err = self.mes.bety.loc[bpms_y[1, -1]].ERRALFY

        prop_front_phasex_error = _propagate_error_phase2(
            betx_ini_err,
            alfx_ini_err,
            self.tw_sbs.rows[bpms_x[0]].mux,
            betx_ini,
            alfx_ini,
        )

        prop_front_phasey_error = _propagate_error_phase2(
            bety_ini_err,
            alfy_ini_err,
            self.tw_sbs.rows[bpms_y[0]].muy,
            bety_ini,
            alfy_ini,
        )

        # prop_back_phasex_error = _propagate_error_phase2(
        #     betx_end_err,
        #     alfx_end_err,
        #     tw_back.rows[bpms_x[0]].mux,
        #     betx_end,
        #     alfx_end,
        # )

        # prop_back_phasey_error = _propagate_error_phase2(
        #     bety_end_err,
        #     alfy_end_err,
        #     tw_back.rows[bpms_y[0]].muy,
        #     bety_end,
        #     alfy_end,
        # )

        erphx = self.mes.total_phase_x.loc[bpms_x[1]].ERRPHASEX.values
        erphy = self.mes.total_phase_y.loc[bpms_y[1]].ERRPHASEY.values

        mux_diff_err_front = np.sqrt(prop_front_phasex_error**2 + erphx**2)
        muy_diff_err_front = np.sqrt(prop_front_phasey_error**2 + erphy**2)

        mux_front = (
            self.tw_sbs.rows[bpms_x[0]].mux - self.tw_sbs.rows[bpms_x[0]].mux[0]
        ) % 1
        muy_front = (
            self.tw_sbs.rows[bpms_y[0]].muy - self.tw_sbs.rows[bpms_y[0]].muy[0]
        ) % 1
        # mux_back = (tw_back.rows[bpms_x[0]].mux - tw_back.rows[bpms_x[0]].mux[0]) % 1
        # muy_back = (tw_back.rows[bpms_y[0]].muy - tw_back.rows[bpms_y[0]].muy[0]) % 1

        resx = {}
        resy = {}

        resx["name"] = bpms_x[0]
        resx["s"] = self.tw_sbs.rows[bpms_x[0]].s
        resx["mux"] = self.tw_sbs.rows[bpms_x[0]].mux
        resx["mux2"] = mux_front
        resx["dmux"] = mux_diff_front
        resx["dmux_err"] = mux_diff_err_front

        # resx["mux_back"] = tw_back.rows[bpms_x[0]].mux
        # resx["mux2_back"] = mux_back
        # resx["dmux_back"] = mux_diff_back
        # resx["dmux_back_err"] = mux_diff_err_back

        resy["name"] = bpms_y[0]
        resy["s"] = self.tw_sbs.rows[bpms_y[0]].s
        resy["muy"] = self.tw_sbs.rows[bpms_y[0]].muy
        resy["muy2"] = muy_front
        resy["dmuy"] = muy_diff_front
        resy["dmuy_err"] = muy_diff_err_front

        # resy["muy_back"] = tw_back.rows[bpms_y[0]].muy
        # resy["muy2_back"] = muy_back
        # resy["dmuy_back"] = muy_diff_back
        # resy["dmuy_back_err"] = muy_diff_err_back

        return (xt.TwissTable(resx), xt.TwissTable(resy))
