import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
from .readers import OptData
from .utils import (
    _get_mu_diff,
    _propagate_error_phase,
    _propagate_error_phase2,
    _propagate_error_dispersion,
)

from typing import Sequence


class Segment:

    def __init__(
        self,
        line: xt.Line,
        start_bpm: str,
        end_bpm: str,
        mes: OptData,
        fmt: str = "bbsrc",
    ) -> None:
        self.line = line.copy()
        self.start_bpm = start_bpm
        self.end_bpm = end_bpm
        self.mes = mes
        self.fmt = fmt

        init_start, _ = self.get_tw_init(at_ele=self.start_bpm)
        init_end, _ = self.get_tw_init(at_ele=self.end_bpm)
        self.init_start = init_start
        self.init_end = init_end

        # Used to save and restore knobs
        self.original_vals = {}

    def activate_knobs(self, knobs: Sequence[str], values: Sequence[float]) -> None:

        for kn, kv in zip(knobs, values):
            self.original_vals[str(kn)] = self.line.varval[str(kn)]
            self.line.vars[str(kn)] += kv

    def restore_knobs(self, knobs: Sequence[str]) -> None:
        for kn in knobs:
            self.line.vars[str(kn)] = self.original_vals[str(kn)]

    def get_R_terms(self, betx, bety, alfx, alfy, f1001r, f1001i, f1010r, f1010i):
        sqrbx = np.sqrt(betx)
        sqrby = np.sqrt(bety)

        ga11 = 1 / sqrbx
        ga12 = 0
        ga21 = alfx / sqrbx
        ga22 = sqrbx
        Ga = np.reshape(np.array([ga11, ga12, ga21, ga22]), (2, 2))

        gb11 = 1 / sqrby
        gb12 = 0
        gb21 = alfy / sqrby
        gb22 = sqrby
        Gb = np.reshape(np.array([gb11, gb12, gb21, gb22]), (2, 2))

        J = np.reshape(np.array([0, 1, -1, 0]), (2, 2))

        absf1001 = np.sqrt(f1001r**2 + f1001i**2)
        absf1010 = np.sqrt(f1010r**2 + f1010i**2)

        gamma2 = 1.0 / (1.0 + 4.0 * (absf1001**2 - absf1010**2))
        c11 = f1001i + f1010i
        c22 = f1001i - f1010i
        c12 = -(f1010r - f1001r)
        c21 = -(f1010r + f1001r)
        Cbar = np.reshape(2 * np.sqrt(gamma2) * np.array([c11, c12, c21, c22]), (2, 2))

        C = np.dot(np.linalg.inv(Ga), np.dot(Cbar, Gb))
        jCj = np.dot(J, np.dot(C, -J))
        c = np.linalg.det(C)
        r = -c / (c - 1)
        R = np.transpose(np.sqrt(1 + r) * jCj)
        return np.ravel(R)

    def get_tw_init(self, at_ele: str) -> tuple[xt.TwissInit, np.ndarray]:
        """
        Return the twiss init at 'at_ele' element
        and the coupling matrix, derived from the measurement
        """
        BPM = at_ele.upper()
        phix_ini = 0.0
        phiy_ini = 0.0
        dpx_ini = 0
        dpy_ini = 0
        dx_ini = 0
        dy_ini = 0
        wx_ini = 0.0
        wy_ini = 0.0

        f_ini_bbs = {}

        if self.fmt == "bbsrc":
            betx_ini = self.mes.betx_free.BETX.loc[BPM]
            bety_ini = self.mes.bety_free.BETY.loc[BPM]
            alfx_ini = self.mes.betx_free.ALFX.loc[BPM]
            alfy_ini = self.mes.bety_free.ALFY.loc[BPM]

            f_ini_bbs["f1001r"] = self.mes.coupling.F1001R.loc[BPM]
            f_ini_bbs["f1001i"] = self.mes.coupling.F1001I.loc[BPM]
            f_ini_bbs["f1010r"] = self.mes.coupling.F1010R.loc[BPM]
            f_ini_bbs["f1010i"] = self.mes.coupling.F1010I.loc[BPM]

        elif self.fmt == "omc3":
            betx_ini = self.mes.betx.BETX.loc[BPM]
            bety_ini = self.mes.bety.BETY.loc[BPM]
            alfx_ini = self.mes.betx.ALFX.loc[BPM]
            alfy_ini = self.mes.bety.ALFY.loc[BPM]

            f_ini_bbs["f1001r"] = self.mes.f1001.REAL.loc[BPM]
            f_ini_bbs["f1001i"] = self.mes.f1001.IMAG.loc[BPM]
            f_ini_bbs["f1010r"] = self.mes.f1010.REAL.loc[BPM]
            f_ini_bbs["f1010i"] = self.mes.f1010.IMAG.loc[BPM]

        try:
            dx_ini = self.mes.data_dx.DX.loc[BPM]
            dy_ini = self.mes.data_dy.DY.loc[BPM]
            dpx_ini = self.mes.data_dx.DPX.loc[BPM]
            dpy_ini = self.mes.data_dy.DPY.loc[BPM]
        except:
            print("No dispersion measurements available")

        ini_r11, ini_r12, ini_r21, ini_r22 = self.get_R_terms(
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
            mux=phix_ini,
            muy=phiy_ini,
        )
        return init, R_mat

    def twiss(self, mode: str = "front") -> xt.TwissTable | None:
        if mode == "front":
            return self.twiss_sbs()
        elif mode == "back":
            return self.backtwiss_sbs()
        else:
            print("Available modes are 'front' and 'back'")

    def twiss_sbs(self) -> xt.TwissTable:
        """
        Get twiss for the segment
        """
        if self.init_start is None:
            self.init_start, Rmat = self.get_tw_init()
        sbs_tw = self.line.twiss(
            start=self.start_bpm, end=self.end_bpm, init=self.init_start
        )

        return sbs_tw

    def backtwiss_sbs(self) -> xt.TwissTable:
        """
        Get twiss for the segment
        """
        if self.init_end is None:
            self.init_end, Rmat = self.get_tw_init(at_ele=self.end_bpm)
        sbs_tw = self.line.twiss(
            start=self.start_bpm, end=self.end_bpm, init=self.init_end
        )

        return sbs_tw

    def get_common_bpms(self, attr: str = "total_phase_"):
        tw_sbs = self.twiss_sbs()
        tw_sbs_bpms = tw_sbs.rows["bpm.*"]
        bpms_names = np.array([[b, b.upper()] for b in tw_sbs_bpms.name]).T

        common_bpm_mes = getattr(self.mes, f"{attr}x").index.intersection(
            getattr(self.mes, f"{attr}y").index
        )
        common_bpms = bpms_names[1][np.isin(bpms_names[1], common_bpm_mes)]

        bpms_names = np.array([[b.lower(), b] for b in common_bpms]).T
        return bpms_names

    def get_s_and_bpms(self, attr: str = "total_phase_"):
        tw_sbs = self.twiss_sbs()
        tw_sbs_bpms = tw_sbs.rows["bpm.*"]
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

        # mes_attrx = get_mes_phase(
        #     getattr(mes, f"{attr}x"), loc0=bpms_x[:, 0], loc=bpms_x, plane="x"
        # )
        # mes_attry = get_mes_phase(
        #     getattr(mes, f"{attr}y"), loc0=bpms_y[:, 0], loc=bpms_y, plane="y"
        # )

        res = {
            "bpms_x": bpms_x,
            "bpms_y": bpms_y,
            "mes_sx": mes_sx,
            "mes_sy": mes_sy,
            # f"{attr}_x": mes_attrx,
            # f"{attr}_y": mes_attry,
        }
        return res

    def get_phase_diffs(
        self, bpms_x=None, bpms_y=None
    ) -> tuple[tuple[xt.TwissTable, xt.TwissTable]]:
        tw_front = self.twiss_sbs()
        tw_back = self.backtwiss_sbs()

        if bpms_x is None and bpms_y is None:
            s_bpms = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = s_bpms["bpms_x"]
            bpms_y = s_bpms["bpms_y"]

        mux_diff_front = _get_mu_diff(
            tw_front,
            self.mes.total_phase_x,
            loc0=bpms_x[:, 0],
            loc=bpms_x,
            plane="x",
        )
        muy_diff_front = _get_mu_diff(
            tw_front,
            self.mes.total_phase_y,
            loc0=bpms_y[:, 0],
            loc=bpms_y,
            plane="y",
        )

        mux_diff_back = _get_mu_diff(
            tw_back,
            self.mes.total_phase_x,
            loc0=bpms_x[:, -1],
            loc=bpms_x,
            plane="x",
        )
        muy_diff_back = _get_mu_diff(
            tw_back,
            self.mes.total_phase_y,
            loc0=bpms_y[:, -1],
            loc=bpms_y,
            plane="y",
        )

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
            tw_front.rows[bpms_x[0]].mux,
            betx_ini,
            alfx_ini,
        )

        prop_front_phasey_error = _propagate_error_phase2(
            bety_ini_err,
            alfy_ini_err,
            tw_front.rows[bpms_y[0]].muy,
            bety_ini,
            alfy_ini,
        )

        prop_back_phasex_error = _propagate_error_phase2(
            betx_end_err,
            alfx_end_err,
            tw_back.rows[bpms_x[0]].mux,
            betx_end,
            alfx_end,
        )

        prop_back_phasey_error = _propagate_error_phase2(
            bety_end_err,
            alfy_end_err,
            tw_back.rows[bpms_y[0]].muy,
            bety_end,
            alfy_end,
        )

        mux_front = (tw_front.rows[bpms_x[0]].mux - tw_front.rows[bpms_x[0]].mux[0]) % 1
        muy_front = (tw_front.rows[bpms_y[0]].muy - tw_front.rows[bpms_y[0]].muy[0]) % 1
        mux_back = (tw_back.rows[bpms_x[0]].mux - tw_back.rows[bpms_x[0]].mux[0]) % 1
        muy_back = (tw_back.rows[bpms_y[0]].muy - tw_back.rows[bpms_y[0]].muy[0]) % 1

        if self.fmt == "bbsrc":
            erphx = self.mes.total_phase_x.loc[bpms_x[1]].STDPHX.values
            erphy = self.mes.total_phase_y.loc[bpms_y[1]].STDPHY.values
        elif self.fmt == "omc3":
            erphx = self.mes.total_phase_x.loc[bpms_x[1]].ERRPHASEX.values
            erphy = self.mes.total_phase_y.loc[bpms_y[1]].ERRPHASEY.values
        else:
            print(f"Wrong {self.fmt=}. Choose bbsrc or omc3")

        mux_diff_err_front = np.sqrt(prop_front_phasex_error**2 + erphx**2)
        muy_diff_err_front = np.sqrt(prop_front_phasey_error**2 + erphy**2)

        # TODO: check if errors are calculated correctly
        mux_diff_err_back = np.sqrt(prop_back_phasex_error**2 + erphx**2)
        muy_diff_err_back = np.sqrt(prop_back_phasey_error**2 + erphy**2)

        resx = {}
        resy = {}

        resx["name"] = bpms_x[0]
        resx["s"] = tw_front.rows[bpms_x[0]].s
        resx["mux"] = tw_front.rows[bpms_x[0]].mux
        resx["mux2"] = mux_front
        resx["dmux"] = mux_diff_front
        resx["dmux_err"] = mux_diff_err_front

        resx["mux_back"] = tw_back.rows[bpms_x[0]].mux
        resx["mux2_back"] = mux_back
        resx["dmux_back"] = mux_diff_back
        resx["dmux_back_err"] = mux_diff_err_back

        resy["name"] = bpms_y[0]
        resy["s"] = tw_front.rows[bpms_y[0]].s
        resy["muy"] = tw_front.rows[bpms_y[0]].muy
        resy["muy2"] = muy_front
        resy["dmuy"] = muy_diff_front
        resy["dmuy_err"] = muy_diff_err_front

        resy["muy_back"] = tw_back.rows[bpms_y[0]].muy
        resy["muy2_back"] = muy_back
        resy["dmuy_back"] = muy_diff_back
        resy["dmuy_back_err"] = muy_diff_err_back

        return (xt.TwissTable(resx), xt.TwissTable(resy))

    def get_disp_diffs(
        self, bpms_x=None, bpms_y=None
    ) -> tuple[xt.TwissTable, xt.TwissTable]:
        resx = {}
        resy = {}

        tw_front = self.twiss_sbs()
        tw_back = self.backtwiss_sbs()

        if bpms_x is None and bpms_y is None:
            s_bpms = self.get_s_and_bpms(attr="data_d")
            bpms_x = s_bpms["bpms_x"]
            bpms_y = s_bpms["bpms_y"]

        if self.fmt == "bbsrc":
            erdx = self.mes.data_dx.loc[bpms_x[1, 0]].STDDX
            erdy = self.mes.data_dy.loc[bpms_y[1, 0]].STDDY
        elif self.fmt == "omc3":
            erdx = self.mes.data_dx.loc[bpms_x[1, 0]].ERRDX
            erdy = self.mes.data_dy.loc[bpms_y[1, 0]].ERRDY

        normal_prop_dx_err = _propagate_error_dispersion(
            erdx,
            tw_front.rows[bpms_x[0]].betx[0],
            tw_front.rows[bpms_x[0]].betx,
            (tw_front.rows[bpms_x[0]].mux % 1),
            tw_front.rows[bpms_x[0]].alfx[0],
        )

        normal_prop_dy_err = _propagate_error_dispersion(
            erdy,
            tw_front.rows[bpms_y[0]].bety[0],
            tw_front.rows[bpms_y[0]].bety,
            (tw_front.rows[bpms_y[0]].muy % 1),
            tw_front.rows[bpms_y[0]].alfy[0],
        )

        resx["name"] = tw_front.rows[bpms_x[0]].name
        resx["s"] = tw_front.rows[bpms_x[0]].s
        resx["dx"] = tw_front.rows[bpms_x[0]].dx
        resx["dx_diff"] = (
            self.mes.data_dx.loc[bpms_x[1]].DX.values - tw_front.rows[bpms_x[0]].dx
        )
        resx["dx_diff_err"] = np.sqrt(erdx**2 + normal_prop_dx_err**2)
        resx["dx_back"] = tw_back.rows[bpms_x[0]].dx
        resx["dx_diff_back"] = (
            self.mes.data_dx.loc[bpms_x[1]].DX.values - tw_back.rows[bpms_x[0]].dx
        )

        resy["name"] = tw_front.rows[bpms_y[0]].name
        resy["s"] = tw_front.rows[bpms_y[0]].s
        resy["dy"] = tw_front.rows[bpms_y[0]].dy
        resy["dy_diff"] = (
            self.mes.data_dy.loc[bpms_y[1]].DY.values - tw_front.rows[bpms_y[0]].dy
        )
        resy["dy_diff_err"] = np.sqrt(erdy**2 + normal_prop_dy_err**2)
        resy["dy_back"] = tw_back.rows[bpms_y[0]].dy
        resy["dy_diff_back"] = (
            self.mes.data_dy.loc[bpms_y[1]].DY.values - tw_back.rows[bpms_y[0]].dy
        )

        return (xt.TwissTable(resx), xt.TwissTable(resy))

    def plot_phase_diff(self, bpms_x=None, bpms_y=None, tw_cor=None):

        if bpms_x is None and bpms_y is None:
            bpms_names = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = bpms_names["bpms_x"]
            bpms_y = bpms_names["bpms_y"]

        bpms = {"x": bpms_x, "y": bpms_y}

        bpms_common = np.intersect1d(bpms_x[0], bpms_y[0])
        tw_sbs = self.twiss_sbs()
        tw_phase, tw_phase_back = self.get_phase_diffs(bpms_x=bpms_x, bpms_y=bpms_y)

        fig, axs = plt.subplots(
            3, 1, figsize=(11, 11), sharex=True, height_ratios=[0.5, 1, 1], dpi=300
        )
        axs[0].plot(
            tw_sbs.rows[bpms_x[0]].s,
            tw_sbs.rows[bpms_x[0]].x * 1e3,
            marker="o",
            ls="-",
            ms=4,
            label="x",
            color="black",
        )
        if isinstance(tw_cor, xt.TwissTable):
            axs[0].plot(
                tw_cor.rows[bpms_x[0]].s,
                tw_cor.rows[bpms_x[0]].x * 1e3,
                marker="o",
                ls="-",
                ms=4,
                label="x",
                color="red",
            )
        elif isinstance(tw_cor, dict):
            for nlabel, twc in tw_cor.items():
                axs[0].plot(
                    twc.rows[bpms_x[0]].s,
                    twc.rows[bpms_x[0]].x * 1e3,
                    marker="o",
                    ls="-",
                    ms=4,
                    label=f"{nlabel}, x",
                )

        axs[0].set_ylabel("co [mm]")

        axs_t = axs[0].twiny()
        axs_t.set_xticks(
            tw_sbs.rows[bpms_common].s,
            tw_sbs.rows[bpms_common].name,
            rotation="vertical",
        )

        axs_t.set_xlim(axs[0].get_xlim()[0], axs[0].get_xlim()[1])

        fig.subplots_adjust(hspace=0)
        for i, PLANE in enumerate(["x", "y"]):
            axs[i + 1].errorbar(
                tw_phase[i].s,
                getattr(tw_phase[i], f"dmu{PLANE}"),
                yerr=getattr(tw_phase[i], f"dmu{PLANE}_err"),
                marker="o",
                ls="-",
                label="Measurement",
                color="black",
            )
            if isinstance(tw_cor, xt.TwissTable):
                axs[i + 1].errorbar(
                    tw_cor.rows[bpms[PLANE][0]].s,
                    -getattr(tw_sbs.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}")
                    + getattr(tw_cor.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}"),
                    marker="o",
                    ls="-",
                    label="Arc Correction",
                    color="red",
                )
            elif isinstance(tw_cor, dict):
                for nlabel, twc in tw_cor.items():
                    axs[i + 1].errorbar(
                        twc.rows[bpms[PLANE][0]].s,
                        -getattr(tw_sbs.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}")
                        + getattr(twc.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}"),
                        marker="o",
                        ls="-",
                        label=nlabel,
                    )

            axs[i + 1].set_ylabel(rf"$\Delta\phi_{PLANE}\ [2\pi]$")

        for i in range(0, 3):
            axs[i].grid()
            axs[i].legend()
            axs[i].set_xlabel(r"$s [m]$")
        plt.show()

    def plot_phase_back_diff(self, bpms_x=None, bpms_y=None, tw_cor=None):

        if bpms_x is None and bpms_y is None:
            bpms_names = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = bpms_names["bpms_x"]
            bpms_y = bpms_names["bpms_y"]

        bpms = {"x": bpms_x, "y": bpms_y}

        bpms_common = np.intersect1d(bpms_x[0], bpms_y[0])
        tw_sbs = self.backtwiss_sbs()
        _, tw_phase = self.get_phase_diffs(bpms_x=bpms_x, bpms_y=bpms_y)

        fig, axs = plt.subplots(
            3, 1, figsize=(11, 11), sharex=True, height_ratios=[0.5, 1, 1], dpi=300
        )
        axs[0].plot(
            tw_sbs.rows[bpms_x[0]].s,
            tw_sbs.rows[bpms_x[0]].x * 1e3,
            marker="o",
            ls="-",
            ms=4,
            label="x",
            color="black",
        )
        if isinstance(tw_cor, xt.TwissTable):
            axs[0].plot(
                tw_cor.rows[bpms_x[0]].s,
                tw_cor.rows[bpms_x[0]].x * 1e3,
                marker="o",
                ls="-",
                ms=4,
                label="x",
                color="red",
            )
        elif isinstance(tw_cor, dict):
            for nlabel, twc in tw_cor.items():
                axs[0].plot(
                    twc.rows[bpms_x[0]].s,
                    twc.rows[bpms_x[0]].x * 1e3,
                    marker="o",
                    ls="-",
                    ms=4,
                    label=f"{nlabel}, x",
                )

        axs[0].set_ylabel("co [mm]")

        axs_t = axs[0].twiny()
        axs_t.set_xticks(
            tw_sbs.rows[bpms_common].s,
            tw_sbs.rows[bpms_common].name,
            rotation="vertical",
        )

        axs_t.set_xlim(axs[0].get_xlim()[0], axs[0].get_xlim()[1])

        fig.subplots_adjust(hspace=0)
        for i, PLANE in enumerate(["x", "y"]):
            axs[i + 1].errorbar(
                tw_phase[i].s,
                getattr(tw_phase[i], f"dmu{PLANE}"),
                yerr=getattr(tw_phase[i], f"dmu{PLANE}_err"),
                marker="o",
                ls="-",
                label="Measurement",
                color="black",
            )
            if isinstance(tw_cor, xt.TwissTable):
                axs[i + 1].errorbar(
                    tw_cor.rows[bpms[PLANE][0]].s,
                    -getattr(tw_sbs.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}")
                    + getattr(tw_cor.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}"),
                    marker="o",
                    ls="-",
                    label="Arc Correction",
                    color="red",
                )
            elif isinstance(tw_cor, dict):
                for nlabel, twc in tw_cor.items():
                    axs[i + 1].errorbar(
                        twc.rows[bpms[PLANE][0]].s,
                        -getattr(tw_sbs.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}")
                        + getattr(twc.rows[bpms[PLANE][0]], f"mu{PLANE.lower()}"),
                        marker="o",
                        ls="-",
                        label=nlabel,
                    )

            axs[i + 1].set_ylabel(rf"$\Delta\phi_{PLANE}\ [2\pi]$")

        for i in range(0, 3):
            axs[i].grid()
            axs[i].legend()
            axs[i].set_xlabel(r"$s [m]$")
        plt.show()

    def plot_phase_diff_fb(self, bpms_x=None, bpms_y=None):

        if bpms_x is None and bpms_y is None:
            bpms_names = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = bpms_names["bpms_x"]
            bpms_y = bpms_names["bpms_y"]

        bpms_common = np.intersect1d(bpms_x[0], bpms_y[0])
        tw_sbs = self.twiss_sbs()
        tw_phase, tw_phase_back = self.get_phase_diffs(bpms_x=bpms_x, bpms_y=bpms_y)

        fig, axs = plt.subplots(2, 1, figsize=(14, 10.5), sharex=True, dpi=300)

        fig.subplots_adjust(hspace=0)
        for i, PLANE in enumerate(["x", "y"]):
            axs[i].errorbar(
                tw_phase[i].s,
                getattr(tw_phase[i], f"dmu{PLANE}"),
                yerr=getattr(tw_phase[i], f"dmu{PLANE}_err"),
                marker="o",
                ls="-",
                label="Measurement, Front Prop.",
                color="black",
            )

            axs[i].errorbar(
                tw_phase_back[i].s,
                getattr(tw_phase_back[i], f"dmu{PLANE}"),
                yerr=getattr(tw_phase_back[i], f"dmu{PLANE}_err"),
                marker="o",
                ls="-",
                label="Measurement, Back Prop.",
                color="red",
            )

            axs[i].set_ylabel(rf"$\Delta\phi_{PLANE}\ [2\pi]$")

        axs_t = axs[0].twiny()
        axs_t.set_xticks(
            tw_sbs.rows[bpms_common].s,
            tw_sbs.rows[bpms_common].name,
            rotation="vertical",
        )

        axs_t.set_xlim(axs[0].get_xlim()[0], axs[0].get_xlim()[1])

        for i in range(0, 2):
            axs[i].grid()
            axs[i].legend()
            axs[i].set_xlabel(r"$s [m]$")
        plt.show()
