import xtrack as xt
import numpy as np
from .readers import OptData
from .utils import _get_mu_diff, _propagate_error_phase, _propagate_error_dispersion


class Segment:

    def __init__(
        self, line: xt.Line, start_bpm: str, end_bpm: str, mes: OptData
    ) -> None:
        self.line = line.copy()
        self.start_bpm = start_bpm
        self.end_bpm = end_bpm
        self.mes = mes

        init, _ = self.get_tw_init()
        self.init = init

        # Used to save and restore knobs
        self.original_vals = {}

    def activate_knobs(self, knobs, values) -> None:

        for kn, kv in zip(knobs, values):
            self.original_vals[str(kn)] = self.line.varval[str(kn)]
            self.line.vars[str(kn)] += kv

    def restore_knobs(self, knobs) -> None:
        for kn in knobs:
            self.line.vars[str(kn)] = self.original_vals[str(kn)]

    def get_R_terms(self, betx, bety, alfx, alfy, f1001r, f1001i, f1010r, f1010i):
        ga11 = 1 / np.sqrt(betx)
        ga12 = 0
        ga21 = alfx / np.sqrt(betx)
        ga22 = np.sqrt(betx)
        Ga = np.reshape(np.array([ga11, ga12, ga21, ga22]), (2, 2))

        gb11 = 1 / np.sqrt(bety)
        gb12 = 0
        gb21 = alfy / np.sqrt(bety)
        gb22 = np.sqrt(bety)
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

    def get_tw_init(self) -> tuple[xt.TwissInit, np.ndarray]:
        """
        Return the twiss init at 'start_bpm'
        and the coupling matrix, derived from the measurement
        """
        start_BPM = self.start_bpm.upper()
        phix_ini = 0.0
        phiy_ini = 0.0
        dpx_ini = 0
        dpy_ini = 0
        dx_ini = 0
        dy_ini = 0
        wx_ini = 0.0
        wy_ini = 0.0

        betx_ini = self.mes.betx_free.BETX.loc[start_BPM]
        bety_ini = self.mes.bety_free.BETY.loc[start_BPM]
        alfx_ini = self.mes.betx_free.ALFX.loc[start_BPM]
        alfy_ini = self.mes.bety_free.ALFY.loc[start_BPM]

        f_ini_bbs = {}
        f_ini_bbs["f1001r"] = self.mes.coupling.F1001R.loc[start_BPM]
        f_ini_bbs["f1001i"] = self.mes.coupling.F1001I.loc[start_BPM]
        f_ini_bbs["f1010r"] = self.mes.coupling.F1010R.loc[start_BPM]
        f_ini_bbs["f1010i"] = self.mes.coupling.F1010I.loc[start_BPM]
        try:
            dx_ini = self.mes.data_dx.DX.loc[start_BPM]
            dy_ini = self.mes.data_dy.DY.loc[start_BPM]
            dpx_ini = self.mes.data_dx.DPX.loc[start_BPM]
            dpy_ini = self.mes.data_dy.DPY.loc[start_BPM]
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

    def twiss_sbs(self):
        """
        Get twiss for the segment
        """
        if self.init is None:
            self.init, Rmat = self.get_tw_init()
        sbs_tw = self.line.twiss(start=self.start_bpm, end=self.end_bpm, init=self.init)

        return sbs_tw

    def get_common_bpms(self, attr="total_phase_"):
        tw_sbs = self.twiss_sbs()
        tw_sbs_bpms = tw_sbs.rows["bpm.*"]
        bpms_names = np.array([[b, b.upper()] for b in tw_sbs_bpms.name]).T

        common_bpm_mes = getattr(self.mes, f"{attr}x").index.intersection(
            getattr(self.mes, f"{attr}y").index
        )
        common_bpms = bpms_names[1][np.isin(bpms_names[1], common_bpm_mes)]

        bpms_names = np.array([[b.lower(), b] for b in common_bpms]).T
        return bpms_names

    def get_s_and_bpms(self, attr="total_phase_"):
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
        self, bpms_x=None, bpms_y=None, fmt="bb"
    ) -> tuple[xt.TwissTable, xt.TwissTable]:
        resx = {}
        resy = {}

        tw = self.twiss_sbs()

        if bpms_x is None and bpms_y is None:
            s_bpms = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = s_bpms["bpms_x"]
            bpms_y = s_bpms["bpms_y"]

        mux_diff = _get_mu_diff(
            tw,
            self.mes.total_phase_x,
            loc0=bpms_x[:, 0],
            loc=bpms_x,
            plane="x",
        )
        muy_diff = _get_mu_diff(
            tw,
            self.mes.total_phase_y,
            loc0=bpms_y[:, 0],
            loc=bpms_y,
            plane="y",
        )

        betx_ini = self.mes.betx.loc[bpms_x[1]].BETX.iloc[0]
        bety_ini = self.mes.bety.loc[bpms_y[1]].BETY.iloc[0]
        betx_ini_err = self.mes.betx.loc[bpms_x[1]].ERRBETX.iloc[0]
        bety_ini_err = self.mes.bety.loc[bpms_y[1]].ERRBETY.iloc[0]

        alfx_ini = self.mes.betx.loc[bpms_x[1]].ALFX.iloc[0]
        alfy_ini = self.mes.bety.loc[bpms_y[1]].ALFY.iloc[0]
        alfx_ini_err = self.mes.betx.loc[bpms_x[1]].ERRALFX.iloc[0]
        alfy_ini_err = self.mes.bety.loc[bpms_y[1]].ERRALFY.iloc[0]

        prop_phasex_error = _propagate_error_phase(
            betx_ini_err,
            alfx_ini_err,
            tw.rows[bpms_x[0]].mux,
            betx_ini,
            alfx_ini,
        )

        prop_phasey_error = _propagate_error_phase(
            bety_ini_err,
            alfy_ini_err,
            tw.rows[bpms_y[0]].muy,
            bety_ini,
            alfy_ini,
        )

        mux = (tw.rows[bpms_x[0]].mux - tw.rows[bpms_x[0]].mux[0]) % 1
        muy = (tw.rows[bpms_y[0]].muy - tw.rows[bpms_y[0]].muy[0]) % 1

        if fmt == "bb":
            mux_diff_err = np.sqrt(
                prop_phasex_error**2
                + self.mes.total_phase_x.loc[bpms_x[1]].STDPHX.values ** 2
            )

            muy_diff_err = np.sqrt(
                prop_phasey_error**2
                + self.mes.total_phase_y.loc[bpms_y[1]].STDPHY.values ** 2
            )
        elif fmt == "omc":
            mux_diff_err = np.sqrt(
                prop_phasex_error**2
                + self.mes.total_phase_x.loc[bpms_x[1]].ERRPHASEX.values ** 2
            )

            muy_diff_err = np.sqrt(
                prop_phasey_error**2
                + self.mes.total_phase_y.loc[bpms_y[1]].ERRPHASEY.values ** 2
            )
        else:
            print(f"Wrong {fmt=}. Choose bb or omc")

        resx["name"] = bpms_x[0]
        resx["s"] = tw.rows[bpms_x[0]].s
        resx["mux"] = tw.rows[bpms_x[0]].mux
        resx["mux2"] = mux
        resx["dmux"] = mux_diff
        resx["dmux_err"] = mux_diff_err

        resy["name"] = bpms_y[0]
        resy["s"] = tw.rows[bpms_y[0]].s
        resy["muy"] = tw.rows[bpms_y[0]].muy
        resy["muy2"] = muy
        resy["dmuy"] = muy_diff
        resy["dmuy_err"] = muy_diff_err

        return (xt.TwissTable(resx), xt.TwissTable(resy))

    def get_disp_diffs(
        self, bpms_x=None, bpms_y=None
    ) -> tuple[xt.TwissTable, xt.TwissTable]:
        resx = {}
        resy = {}

        tw = self.twiss_sbs()

        if bpms_x is None and bpms_y is None:
            s_bpms = self.get_s_and_bpms(attr="data_d")
            bpms_x = s_bpms["bpms_x"]
            bpms_y = s_bpms["bpms_y"]

        normal_prop_dx_err = _propagate_error_dispersion(
            self.mes.data_dx.loc[bpms_x[1][0]].STDDX,
            tw.rows[bpms_x[0]].betx[0],
            tw.rows[bpms_x[0]].betx,
            (tw.rows[bpms_x[0]].mux % 1),
            tw.rows[bpms_x[0]].alfx[0],
        )

        normal_prop_dy_err = _propagate_error_dispersion(
            self.mes.data_dy.loc[bpms_y[1][0]].STDDY,
            tw.rows[bpms_y[0]].bety[0],
            tw.rows[bpms_y[0]].bety,
            (tw.rows[bpms_y[0]].muy % 1),
            tw.rows[bpms_y[0]].alfy[0],
        )

        resx["name"] = tw.rows[bpms_x[0]].name
        resx["s"] = tw.rows[bpms_x[0]].s
        resx["dx_diff"] = (
            self.mes.data_dx.loc[bpms_x[1]].DX.values - tw.rows[bpms_x[0]].dx
        )
        resx["dx_diff_err"] = np.sqrt(
            self.mes.data_dx.loc[bpms_x[1]].STDDX.values ** 2 + normal_prop_dx_err**2
        )
        resy["name"] = tw.rows[bpms_y[0]].name
        resy["s"] = tw.rows[bpms_y[0]].s
        resy["dy_diff"] = (
            self.mes.data_dy.loc[bpms_y[1]].DY.values - tw.rows[bpms_y[0]].dy
        )
        resy["dy_diff_err"] = np.sqrt(
            self.mes.data_dy.loc[bpms_y[1]].STDDY.values ** 2 + normal_prop_dy_err**2
        )

        return (xt.TwissTable(resx), xt.TwissTable(resy))

    def plot_phase_diff(self, bpms_x=None, bpms_y=None, tw_cor=None):
        import matplotlib.pyplot as plt

        if bpms_x is None and bpms_y is None:
            bpms_names = self.get_s_and_bpms(attr="total_phase_")
            bpms_x = bpms_names["bpms_x"]
            bpms_y = bpms_names["bpms_y"]

        bpms = {"x": bpms_x, "y": bpms_y}

        bpms_common = np.intersect1d(bpms_x[0], bpms_y[0])
        tw_sbs = self.twiss_sbs()
        tw_phase = self.get_phase_diffs(bpms_x=bpms_x, bpms_y=bpms_y, fmt="bb")

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
