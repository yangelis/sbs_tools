import tfs


class OptData:
    def __init__(self, foldername, fmt="omc3") -> None:
        self.name = foldername.split("/")[-1]
        self.has_dispersion = False
        if fmt == "omc3":
            self.x = tfs.read_tfs(f"{foldername}/orbit_x.tfs", index="NAME")
            self.y = tfs.read_tfs(f"{foldername}/orbit_y.tfs", index="NAME")

            self.betx = tfs.read_tfs(f"{foldername}/beta_phase_x.tfs", index="NAME")
            self.bety = tfs.read_tfs(f"{foldername}/beta_phase_y.tfs", index="NAME")

            self.total_phase_x = tfs.read_tfs(
                f"{foldername}/total_phase_x.tfs", index="NAME"
            )
            self.total_phase_y = tfs.read_tfs(
                f"{foldername}/total_phase_y.tfs", index="NAME"
            )

            self.phase_x = tfs.read_tfs(f"{foldername}/phase_x.tfs", index="NAME")
            self.phase_y = tfs.read_tfs(f"{foldername}/phase_y.tfs", index="NAME")

            self.f1001 = tfs.read_tfs(f"{foldername}/f1001.tfs", index="NAME")
            self.f1010 = tfs.read_tfs(f"{foldername}/f1010.tfs", index="NAME")

            try:
                self.data_dx = tfs.read_tfs(
                    f"{foldername}/dispersion_x.tfs", index="NAME"
                )
                self.data_dy = tfs.read_tfs(
                    f"{foldername}/dispersion_y.tfs", index="NAME"
                )
                self.has_dispersion = True
            except FileNotFoundError:
                print("No dispersion files found")

            # self.betx_amp = tfs.read_tfs(
            #     f"{foldername}/beta_amplitude_x.tfs", index="NAME"
            # )
            # self.bety_amp = tfs.read_tfs(
            #     f"{foldername}/beta_amplitude_y.tfs", index="NAME"
            # )

        else:
            self.x = tfs.read_tfs(f"{foldername}/getCOx.out", index="NAME")
            self.y = tfs.read_tfs(f"{foldername}/getCOy.out", index="NAME")
            self.betx = tfs.read_tfs(f"{foldername}/getbetax.out", index="NAME")
            self.bety = tfs.read_tfs(f"{foldername}/getbetay.out", index="NAME")

            self.betx_free = tfs.read_tfs(
                f"{foldername}/getbetax_free.out", index="NAME"
            )
            self.bety_free = tfs.read_tfs(
                f"{foldername}/getbetay_free.out", index="NAME"
            )

            # , "getphasex.out"
            self.phase_x = tfs.read_tfs(
                f"{foldername}/getphasex_free.out", index="NAME"
            )
            # , "getphasey.out"
            self.phase_y = tfs.read_tfs(
                f"{foldername}/getphasey_free.out", index="NAME"
            )

            # "getphasetotx.out"
            self.total_phase_x = tfs.read_tfs(
                f"{foldername}/getphasetotx_free.out", index="NAME"
            )
            # "getphasetoty.out"
            self.total_phase_y = tfs.read_tfs(
                f"{foldername}/getphasetoty_free.out", index="NAME"
            )

            self.coupling = tfs.read_tfs(
                f"{foldername}/getcouple_free.out", index="NAME"
            )

            try:
                self.data_dx = tfs.read_tfs(f"{foldername}/getDx.out", index="NAME")
                self.data_dy = tfs.read_tfs(f"{foldername}/getDy.out", index="NAME")
                self.data_ndx = tfs.read_tfs(f"{foldername}/getNDx.out", index="NAME")
                self.has_dispersion = True
            except FileNotFoundError:
                print("No dispersion files found")

            try:
                self.data_ndy = tfs.read_tfs(f"{foldername}/getNDy.out", index="NAME")
                self.has_normdy = True
            except FileNotFoundError:
                self.has_normdy = False
                # print("No NDy found")

            self.data_f1001 = tfs.read_tfs(f"{foldername}/getcouple.out", index="NAME")
