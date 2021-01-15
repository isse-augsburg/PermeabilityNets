from collections import OrderedDict
import socket
import torch
from Models.sensor_to_fiberfraction_model import FFTFF
import io
import logging
import re
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import Resources.training as tr_resources
from Pipeline.resampling import get_fixed_number_of_indices
from Utils.data_utils import extract_coords_of_mesh_nodes, load_mean_std
# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from Utils.img_utils import create_np_image


# This class provides all original functions but tries to improve the performance of consecutive calls
class DataloaderImages:
    def __init__(
        self,
        image_size=(135, 103),
        ignore_useless_states=True,
        sensor_indizes=((0, 1), (0, 1)),
        skip_indizes=(0, None, 1),
        divide_by_100k=True,
    ):
        self.image_size = image_size
        self.coords = None
        self.ff_coords = None
        self.fftriang = None
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        self.divide_by_100k = divide_by_100k
        self.mean = None
        self.std = None
        if not self.divide_by_100k:
            self.mean, self.std = load_mean_std(
                tr_resources.mean_std_1140_pressure_sensors
            )

    def _get_flowfront(self, f: h5py.File, meta_f: h5py.File, states=None):
        """
        Load the flow front for the given states or all available states if states is None
        """
        useless_states = None
        try:
            coords = self._get_coords(f)
            if not states:
                states = f["post"]["singlestate"]
            states = list(states)[
                self.skip_indizes[0]: self.skip_indizes[1]: self.skip_indizes[2]
            ]
            if meta_f is not None:
                useless_states = meta_f["useless_states/singlestates"][()]
                if len(useless_states) == 0:
                    useless_states = None
            filling_factors_at_certain_times = []
            for state in states:
                if (
                    useless_states is not None
                    and state == f"state{useless_states[0]:012d}"
                ):
                    break
                else:
                    filling_factors_at_certain_times.append(
                        f["post"]["singlestate"][state]["entityresults"]["NODE"][
                            "FILLING_FACTOR"
                        ]["ZONE1_set1"]["erfblock"]["res"][()]
                    )

            flat_fillings = np.squeeze(filling_factors_at_certain_times)
            return (
                create_np_image(
                    target_shape=self.image_size, norm_coords=coords, data=filling
                )
                for filling in flat_fillings
            )
        except KeyError:
            return None

    def _get_fiber_fraction(self, f):
        if self.ff_coords is None:
            coords = self._get_coords(f).copy()
            x = coords[:, 0]
            y = coords[:, 1]
            x *= 375
            y *= 300
            self.ff_coords = x, y
        x, y = self.ff_coords

        if self.fftriang is None:
            triangles = f["/post/constant/connectivities/SHELL/erfblock/ic"][()]
            triangles = triangles - triangles.min()
            triangles = triangles[:, :-1]
            xi = np.linspace(0, 375, 376)
            yi = np.linspace(0, 300, 301)
            Xi, Yi = np.meshgrid(xi, yi)
            self.fftriang = tri.Triangulation(x, y, triangles=triangles)

        # Fiber fraction map creation with tripcolor
        fvc_old = f["/post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res"][()].flatten()
        fvc = f["/post/constant/entityresults/SHELL/PERMEABILITY1/ZONE1_set1/erfblock/res"][()][:, 0] * 100000

        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tripcolor(self.fftriang, fvc, cmap="gray",vmin=0, vmax=10)

        ax.set(xlim=(0, 375), ylim=(0, 300))
        ax.axis("off")
        fig.set_tight_layout(True)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for im in ax.get_images():
            im.set_clim(0, 1)
            
        perm_bytes = io.BytesIO()
        fig.savefig(perm_bytes, bbox_inches=extent, format="png")
        #fig.savefig("/home/schroeni/imgs/out.png", bbox_inches=extent, format="png")
        plt.close(fig)
        perm_bytes.seek(0)

        file_bytes = np.asarray(perm_bytes.getbuffer(), dtype=np.uint8)
        perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        perm_map = cv2.resize(perm_map, self.image_size)
        perm_map = cv2.rotate(perm_map, cv2.ROTATE_90_CLOCKWISE)

        return perm_map

    def _get_timesteps(self, outfile):
        content = outfile.readlines()
        finder = re.compile("STEP NO.")
        sci_num = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?")
        lis = []
        for line in content:
            find = finder.search(line)
            if find is not None:
                t = sci_num.findall(line)
                lis.append(np.array([float(i) for i in t]))
        return np.stack(lis)

    def _get_sensordata(self, f):
        try:
            data = f["post"]["multistate"]["TIMESERIES1"]["multientityresults"][
                "SENSOR"
            ]["PRESSURE"]["ZONE1_set1"]["erfblock"]["res"][()]

            states = f["post"]["singlestate"]
        except KeyError:
            return None

        states = list(states)[
            self.skip_indizes[0]: self.skip_indizes[1]: self.skip_indizes[2]
        ]

        def sensordata_gen():
            for state in states:
                try:
                    s = state.replace("state", "")
                    state_num = int(s)
                    sensordata = np.squeeze(data[state_num - 1])
                    if self.divide_by_100k:
                        # convert barye to bar ( smaller values are more stable while training)
                        sensordata = sensordata / 100000
                    else:
                        # Standardize
                        sensordata = (sensordata - self.mean) / self.std
                    if self.sensor_indizes != ((0, 1), (0, 1)):
                        sensordata = sensordata.reshape((38, 30))
                        sensordata = sensordata[
                            self.sensor_indizes[0][0]:: self.sensor_indizes[0][1],
                            self.sensor_indizes[1][0]:: self.sensor_indizes[1][1],
                        ]
                        sensordata = sensordata.flatten()
                    yield sensordata
                except IndexError:
                    yield None

        return sensordata_gen()

    def get_data(self, file):
        try:
            result_f = h5py.File(file, "r")
            p_out_f = open(str(file).replace("_RESULT.erfh5", "p.out"), "r")
            """ if self.ignore_useless_states:
                meta_f = h5py.File(
                    str(file).replace("RESULT.erfh5", "meta_data.hdf5"), "r"
                )
            else:
                meta_f = None """
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error: File not found: {file} (or meta_data.hdf5)")
            return None

        time_steps = self._get_timesteps(p_out_f)
        states = list(result_f["post"]["singlestate"])
        states_int = [int(r.replace("state", "0")) - 1 for r in states]
        fillings = []
        for state in states:
            try:
                fillings.append(
                    result_f["post"]["singlestate"][state]["entityresults"]["NODE"][
                        "FILLING_FACTOR"
                    ]["ZONE1_set1"]["erfblock"]["res"][()]
                )
            except KeyError:
                return
        ones = np.ones_like(fillings[0])
        u = np.sum(ones)
        percentages = [(np.sum(k) / u) for k in fillings]
        multi_state_pressure = result_f[
            "/post/multistate/TIMESERIES1/multientityresults/SENSOR"
            "/PRESSURE/ZONE1_set1/erfblock/res"
        ][()]
        # Assuming that hitting all sensors = 100 %
        all_states_int = list(range(len(time_steps)))
        m = multi_state_pressure.squeeze()
        activated_sensors = np.count_nonzero(m, axis=1)
        percentage_of_all_sensors = activated_sensors / 1140  # Number of sensors
        y_label = "filling_perc"
        return (
            states_int,
            time_steps,
            all_states_int,
            y_label,
            percentages,
            percentage_of_all_sensors,
        )

    def plot_times_more_detailed(self, file: Path, name, save_to_file=True):
        (
            states_int,
            time_steps,
            all_states_int,
            y_label,
            percentages,
            percentage_of_all_sensors,
        ) = self.get_data(file)
        import matplotlib.pyplot as plt

        k = len(states_int)
        cut = int(0.20 * k)

        fig, ax1 = plt.subplots()
        color = "tab:blue"
        ax1.set_xlabel("steps")
        ax1.set_ylabel("huuis", color=color)
        ax1.plot(time_steps[all_states_int[:-cut], 1], color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:red"
        # we already handled the x-label with ax1
        ax2.set_ylabel(y_label, color=color)
        ax2.scatter(
            states_int[:-cut],
            percentages[:-cut],
            color=color,
            label="Filling perc / Single states",
            s=2,
        )
        ax2.plot(percentage_of_all_sensors[:-cut],
                 color="green", label="Sensor hit %")
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()
        plt.title(p.stem)
        if save_to_file:
            plt.savefig(f"figs/{name}.png")
        else:
            plt.show()
        plt.close()

    def plot_times_thresholded(self, file: Path, name, save_to_file=True):
        (
            states_int,
            time_steps,
            all_states_int,
            y_label,
            percentages,
            percentage_of_all_sensors,
        ) = self.get_data(file)
        import matplotlib.pyplot as plt

        # k = len(states_int)
        # cut = int(0.20*k)
        threshold = 0.97
        cut = np.count_nonzero(
            np.where(percentage_of_all_sensors < threshold, 0, 1))
        fig, ax1 = plt.subplots()
        color = "tab:blue"
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Sensor hit %", color="green")
        ax1.tick_params(axis="y", labelcolor=color)
        if len(time_steps[1:, 1]) != len(percentage_of_all_sensors):
            print("NON matching lenghts.")
            plt.close()
            return
        if cut == 0:  # Filling never reaches the threshold -> No need to cut off
            _t = time_steps[:, 1][1:]
            _p = percentage_of_all_sensors
        else:
            _t = time_steps[:, 1][1:-cut]
            _p = percentage_of_all_sensors[:-cut]
        ax1.plot(_t, _p, color="green")

        plt.title(p.stem)
        if save_to_file:
            plt.savefig(f"figs/{name}_sensor_hit_only_20perc_cut.png")
        else:
            plt.show()
        plt.close()

    def plot_times(self, file: Path, name, save_to_file=True):
        (
            states_int,
            time_steps,
            all_states_int,
            y_label,
            percentages,
            percentage_of_all_sensors,
        ) = self.get_data(file)
        import matplotlib.pyplot as plt

        k = len(states_int)
        cut = int(0.20 * k)

        fig, ax1 = plt.subplots()
        color = "tab:blue"
        ax1.set_xlabel("steps")
        ax1.set_ylabel("huuis", color=color)
        ax1.plot(time_steps[states_int[:-cut], 1], color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = "tab:red"
        ax2.set_ylabel(
            "filling_perc", color=color
        )  # we already handled the x-label with ax1
        ax2.plot(percentages[:-cut], color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        fig.tight_layout()
        if save_to_file:
            plt.savefig("figs/" + name + ".png")
        else:
            plt.show()
        plt.close()

    def get_sensordata_and_flowfront(self, file: Path):
        try:
            result_f = h5py.File(file, "r")
            if self.ignore_useless_states:
                meta_f = h5py.File(
                    str(file).replace("RESULT.erfh5", "meta_data.hdf5"), "r"
                )
            else:
                meta_f = None
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error: File not found: {file} (or meta_data.hdf5)")
            return None

        fillings = self._get_flowfront(result_f, meta_f)
        if not fillings:
            return None

        sensor_data = self._get_sensordata(result_f)
        if not sensor_data:
            return None

        # Return only tuples without None values and if we get no data at all, return None
        # `if not None in t` does not work here because numpy does some weird stuff on
        # such comparisons
        return (
            list(
                (sens_data, filling, {"state": state})
                for sens_data, filling, state in zip(
                    sensor_data, fillings, result_f["post"]["singlestate"]
                )
                if sens_data is not None and filling is not None
            )
            or None
        )

    def _get_coords(self, f: h5py.File):
        if self.coords is not None:
            return self.coords
        self.coords = extract_coords_of_mesh_nodes(Path(f.filename))
        return self.coords


class DataloaderImageSequences(DataloaderImages):
    """
    Subclass for dataloader functions that generate sequences of images
    as samples and only generate one sample per file
    """

    def __init__(self, image_size=(135, 103), wanted_frames=10):
        super().__init__(image_size=image_size)
        self.wanted_frames = wanted_frames
        self.coords = None
    
    def get_coords(self,f):
        if self.coords is None:
           self.coords = extract_coords_of_mesh_nodes(f)
        return self.coords

    def get_sensor_to_perm_map(self, filename):
        try:
            per_step = 0.01
            # logger = logging.getLogger(__name__)
            # logger.debug(
            #     "Loading flow front and permeability maps from {}".format(
            #         filename)
            # )
            f = h5py.File(filename, "r")
            perm_map = self._get_fiber_fraction(f)
            perm_map = perm_map.astype(np.float) / 255
            multi_state_pressure = f[
                "/post/multistate/TIMESERIES1/multientityresults/SENSOR"
                "/PRESSURE/ZONE1_set1/erfblock/res"
            ][()]
            m = multi_state_pressure.squeeze()
            activated_sensors = np.count_nonzero(m, axis=1)
            percentage_of_all_sensors = activated_sensors / 1140  # Number of sensors
            sequence = np.zeros((100, 1140))
            current = 0
            for i, sample in enumerate(percentage_of_all_sensors):
                if sample >= current:
                    sequence[int(current * 100), :] = m[i, :]
                    current += per_step
            return [(sequence, np.array(perm_map))]
        except Exception:
            return None

    def get_flowfront_to_perm_map(self, filename):
        try:
            per_step = 0.01
            # logger = logging.getLogger(__name__)
            # logger.debug(
            #     "Loading flow front and premeability maps from {}".format(
            #         filename)
            # )
            f = h5py.File(filename, "r")
            perm_map = self._get_fiber_fraction(f)
            perm_map = perm_map.astype(np.float) / 255
            multi_state_pressure = f[
                "/post/multistate/TIMESERIES1/multientityresults/SENSOR"
                "/PRESSURE/ZONE1_set1/erfblock/res"
            ][()]
            m = multi_state_pressure.squeeze()
            states = list(f["post"]["singlestate"])
            
            fillings = []
            for state in states:
                try:
                    fillings.append(
                        f["post"]["singlestate"][state]["entityresults"]["NODE"][
                            "FILLING_FACTOR"
                        ]["ZONE1_set1"]["erfblock"]["res"][()]
                    )
                except KeyError as e:
                    return None
            fillings = np.stack(fillings).squeeze()
            activated_pixels = np.count_nonzero(fillings, axis=1)
            percentage_of_all_sensors = activated_pixels / 28464  # Number individual points
            sequence = np.zeros((100, 143, 111))
            current = 0
            coords = self.get_coords(filename)
            for i, sample in enumerate(percentage_of_all_sensors):
                if sample >= current:
                    sequence[int(current * 100), :, :] = create_np_image((143,111), coords, fillings[i, :])
                    current += per_step
            return [(sequence, np.array(perm_map))]
        except Exception as e:
            return None

    def get_images_of_flow_front_and_permeability_map(self, filename):
        logger = logging.getLogger(__name__)
        logger.debug(
            "Loading flow front and permeability maps from {}".format(filename)
        )
        f = h5py.File(filename, "r")

        perm_map = self._get_fiber_fraction(f)
        perm_map = perm_map.astype(np.float) / 255

        all_states = list(f["post"]["singlestate"].keys())
        indices = get_fixed_number_of_indices(
            len(all_states), self.wanted_frames)
        if indices is None:
            return None
        try:
            wanted_states = [all_states[i] for i in indices]
        except IndexError or OSError:
            logger.error(
                f"ERROR at {filename}, available states: {all_states},"
                f"wanted indices: {indices}"
            )
            raise
        ffgen = self._get_flowfront(f, states=wanted_states, meta_f=None)
        if ffgen is None:
            return None
        images = list(ffgen)

        img_stack = np.stack(images)
        return [(img_stack[0: self.wanted_frames], perm_map)]


if __name__ == "__main__":
    dl = DataloaderImageSequences()

    model = FFTFF()
    path = r"/cfs/share/cache/output_schroeni/2021-01-14_17-02-09/checkpoint.pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")

    new_model_state_dict = OrderedDict()
    model_state_dict = checkpoint["model_state_dict"]
    if "swt-dgx" not in socket.gethostname():
        for k, v in model_state_dict.items():
            if k.startswith("module"):
                k = k[7:]  # remove `module.`
            new_model_state_dict[k] = v
        model.load_state_dict(new_model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
   

   

    root = tr_resources.data_root / "2019-11-29_16-56-17_10000p"
    for num in range(100):

        p = Path(root / f"{num}/2019-11-29_16-56-17_{num}_RESULT.erfh5")

        ret = dl.get_flowfront_to_perm_map(p)
        # dl.plot_times(p, num, save_to_file=False)
        # dl.plot_times_thresholded(p, num, save_to_file=False)
        # dl.plot_times_more_detailed(p, num, save_to_file=False)
        # dl.plot_times(p, use_single_states=True)
