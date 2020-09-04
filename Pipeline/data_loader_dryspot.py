import logging

import h5py
import numpy as np
from PIL import Image
from itertools import groupby

import Resources.training as tr_resources
from Utils.data_utils import normalize_coords, load_mean_std
from Utils.img_utils import create_np_image


class DataloaderDryspots:
    def __init__(self, image_size=None,
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1),
                 divide_by_100k=True,
                 aux_info=False
                 ):
        self.image_size = image_size
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        self.divide_by_100k = divide_by_100k
        self.mean = None
        self.std = None
        self.aux_info = aux_info
        if not self.divide_by_100k:
            self.mean, self.std = load_mean_std(tr_resources.mean_std_1140_pressure_sensors)

    def get_flowfront_bool_dryspot(self, filename, states=None):
        """
        Load the flow front for the all states or given states. Returns a bool label for dryspots.
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
            coord_as_np_array = f[
                "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/"
                "erfblock/res"
            ][()]
            # Cut off last column (z), since it is filled with 1s anyway
            _coords = coord_as_np_array[:, :-1]
            _coords = normalize_coords(_coords)
            if not states:
                states = f["post"]["singlestate"]

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

            filling_factors_at_certain_times = [
                f["post"]["singlestate"][state]["entityresults"]["NODE"][
                    "FILLING_FACTOR"][
                    "ZONE1_set1"
                ]["erfblock"]["res"][()]
                for state in states
            ]

            flat_fillings = np.squeeze(filling_factors_at_certain_times)
            instances = []
            for filling, state in zip(flat_fillings, states):
                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                if int(str(state).replace("state", "0")) in set_of_states:
                    label = 1
                instances.append((create_np_image(target_shape=self.image_size, norm_coords=_coords, data=filling),
                                  label))
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            logger = logging.getLogger()

            logger.warning(f'Warning: {filename}')
            f.close()
            meta_file.close()
            return None

    def get_sensor_bool_dryspot(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
            pressure_array = \
                f['post']['multistate']['TIMESERIES1']['multientityresults'][
                    'SENSOR']['PRESSURE']['ZONE1_set1'][
                    'erfblock'][
                    'res'][()]
            instances = []
            states = f["post"]["singlestate"]

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

            for i, state in enumerate(states):
                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                state_num = int(str(state).replace("state", "0"))
                if state_num in set_of_states:
                    label = 1
                try:
                    data = np.squeeze(pressure_array[state_num - 1])
                    if self.divide_by_100k:
                        # "Normalize data" to fit betw. 0 and 1
                        data = data / 100000
                    else:
                        # Standardize data for each sensor
                        data = (data - self.mean) / self.std
                    if self.sensor_indizes != ((0, 1), (0, 1)):
                        rect = data.reshape(38, 30)
                        sel = rect[self.sensor_indizes[0][0]::self.sensor_indizes[0][1],
                                   self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
                        data = sel.flatten()
                    if self.aux_info:
                        instances.append((data, label, {"ix": i, "max": len(states)}))
                    else:
                        instances.append((data, label))
                except IndexError:
                    continue
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            f.close()
            meta_file.close()
            return None

    def get_sensor_bool_dryspot_resized_matrix(self, filename, output_size=(224, 224)):
        """
        Loads the sensor values of a files resized as a matrix (size specified in output_size) and
        Dryspot yes/no as label.
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
            pressure_array = \
                f['post']['multistate']['TIMESERIES1']['multientityresults'][
                    'SENSOR']['PRESSURE']['ZONE1_set1'][
                    'erfblock'][
                    'res'][()]
            instances = []
            states = f["post"]["singlestate"]

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

            for i, state in enumerate(states):
                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                state_num = int(str(state).replace("state", "0"))
                if state_num in set_of_states:
                    label = 1
                try:
                    data = np.squeeze(pressure_array[state_num - 1])
                    if self.divide_by_100k:
                        # "Normalize data" to fit betw. 0 and 1
                        data = data / 100000
                    else:
                        # Standardize data for each sensor
                        data = (data - self.mean) / self.std

                    rect = data.reshape(38, 30)
                    sel = rect[self.sensor_indizes[0][0]::self.sensor_indizes[0][1], 
                               self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
                    data = np.array(Image.fromarray(sel).resize(size=output_size, resample=Image.BILINEAR))
                    data = np.expand_dims(data, axis=0)
                    data = np.repeat(data, 3, axis=0)               
                    if self.aux_info:
                        instances.append((data, label, {"ix": i, "max": len(states)}))
                    else:
                        instances.append((data, label))
                except IndexError:
                    continue
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            f.close()
            meta_file.close()
            return None

    def get_sensor_bool_dryspot_runlevel(self, filename, threshold_min_counted_dryspots=5):
        """
        Loads sensor values of a run and subsamples the number of timesteps (based on filling percentage),
        and determines a runwise label ("ok"/"not ok") based on consecutive dryspots counted

        Args:
            filename: full path (pathlib object) to an ERFH5-file containing sensor data
            threshold_min_counted_dryspots: min. number of consecutive dryspots to turn the runlevel label to "not ok"

        Returns: List of a single tuple containing
                    1) sequence of sensor values from 100 selected timesteps
                    2) runlevel label (0 or 1)

        """
        f = h5py.File(filename, "r")
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')

        try:
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
            states = f["post"]["singlestate"]

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

            multi_state_pressure = f[
                "/post/multistate/TIMESERIES1/multientityresults/SENSOR"
                "/PRESSURE/ZONE1_set1/erfblock/res"
            ][()]
            m = multi_state_pressure.squeeze()
            activated_sensors = np.count_nonzero(m, axis=1)
            percentage_of_all_sensors = activated_sensors / 1140  # Number of sensors
            sequence = np.zeros((100, 1140))
            per_step = 0.01
            current = 0
            frame_labels = []

            for sample in states:
                state_num = int(str(sample).replace("state", "0"))
                try:
                    sample_percentage = percentage_of_all_sensors[state_num - 1]
                    if sample_percentage >= current:
                        sequence[int(current * 100), :] = m[state_num - 1, :]
                        current += per_step

                        frame_label = 0
                        if state_num in set_of_states:
                            frame_label = 1
                        frame_labels.append(frame_label)
                except IndexError:
                    continue

            # determine runlevel label using frame labels and threshold
            lens_of_runs_of_dryspots = [sum(1 for _ in group) for key, group in
                                               groupby(np.array(frame_labels) == 1) if key]
            max_len = 0 if len(lens_of_runs_of_dryspots) == 0 else max(lens_of_runs_of_dryspots)
            label = 0 if max_len < threshold_min_counted_dryspots else 1

            f.close()
            meta_file.close()

            return [(sequence, label)]
        except KeyError:
            f.close()
            meta_file.close()
            return None
