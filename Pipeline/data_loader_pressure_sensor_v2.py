import logging
import pickle
from pathlib import Path

import h5py
import numpy as np

import Resources.training as r
from Utils.data_utils import extract_nearest_mesh_nodes_to_sensors, get_node_propery_at_states_and_indices, \
    extract_coords_of_mesh_nodes


class DataloaderPressureSensorV2:
    def __init__(self, image_size=None,
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1),
                 ):
        self.image_size = image_size
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        self.mean = None
        self.std = None
        self.indeces_of_sensors = None
        self.coords = None
        self.load_coords_sensors()

    def load_coords_sensors(self):
        if r.nearest_nodes_to_sensors.is_file():
            with open(r.nearest_nodes_to_sensors, "rb") as nearest_nodes:
                _all_sensors = pickle.load(nearest_nodes)
        else:
            _all_sensors = extract_nearest_mesh_nodes_to_sensors(
                r.data_root / "2019-07-24_16-32-40_10p/0/2019-07-24_16-32-40_0")
            _all_sensors = _all_sensors.reshape((38, 30))
        indices_of_sensors = _all_sensors[self.sensor_indizes[0][0]::self.sensor_indizes[0][1],
                                          self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
        self.indeces_of_sensors = indices_of_sensors.flatten()

    def extract_data_from_result_file(self, filename):
        with h5py.File(filename, 'r') as f:
            try:
                states = f["post"]["singlestate"]
                states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

                self.extract_coords_data(f)
                pressure_values = get_node_propery_at_states_and_indices(f, "PRESSURE", states, self.indeces_of_sensors)
                pressure_values = np.squeeze(pressure_values)
            except KeyError:
                logger = logging.getLogger()
                logger.warning(f'Warning: {filename}')
                return None, None
        return states, pressure_values

    def extract_data_from_meta_file(self, filename):
        with h5py.File(filename, 'r') as meta_file:
            try:
                useless_states = []
                if self.ignore_useless_states:
                    useless_states = meta_file["useless_states/singlestates"][()]
                array_of_states = meta_file["dryspot_states/singlestates"][()]
                set_of_dryspot_states = set(array_of_states.flatten())
            except KeyError:
                logger = logging.getLogger()
                logger.warning(f'Warning: {filename}')
                return None, None
        return useless_states, set_of_dryspot_states

    def extract_coords_data(self, f: h5py.File):
        if self.coords is not None:
            return self.coords
        self.coords = extract_coords_of_mesh_nodes(Path(f.filename))

    def get_pressure_sensor_v2_bool_dryspot(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        states, pressure_values = self.extract_data_from_result_file(filename)
        if pressure_values is not None:
            pressure_values = pressure_values / 100000
        else:
            return None
        meta_fn = str(filename).replace("RESULT.erfh5", "meta_data.hdf5")
        useless_states, set_of_dryspot_states = self.extract_data_from_meta_file(meta_fn)
        if states is None or \
                pressure_values is None or \
                useless_states is None or \
                set_of_dryspot_states is None:
            return None
        instances = []
        for i, (pressure_values, state) in enumerate(zip(pressure_values, states)):
            if self.ignore_useless_states \
                    and len(useless_states) > 0 \
                    and state == f'state{useless_states[0]:012d}':
                break
            label = 0
            if int(str(state).replace("state", "0")) in set_of_dryspot_states:
                label = 1
            instances.append((pressure_values, label))
        return instances


if __name__ == '__main__':
    dl = DataloaderPressureSensorV2(sensor_indizes=((0, 1), (0, 1)))
    dl.get_pressure_sensor_v2_bool_dryspot(
        r'Y:\data\RTM\Leoben\sim_output\2019-07-24_16-32-40_10p\0\2019-07-24_16-32-40_0_RESULT.erfh5')
