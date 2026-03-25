"""
Dataset and HDF5 creation for shellgnn data processing.
"""
import re
import h5py
import numpy as np
from preprocess.utils import get_nested_value
from preprocess.read_raw_data import ReadRawData

class ReadData(ReadRawData):
    def __init__(self, logger):
        super().__init__(logger)  # Pass logger to ReadData

    def read_fem_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                for line in file:
                    if re.match(r"^SPCADD", line):
                        self.read_spc_subcase(line, mode="spc")
        except Exception as e:
            self.logger.error(f"Error reading FEM file {file_path}: {e}")

    def read_constr_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                for line in file:
                    if re.match(r"^SPC", line):
                        self.read_spc(line, mode="spc")
        except Exception as e:
            self.logger.error(f"Error reading constraint file {file_path}: {e}")

    def read_geom_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                rbe_f = ""  # flag for reading rbe cont lines
                rb2_id, rb3_id = None, None
                for line in file:
                    if re.match(r"^GRID", line):
                        self.read_gen(line, mode="node")
                    elif re.match(r"^CTRIA3", line):
                        self.read_gen(line, mode="tria")
                    elif re.match(r"^PSHELL", line):
                        self.read_gen(line, mode="pshell")
                    elif re.match(r"^RBE2", line):
                        rb2_id = self.read_rbe(line, mode="rb2")
                        rbe_f = "rb2"
                    elif re.match(r"^\+", line) and rbe_f == "rb2" and rb2_id:
                        self.read_rbe(line, mode="rb2", cont=True, rid=rb2_id)
                    elif re.match(r"^RBE3", line):
                        rb3_id = self.read_rbe(line, mode="rb3")
                        rbe_f = "rb3"
                    elif re.match(r"^\+", line) and rbe_f == "rb3" and rb3_id:
                        self.read_rbe(line, mode="rb3", cont=True, rid=rb3_id)
                    else:
                        rb2_id, rb3_id, rbe_f = None, None, ""
        except Exception as e:
            self.logger.error(f"Error reading geometry file {file_path}: {e}")

    def read_pch_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                sub_id = None
                for line in file:
                    if re.match(r"^\$SUBCASE", line):
                        sub_id = re.search(r"(\d+)", line).group(1)
                    elif re.match(r"^\s+(\d+)", line):
                        self.read_disp(line, sub_id)
        except Exception as e:
            self.logger.error(f"Error reading PCH file {file_path}: {e}")

    def read(self, geom_file, constr_file, fem_file, pch_file):
        try:
            self.read_geom_file(geom_file)
            self.read_pch_file(pch_file)
            self.read_fem_file(fem_file)
            self.read_constr_file(constr_file)
            self.organize()
            self.rescale_node_indices()
            self.log_metadata()
        except Exception as e:
            self.logger.error(f"Error during file reading and processing: {e}")

class Dataset(ReadData):
    def __init__(self, logger):
        super().__init__(logger)  # Pass logger to ReadData

    def _get_edges(self, edge_conn, edge_types):
        try:
            for elem_id, edges in self.edge_features.items():
                if edges:
                    for edge_id, data in edges.items():
                        edge_conn.append(tuple(map(int, data["edge"])))
                        edge_types.append(int(data["edge_type"]))
            edge_conn = np.array(edge_conn, dtype=np.int64)
            edge_types = np.array(edge_types, dtype=np.int64)
            assert len(edge_conn) == len(edge_types)
            return edge_conn, edge_types
        except Exception as e:
            self.logger.error(f"Error getting edges: {e}")

    def _get_node_features(self, sub_id, coords, ntypes, thickness, spc, force, disp):
        try:
            for nid, ndata in self.nodal_features[sub_id].items():
                coords.append(
                    np.array(list(map(np.float32, get_nested_value(ndata, ["data"]))))
                )
                ntypes.append(int(get_nested_value(ndata, ["type"])))
                thickness.append(np.float32(get_nested_value(ndata, ["thickness"])))
                spc.append(np.array(list(map(int, get_nested_value(ndata, ["spc"])))) )
                force.append(
                    np.array(list(map(np.float32, get_nested_value(ndata, ["force"]))))
                )
                disp.append(
                    np.array(list(map(np.float32, get_nested_value(ndata, ["y"]))))
                )
            assert all(
                len(lst) == len(coords) for lst in [ntypes, thickness, spc, force, disp]
            )
        except Exception as e:
            self.logger.error(f"Error getting node features for sub_id {sub_id}: {e}")

    def create_hdf5(self, data_file: h5py.File, variant):
        try:
            var_no = data_file.require_group(f"{variant}")
            edge_conn, edge_types = [], []
            edge_conn, edge_types = self._get_edges(edge_conn, edge_types)
            for sub_id in self.nodal_features:
                coords, ntypes, thickness, spc, force, disp = [list() for i in range(6)]
                self._get_node_features(
                    sub_id, coords, ntypes, thickness, spc, force, disp
                )
                var_no.create_dataset(f"{sub_id}/pos", data=np.array(coords))
                var_no.create_dataset(f"{sub_id}/ntypes", data=ntypes)
                var_no.create_dataset(f"{sub_id}/thickness", data=thickness)
                var_no.create_dataset(f"{sub_id}/spc", data=spc)
                var_no.create_dataset(f"{sub_id}/load", data=force)
                var_no.create_dataset(f"{sub_id}/y", data=disp)
                var_no.create_dataset(f"{sub_id}/connectivity", data=edge_conn)
                var_no.create_dataset(f"{sub_id}/etypes", data=edge_types)
        except Exception as e:
            self.logger.error(f"Error creating HDF5 dataset: {e}")
