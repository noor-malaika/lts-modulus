"""
Raw data reading and feature extraction for shellgnn data processing.
"""
from itertools import combinations
import traceback
import copy
from preprocess.utils import fix_scientific_notation

class ReadRawData:
    def __init__(self, logger):
        self.logger = logger
        self.nodal_features = {}
        self.node = {}
        self.edge_features = {}
        self.tria = {}
        self.rb2 = {}
        self.rb3 = {}
        self.pshell = {}
        self.spc = {}
        self.force = {
            "805": {
                "nid": "74026",
                "force": [0, 4000, 0],
            },
            "806": {
                "nid": "74027",
                "force": [0, 4000, 0],
            },
        }
        self.outputs = {
            "805": {},
            "806": {},
        }
        self.material = {}

    def split_components(self, line):
        vals = [
            line[i : i + 8].strip()
            for i in range(0, len(line), 8)
            if line[i : i + 8].strip()
        ]
        return vals

    def read_gen(self, line, mode):
        vals = self.split_components(line)
        try:
            if mode == "node":
                getattr(self, mode)[vals[1]] = {"data": tuple(map(float, vals[2:5]))}
                if len(getattr(self, mode)[vals[1]]["data"]) != 3:
                    self.logger.error("Incompatible dimension for nodes.")
            else:
                getattr(self, mode)[vals[1]] = {"data": tuple(map(str, vals[2:]))}
        except ValueError:
            if mode == "node":
                getattr(self, mode)[vals[1]] = {
                    "data": tuple([fix_scientific_notation(i) for i in vals[2:5]])
                }
                self.logger.warning(f"Fixed notation for {vals[2:5]}")
                if len(getattr(self, mode)[vals[1]]["data"]) != 3:
                    self.logger.error("Incompatible dimension for nodes.")
        except Exception as e:
            self.logger.error(f"Couldn't read {mode} data {e}")
            self.logger.error(traceback.format_exc())

    def read_rbe(self, line, mode, cont=False, rid=None):
        vals = self.split_components(line)
        try:
            if cont:
                getattr(self, mode)[rid]["data"] += tuple(vals[1:])
            else:
                rid = vals[1]
                getattr(self, mode)[rid] = {"data": tuple(vals[2:])}
                return rid
        except Exception as e:
            self.logger.error(f"Error reading {mode} data: {str(e)}")
            self.logger.error(traceback.format_exc())

    def read_spc_subcase(self, line, mode):
        vals = self.split_components(line)
        try:
            getattr(self, mode)[vals[1]] = {}
            getattr(self, mode)[vals[1]][vals[2]] = {}
            getattr(self, mode)[vals[1]][vals[3]] = {}
        except Exception as e:
            self.logger.error(f"Error reading {mode} data: {str(e)}")
            self.logger.error(traceback.format_exc())

    def read_spc(self, line, mode):
        vals = self.split_components(line)
        try:
            for sub_id in self.spc.keys():
                for sid in self.spc[sub_id].keys():
                    if vals[1] == sid:
                        self.spc[sub_id][sid][f"7{vals[2][2:]}"] = list(vals[3])
        except Exception as e:
            self.logger.error(f"Error reading {mode} data: {str(e)}")
            self.logger.error(traceback.format_exc())

    def read_disp(self, line, sub_id):
        vals = [i.strip() for i in line.split(" ") if i]
        self.outputs[sub_id][vals[0]] = vals[2:]

    def rm_nodes_with_no_types(self, no_types):
        for nid in no_types:
            del self.node[nid]

    def _create_tria_edges(self):
        try:
            tria_edges = set()
            for eid, data in self.tria.items():
                self.edge_features[eid] = {}
                nids = data["data"][1:]
                if len(nids) != 3:
                    self.logger.error(f"Warning: TRIA {eid} has incorrect node count: {len(nids)}")
                temp_edges = list(combinations(nids, 2))
                for i, edge in enumerate(temp_edges):
                    edge = tuple(sorted(edge))
                    if edge not in tria_edges:
                        tria_edges.add(edge)
                        self.edge_features[eid][i] = {"edge": edge, "edge_type": "0"}
        except Exception as e:
            self.logger.error(f"Error creating tria edges: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _create_rb2_edges(self):
        try:
            for eid, data in self.rb2.items():
                self.edge_features[eid] = {}
                master_id = data["data"][0]
                slave_ids = data["data"][2:]
                if master_id is None or not slave_ids:
                    self.logger.error(f"Warning: RB2 {eid} has missing master/slave data")
                for i, sl_id in enumerate(slave_ids):
                    self.edge_features[eid][i] = {"edge": (master_id, sl_id), "edge_type": "1"}
        except Exception as e:
            self.logger.error(f"Error creating rb2 edges: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _create_rb3_edges(self):
        try:
            for eid, data in self.rb3.items():
                self.edge_features[eid] = {}
                master_id = data["data"][0]
                slave_ids = data["data"][4:]
                if master_id is None or not slave_ids:
                    self.logger.error(f"Warning: RB3 {eid} has missing master/slave data")
                for i, sl_id in enumerate(slave_ids):
                    self.edge_features[eid][i] = {"edge": (master_id, sl_id), "edge_type": "2"}
        except Exception as e:
            self.logger.error(f"Error creating rb3 edges: {str(e)}")
            self.logger.error(traceback.format_exc())

    def create_edges(self):
        try:
            self._create_tria_edges()
            self._create_rb2_edges()
            self._create_rb3_edges()
        except Exception as e:
            self.logger.error(f"Error creating edges: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _add_tria_nodes(self):
        try:
            for eid, data in self.tria.items():
                data = data["data"]
                pid = data[0]
                node_ids = data[1:]
                for node_id in node_ids:
                    self.node[node_id]["type"] = "0"
                    self.node[node_id]["thickness"] = self.pshell[pid]["data"][1]
        except Exception as e:
            self.logger.error(f"Error adding tria nodes: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _add_rb2_nodes(self):
        try:
            for eid, data in self.rb2.items():
                slave_nids = data["data"][2:]
                master_nid = data["data"][0]
                self.node[master_nid]["type"] = "1"
                self.node[master_nid]["thickness"] = "0"
                for nid in slave_nids:
                    self.node[nid]["type"] = "3"
        except Exception as e:
            self.logger.error(f"Error adding rb2 nodes: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _add_rb3_nodes(self):
        try:
            for eid, data in self.rb3.items():
                slave_nids = data["data"][4:]
                master_nid = data["data"][0]
                self.node[master_nid]["type"] = "2"
                self.node[master_nid]["thickness"] = "0"
                for nid in slave_nids:
                    self.node[nid]["type"] = "4"
        except Exception as e:
            self.logger.error(f"Error adding rb3 nodes: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _add_spc_and_force(self):
        try:
            for nid in self.node.keys():
                self.node[nid]["spc"] = [0, 0, 0]
                self.node[nid]["force"] = [0, 0, 0]
            for sub_id in self.spc.keys():
                node_copy = copy.deepcopy(self.node)
                for sid in self.spc[sub_id]:
                    for nid, comps in self.spc[sub_id][sid].items():
                        dof_constraints = [int(i) for i in comps]
                        for dof in dof_constraints:
                            node_copy[nid]["spc"][dof - 1] = 1
                self.nodal_features[sub_id] = node_copy
                force_node_id = self.force[sub_id]["nid"]
                self.nodal_features[sub_id][force_node_id]["force"] = self.force[sub_id]["force"]
        except Exception as e:
            self.logger.error(f"Error adding force and spc constraints: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _set_zero_disp_for_constrained_nodes(self):
        try:
            for sub_id in self.spc.keys():
                for sid in self.spc[sub_id]:
                    for nid, comps in self.spc[sub_id][sid].items():
                        if nid in self.nodal_features[sub_id].keys():
                            comp_ids = [
                                (
                                    int(comps[i]) - 1
                                    if i < len(comps) and comps[i]
                                    else None
                                )
                                for i in range(3)
                            ]
                            self.nodal_features[sub_id][nid]["y"] = [
                                (
                                    0
                                    if i in comp_ids
                                    else self.nodal_features[sub_id][nid]["y"][i]
                                )
                                for i in range(3)
                            ]
        except Exception as e:
            self.logger.error(
                f"Error setting spc nodes displacements to zero: {str(e)}"
            )
            self.logger.error(traceback.format_exc())

    def _add_displacements_as_outputs(self):
        try:
            for sub_id in self.outputs.keys():
                for nid, disps in self.outputs[sub_id].items():
                    if nid in self.nodal_features[sub_id].keys():
                        self.nodal_features[sub_id][nid]["y"] = disps
            self._set_zero_disp_for_constrained_nodes()
        except Exception as e:
            self.logger.error(f"Error adding displacements: {str(e)}")
            self.logger.error(traceback.format_exc())

    def create_node_features(self):
        try:
            self._add_tria_nodes()
            self._add_rb2_nodes()
            self._add_rb3_nodes()
            self.logger.info(f"Total read nodes: {len(self.node.items())}")
            no_types = list(
                k for k, node in self.node.items() if "type" not in node.keys()
            )
            self.rm_nodes_with_no_types(no_types)
            self._add_spc_and_force()
            self._add_displacements_as_outputs()
        except Exception as e:
            self.logger.error(f"Error creating node features: {str(e)}")
            self.logger.error(traceback.format_exc())

    def rescale_node_indices(self):
        sub_old_to_new = {}
        try:
            for sub_id in self.nodal_features:
                old_to_new = {}
                sorted_nids = [
                    str(x) for x in sorted(map(int, self.nodal_features[sub_id].keys()))
                ]
                for i, nid in enumerate(sorted_nids):
                    old_to_new[nid] = i
                sub_old_to_new[sub_id] = old_to_new
                self.nodal_features[sub_id] = {
                    old_to_new[nid]: ndata
                    for nid, ndata in self.nodal_features[sub_id].items()
                }
            for elem_id, edges in self.edge_features.items():
                for eid in self.edge_features[elem_id]:
                    self.edge_features[elem_id][eid]["edge"] = [
                        (sub_old_to_new["806"][src], sub_old_to_new["806"][dst])
                        for src, dst in [self.edge_features[elem_id][eid]["edge"]]
                    ][0]
        except Exception as e:
            self.logger.error(f"Some error occurred during rescaling node ids {e}")

    def organize(self):
        try:
            self.create_node_features()
            self.create_edges()
        except Exception as e:
            self.logger.error(
                f"Error restructuring data to node and edge features: {str(e)}"
            )
            self.logger.error(traceback.format_exc())

    def log_metadata(self):
        self.logger.info(f"Total read tria elements: {len(self.tria)}")
        self.logger.info(f"Total read rb2 elements: {len(self.rb2)}")
        self.logger.info(f"Total read rb3 elements: {len(self.rb3)}")
        for sub_id in self.nodal_features:
            self.logger.info(
                f"Total extracted nodes for subcase {sub_id}: {len(self.nodal_features[sub_id])}"
            )
            tr_nodes, rb2_nodes, rb3_nodes = 0, [0, 0], [0, 0]
            for nid, data in self.nodal_features[sub_id].items():
                if data["type"] == "0":
                    tr_nodes += 1
                elif data["type"] == "1":
                    rb2_nodes[0] += 1
                elif data["type"] == "2":
                    rb3_nodes[0] += 1
                elif data["type"] == "3":
                    rb2_nodes[1] += 1
                elif data["type"] == "4":
                    rb3_nodes[1] += 1
            self.logger.info(f"\tTria nodes: {tr_nodes}")
            self.logger.info(f"\tRb2 master nodes: {rb2_nodes[0]}")
            self.logger.info(f"\tRb2 slave nodes: {rb2_nodes[1]}")
            self.logger.info(f"\tRb3 master nodes: {rb3_nodes[0]}")
            self.logger.info(f"\tRb3 slave nodes: {rb3_nodes[1]}")
        tr_edges, rb2_edges, rb3_edges = 0, 0, 0
        total_edges = sum(len(edges) for edges in self.edge_features.values())
        self.logger.info(f"Total extracted edges: {total_edges}")
        for elem_id in self.edge_features:
            for edge_id, data in self.edge_features[elem_id].items():
                if data["edge_type"] == "0":
                    tr_edges += 1
                elif data["edge_type"] == "1":
                    rb2_edges += 1
                elif data["edge_type"] == "2":
                    rb3_edges += 1
        assert (
            total_edges == tr_edges + rb2_edges + rb3_edges
        ), "Mismatch between total edges and individual edge counts"
        self.logger.info(f"\tTria edges: {tr_edges}")
        self.logger.info(f"\tRb2 edges: {rb2_edges}")
        self.logger.info(f"\tRb3 edges: {rb3_edges}")
