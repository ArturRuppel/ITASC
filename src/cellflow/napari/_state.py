"""Serialize/restore nucleus-workflow widget control values.

Pure helpers: ``dump_state`` reads spinbox/checkbox/combo values into a dict,
``load_state`` writes them back. Each key is guarded so older snapshots that
predate newer controls still load cleanly.

The dict layout is part of the public state contract — main_widget persists
it across sessions and tests round-trip it. Keep keys stable; add, don't
rename.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget


def dump_state(w: NucleusWorkflowWidget) -> dict:
    return {
        "db_generation": {
            "min_area": w.db_gen_min_area_spin.value(),
            "max_area": w.db_gen_max_area_spin.value(),
            "threshold_pairs": w.threshold_pairs(),
            "min_frontier": w.db_gen_min_frontier_spin.value(),
            "ws_hierarchy": w.db_gen_ws_hierarchy_combo.currentText(),
            "max_distance": w.db_gen_max_dist_spin.value(),
            "max_neighbors": w.db_gen_max_neighbors_spin.value(),
            "linking_mode": w.db_gen_linking_mode_combo.currentText(),
            "area_weight": w.db_gen_area_weight_spin.value(),
            "iou_weight": w.db_gen_iou_weight_spin.value(),
            "distance_weight": w.db_gen_distance_weight_spin.value(),
            "quality_weight": w.db_gen_quality_weight_spin.value(),
            "quality_exponent": w.db_gen_quality_exp_spin.value(),
            "circularity_weight": w.db_gen_circularity_weight_spin.value(),
            "n_workers": w.db_gen_n_workers_spin.value(),
            "use_validated": w.db_gen_use_validated_check.isChecked(),
        },
        "extend": {
            "max_distance": w.extend_max_dist_spin.value(),
            "area_weight": w.extend_area_weight_spin.value(),
            "iou_weight": w.extend_iou_weight_spin.value(),
            "distance_weight": w.extend_distance_weight_spin.value(),
            "overlap_penalty": w.extend_overlap_penalty_spin.value(),
            "greedy_overwrite": w.extend_greedy_overwrite_check.isChecked(),
        },
        "ultrack": {
            "max_partitions": w.ultrack_max_partitions_spin.value(),
            "n_frames": w.ultrack_n_frames_spin.value(),
            "appear_weight": w.ultrack_appear_spin.value(),
            "disappear_weight": w.ultrack_disappear_spin.value(),
            "division_weight": w.ultrack_division_spin.value(),
            "power": w.ultrack_power_spin.value(),
            "bias": w.ultrack_bias_spin.value(),
        },
    }


def _set_combo(combo, text: str) -> None:
    idx = combo.findText(text)
    if idx >= 0:
        combo.setCurrentIndex(idx)


def load_state(w: NucleusWorkflowWidget, state: dict) -> None:
    if not isinstance(state, dict):
        return
    if "db_generation" in state:
        dbg = state["db_generation"]
        if "min_area" in dbg: w.db_gen_min_area_spin.setValue(dbg["min_area"])
        if "max_area" in dbg: w.db_gen_max_area_spin.setValue(dbg["max_area"])
        if "threshold_pairs" in dbg:
            w.set_threshold_pairs(list(dbg["threshold_pairs"]))
        if "min_frontier" in dbg: w.db_gen_min_frontier_spin.setValue(dbg["min_frontier"])
        if "ws_hierarchy" in dbg: _set_combo(w.db_gen_ws_hierarchy_combo, dbg["ws_hierarchy"])
        if "max_distance" in dbg: w.db_gen_max_dist_spin.setValue(dbg["max_distance"])
        if "max_neighbors" in dbg: w.db_gen_max_neighbors_spin.setValue(dbg["max_neighbors"])
        if "linking_mode" in dbg: _set_combo(w.db_gen_linking_mode_combo, dbg["linking_mode"])
        if "area_weight" in dbg: w.db_gen_area_weight_spin.setValue(dbg["area_weight"])
        if "iou_weight" in dbg: w.db_gen_iou_weight_spin.setValue(dbg["iou_weight"])
        if "distance_weight" in dbg: w.db_gen_distance_weight_spin.setValue(dbg["distance_weight"])
        if "quality_weight" in dbg: w.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
        if "quality_exponent" in dbg: w.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
        if "circularity_weight" in dbg: w.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
        if "n_workers" in dbg: w.db_gen_n_workers_spin.setValue(dbg["n_workers"])
        if "use_validated" in dbg: w.db_gen_use_validated_check.setChecked(dbg["use_validated"])
    if "extend" in state:
        ext = state["extend"]
        if "max_distance" in ext: w.extend_max_dist_spin.setValue(ext["max_distance"])
        if "area_weight" in ext: w.extend_area_weight_spin.setValue(ext["area_weight"])
        if "iou_weight" in ext: w.extend_iou_weight_spin.setValue(ext["iou_weight"])
        if "distance_weight" in ext: w.extend_distance_weight_spin.setValue(ext["distance_weight"])
        if "overlap_penalty" in ext: w.extend_overlap_penalty_spin.setValue(ext["overlap_penalty"])
        if "greedy_overwrite" in ext: w.extend_greedy_overwrite_check.setChecked(ext["greedy_overwrite"])
    if "ultrack" in state:
        ul = state["ultrack"]
        dbg_present = "db_generation" in state
        if "min_area" in ul and not (dbg_present and "min_area" in state["db_generation"]):
            w.db_gen_min_area_spin.setValue(ul["min_area"])
        if "max_partitions" in ul: w.ultrack_max_partitions_spin.setValue(ul["max_partitions"])
        if "n_frames" in ul: w.ultrack_n_frames_spin.setValue(ul["n_frames"])
        if "max_distance" in ul and not (dbg_present and "max_distance" in state["db_generation"]):
            w.db_gen_max_dist_spin.setValue(ul["max_distance"])
        if "linking_mode" in ul and not (dbg_present and "linking_mode" in state["db_generation"]):
            _set_combo(w.db_gen_linking_mode_combo, ul["linking_mode"])
        if "iou_weight" in ul and not (dbg_present and "iou_weight" in state["db_generation"]):
            w.db_gen_iou_weight_spin.setValue(ul["iou_weight"])
        if "area_weight" in ul and not (dbg_present and "area_weight" in state["db_generation"]):
            w.db_gen_area_weight_spin.setValue(ul["area_weight"])
        if "distance_weight" in ul and not (dbg_present and "distance_weight" in state["db_generation"]):
            w.db_gen_distance_weight_spin.setValue(ul["distance_weight"])
        if "appear_weight" in ul: w.ultrack_appear_spin.setValue(ul["appear_weight"])
        if "disappear_weight" in ul: w.ultrack_disappear_spin.setValue(ul["disappear_weight"])
        if "division_weight" in ul: w.ultrack_division_spin.setValue(ul["division_weight"])
        if "max_neighbors" in ul and not (dbg_present and "max_neighbors" in state["db_generation"]):
            w.db_gen_max_neighbors_spin.setValue(ul["max_neighbors"])
        if "power" in ul: w.ultrack_power_spin.setValue(ul["power"])
        if "bias" in ul: w.ultrack_bias_spin.setValue(ul["bias"])
        if "resolve_from_validated" in ul and not (dbg_present and "use_validated" in state["db_generation"]):
            w.db_gen_use_validated_check.setChecked(ul["resolve_from_validated"])
        if "quality_exponent" in ul and not (dbg_present and "quality_exponent" in state["db_generation"]):
            w.db_gen_quality_exp_spin.setValue(ul["quality_exponent"])
