from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rerun as rr

from robot_world_models.contracts import JointTransform


def apply_joint_transform(value: float, transform: JointTransform) -> float:
    return value * transform.scale + transform.offset


def _render_value(
    *,
    value: float,
    joint: rr.urdf.UrdfJoint,
    out_of_range_policy: str,
) -> tuple[float, bool]:
    lower = joint.limit_lower
    upper = joint.limit_upper
    below = math.isfinite(lower) and value < lower
    above = math.isfinite(upper) and value > upper
    if not below and not above:
        return value, False
    if out_of_range_policy == "reject":
        raise ValueError(
            f"joint {joint.name} value {value} is outside URDF limits [{lower}, {upper}]"
        )
    if out_of_range_policy != "clamp":
        raise ValueError(f"unknown out-of-range policy: {out_of_range_policy}")
    return min(max(value, lower), upper), True


def _tint_tree(
    recording: rr.RecordingStream,
    tree: rr.urdf.UrdfTree,
    color: list[int],
) -> None:
    links = {tree.root_link().name, *(joint.child_link for joint in tree.joints())}
    for link in sorted(links):
        for path in tree.get_visual_geometry_paths(link):
            recording.log(
                path,
                rr.Asset3D.from_fields(albedo_factor=color),
                static=True,
            )


def write_state_evaluation(
    *,
    output_path: Path,
    run_id: str,
    joint_names: Sequence[str],
    actual_states: Sequence[Sequence[float]],
    predicted_states: Sequence[Sequence[float]],
    metrics: Mapping[str, float],
    provenance: Mapping[str, object],
    urdf_path: Path | None = None,
    joint_mapping: Mapping[str, JointTransform] | None = None,
    unmapped_features: Sequence[str] = (),
    out_of_range_policy: str = "reject",
    actual_images: Sequence[np.ndarray] | None = None,
    predicted_images: Sequence[np.ndarray] | None = None,
) -> tuple[Path, dict[str, object]]:
    if len(actual_states) != len(predicted_states):
        raise ValueError("actual and predicted state sequences must have equal length")
    if any(len(row) != len(joint_names) for row in [*actual_states, *predicted_states]):
        raise ValueError("every state row must match joint_names")
    if (actual_images is None) != (predicted_images is None):
        raise ValueError("actual_images and predicted_images must be provided together")
    if actual_images is not None and predicted_images is not None and (
        len(actual_images) != len(actual_states)
        or len(predicted_images) != len(predicted_states)
    ):
        raise ValueError("image and state sequences must have equal length")
    if joint_mapping is not None:
        mapped = set(joint_mapping)
        named = set(joint_names)
        if not mapped or not mapped <= named:
            raise ValueError("joint mapping must contain known dataset joints")
        if set(unmapped_features) != named - mapped:
            raise ValueError("unmapped_features must name every dataset joint omitted from the map")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation: dict[str, object] = {
        "enabled": False,
        "mappedFeatures": [],
        "unmappedFeatures": list(unmapped_features),
        "outOfRangePolicy": out_of_range_policy,
        "actualClampedValues": 0,
        "predictedClampedValues": 0,
    }
    with rr.RecordingStream(
        "robot_world_models_evaluation",
        recording_id=run_id,
        send_properties=False,
    ) as recording:
        recording.save(output_path)
        recording.log(
            "receipt/provenance",
            rr.TextDocument(json.dumps(provenance, indent=2, sort_keys=True)),
            static=True,
        )
        recording.log(
            "receipt/metrics",
            rr.TextDocument(json.dumps(dict(metrics), indent=2, sort_keys=True)),
            static=True,
        )

        actual_tree = None
        predicted_tree = None
        if urdf_path is not None:
            if joint_mapping is None:
                recording.log(
                    "receipt/joint_mapping_warning",
                    rr.TextLog(
                        "URDF is present, but animation is disabled until joint names, units, "
                        "offsets, signs, and limits are validated."
                    ),
                    static=True,
                )
                recording.log_file_from_path(urdf_path, static=True)
            else:
                actual_tree = rr.urdf.UrdfTree.from_file_path(
                    urdf_path,
                    entity_path_prefix="robot/actual",
                    frame_prefix="actual/",
                    static_transform_entity_path="robot/actual/tf_static",
                )
                predicted_tree = rr.urdf.UrdfTree.from_file_path(
                    urdf_path,
                    entity_path_prefix="robot/predicted",
                    frame_prefix="predicted/",
                    static_transform_entity_path="robot/predicted/tf_static",
                )
                actual_tree.log_urdf_to_recording(recording)
                predicted_tree.log_urdf_to_recording(recording)
                _tint_tree(recording, actual_tree, [70, 150, 255, 255])
                _tint_tree(recording, predicted_tree, [255, 145, 55, 255])
                recording.log(
                    "robot/actual/root",
                    rr.Transform3D(
                        translation=[0.0, -0.22, 0.0],
                        parent_frame="world",
                        child_frame=f"actual/{actual_tree.root_link().name}",
                    ),
                    static=True,
                )
                recording.log(
                    "robot/predicted/root",
                    rr.Transform3D(
                        translation=[0.0, 0.22, 0.0],
                        parent_frame="world",
                        child_frame=f"predicted/{predicted_tree.root_link().name}",
                    ),
                    static=True,
                )
                animation["enabled"] = True
                animation["mappedFeatures"] = list(joint_mapping)
                if unmapped_features:
                    recording.log(
                        "receipt/joint_mapping_warning",
                        rr.TextLog(
                            "Partial URDF animation: "
                            f"{', '.join(joint_mapping)} are animated; "
                            f"{', '.join(unmapped_features)} remain at the URDF default."
                        ),
                        static=True,
                    )

        paired_states = zip(actual_states, predicted_states, strict=True)
        for step, (actual, predicted) in enumerate(paired_states):
            recording.set_time("step", sequence=step)
            if actual_images is not None and predicted_images is not None:
                actual_image = np.asarray(actual_images[step], dtype=np.uint8)
                predicted_image = np.asarray(predicted_images[step], dtype=np.uint8)
                if actual_image.shape != predicted_image.shape:
                    raise ValueError("actual and predicted image shapes must match")
                error_image = np.abs(
                    actual_image.astype(np.int16) - predicted_image.astype(np.int16)
                ).astype(np.uint8)
                recording.log("vision/actual", rr.Image(actual_image))
                recording.log("vision/predicted", rr.Image(predicted_image))
                recording.log("vision/absolute_error", rr.Image(error_image))
            for index, joint_name in enumerate(joint_names):
                recording.log(f"state/actual/{joint_name}", rr.Scalars(actual[index]))
                recording.log(f"state/predicted/{joint_name}", rr.Scalars(predicted[index]))
                recording.log(
                    f"state/error/{joint_name}",
                    rr.Scalars(predicted[index] - actual[index]),
                )
                if (
                    actual_tree is not None
                    and predicted_tree is not None
                    and joint_mapping is not None
                    and joint_name in joint_mapping
                ):
                    mapping = joint_mapping[joint_name]
                    actual_joint = actual_tree.get_joint_by_name(mapping.urdf_joint)
                    predicted_joint = predicted_tree.get_joint_by_name(mapping.urdf_joint)
                    if actual_joint is None or predicted_joint is None:
                        raise ValueError(f"URDF joint does not exist: {mapping.urdf_joint}")
                    actual_value = apply_joint_transform(actual[index], mapping)
                    predicted_value = apply_joint_transform(predicted[index], mapping)
                    actual_render_value, actual_clamped = _render_value(
                        value=actual_value,
                        joint=actual_joint,
                        out_of_range_policy=out_of_range_policy,
                    )
                    predicted_render_value, predicted_clamped = _render_value(
                        value=predicted_value,
                        joint=predicted_joint,
                        out_of_range_policy=out_of_range_policy,
                    )
                    animation["actualClampedValues"] = int(animation["actualClampedValues"]) + int(
                        actual_clamped
                    )
                    animation["predictedClampedValues"] = int(
                        animation["predictedClampedValues"]
                    ) + int(predicted_clamped)
                    recording.log(
                        f"joint_position/actual_radians/{joint_name}",
                        rr.Scalars(actual_value),
                    )
                    recording.log(
                        f"joint_position/predicted_radians/{joint_name}",
                        rr.Scalars(predicted_value),
                    )
                    recording.log(
                        f"joint_position/limit_clamped/actual/{joint_name}",
                        rr.Scalars(int(actual_clamped)),
                    )
                    recording.log(
                        f"joint_position/limit_clamped/predicted/{joint_name}",
                        rr.Scalars(int(predicted_clamped)),
                    )
                    recording.log(
                        "robot/actual/transforms",
                        actual_joint.compute_transform(actual_render_value, clamp=False),
                    )
                    recording.log(
                        "robot/predicted/transforms",
                        predicted_joint.compute_transform(predicted_render_value, clamp=False),
                    )
        mapped_value_count = len(actual_states) * len(joint_mapping or {})
        if mapped_value_count:
            animation["actualLimitViolationRate"] = (
                int(animation["actualClampedValues"]) / mapped_value_count
            )
            animation["predictedLimitViolationRate"] = (
                int(animation["predictedClampedValues"]) / mapped_value_count
            )
        recording.log(
            "receipt/joint_animation",
            rr.TextDocument(json.dumps(animation, indent=2, sort_keys=True)),
            static=True,
        )
    return output_path, animation
