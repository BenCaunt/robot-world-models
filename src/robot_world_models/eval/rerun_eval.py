from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import rerun as rr


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
    validated_joint_mapping: Mapping[str, str] | None = None,
) -> Path:
    if len(actual_states) != len(predicted_states):
        raise ValueError("actual and predicted state sequences must have equal length")
    if any(len(row) != len(joint_names) for row in [*actual_states, *predicted_states]):
        raise ValueError("every state row must match joint_names")
    if validated_joint_mapping is not None and set(validated_joint_mapping) != set(joint_names):
        raise ValueError("validated joint mapping must cover every dataset joint")

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            if validated_joint_mapping is None:
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
                )
                predicted_tree = rr.urdf.UrdfTree.from_file_path(
                    urdf_path,
                    entity_path_prefix="robot/predicted",
                    frame_prefix="predicted/",
                )
                actual_tree.log_urdf_to_recording(recording)
                predicted_tree.log_urdf_to_recording(recording)

        paired_states = zip(actual_states, predicted_states, strict=True)
        for step, (actual, predicted) in enumerate(paired_states):
            recording.set_time("step", sequence=step)
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
                    and validated_joint_mapping is not None
                ):
                    urdf_joint_name = validated_joint_mapping[joint_name]
                    recording.log(
                        "robot/actual/transforms",
                        actual_tree.get_joint_by_name(urdf_joint_name).compute_transform(actual[index]),
                    )
                    recording.log(
                        "robot/predicted/transforms",
                        predicted_tree.get_joint_by_name(urdf_joint_name).compute_transform(
                            predicted[index]
                        ),
                    )
    return output_path
