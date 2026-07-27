from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from robot_world_models.contracts import JointTransform
from robot_world_models.eval.rerun_eval import apply_joint_transform, write_state_evaluation


def test_legacy_lerobot_joint_conversions() -> None:
    shoulder_lift = JointTransform(
        urdf_joint="2",
        scale=-math.pi / 180,
        offset=math.pi / 2,
        evidence=["LeRobot backward compatibility"],
    )
    elbow_flex = JointTransform(
        urdf_joint="3",
        scale=math.pi / 180,
        offset=-math.pi / 2,
        evidence=["LeRobot backward compatibility"],
    )

    assert apply_joint_transform(90, shoulder_lift) == 0
    assert apply_joint_transform(0, shoulder_lift) == math.pi / 2
    assert apply_joint_transform(90, elbow_flex) == 0


def test_rerun_recording_contains_time_varying_robot_transforms(tmp_path: Path) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """\
<robot name="test_robot">
  <link name="base">
    <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
  </link>
  <link name="arm">
    <visual><geometry><box size="0.3 0.05 0.05"/></geometry></visual>
  </link>
  <joint name="1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
</robot>
"""
    )
    transform = JointTransform(
        urdf_joint="1",
        scale=math.pi / 180,
        offset=0,
        evidence=["test fixture"],
    )

    output_path, animation = write_state_evaluation(
        output_path=tmp_path / "evaluation.rrd",
        run_id="test-animation",
        joint_names=["legacy_joint", "gripper"],
        actual_states=[[0, 10], [90, 20]],
        predicted_states=[[0, 10], [45, 20]],
        metrics={"mse": 0.1},
        provenance={"test": True},
        urdf_path=urdf_path,
        joint_mapping={"legacy_joint": transform},
        unmapped_features=["gripper"],
        out_of_range_policy="clamp",
        actual_images=[
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.ones((16, 16, 3), dtype=np.uint8),
        ],
        predicted_images=[
            np.ones((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ],
    )

    assert animation["enabled"] is True
    assert animation["mappedFeatures"] == ["legacy_joint"]
    assert animation["unmappedFeatures"] == ["gripper"]
    assert animation["actualClampedValues"] == 0

    rerun = shutil.which("rerun")
    assert rerun is not None
    stats = subprocess.run(
        [rerun, "rrd", "stats", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "/robot/actual/transforms: 1" in stats
    assert "/robot/predicted/transforms: 1" in stats
    assert "/receipt/joint_animation: 1" in stats
    assert "/vision/actual: 1" in stats
    assert "/vision/predicted: 1" in stats

    printed = subprocess.run(
        [rerun, "rrd", "print", str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert re.search(r"with 2 rows .* - /robot/actual/transforms", printed)
    assert re.search(r"with 2 rows .* - /robot/predicted/transforms", printed)
