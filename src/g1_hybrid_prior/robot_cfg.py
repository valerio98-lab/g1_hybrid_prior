from dataclasses import dataclass
from typing import List, Dict, Any
import yaml


@dataclass
class RobotCfg:
    name: str
    fps: int
    root_fields: List[str]  # ["X","Y","Z","QX","QY","QZ","QW"]
    quaternion_order: List[str]
    joint_order: List[str]  # ordered list of joint names in the CSV

    @property
    def root_dim(self) -> int:
        return len(self.root_fields)  # 7

    @property
    def dof(self) -> int:
        return len(self.joint_order)
    
    @property
    def num_ee(self) -> int:
        return len(self.ee_link_names)

    @property
    def expected_cols(self) -> int:
        return self.root_dim + self.dof


def load_robot_cfg(yaml_path: str, robot: str) -> RobotCfg:
    with open(yaml_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    if robot not in data:
        raise ValueError(
            f"Robot {robot!r} not found in {yaml_path}. "
            f"Available: {list(data.keys())}"
        )
    entry = data[robot]
    return RobotCfg(
        name=robot,
        fps=int(entry["fps"]),
        root_fields=list(entry["root_fields"]),
        quaternion_order=list(entry["quaternion_order"]),
        joint_order=list(entry["joint_order"]),
        ee_link_names=list(entry.get("ee_link_names", [])),
    )
