from .dataset import G1HybridPriorDataset
from .dataset_amp import G1AMPDataset
from typing import Tuple


def make_dataset(cfg, device):
    """
    Factory per creare il dataset corretto.
    Accetta sia un dizionario che un oggetto config (OmegaConf/Hydra).
    """

    # Helper per estrarre valori indifferentemente dal tipo di cfg
    def get_cfg_val(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    mode = str(get_cfg_val("training_type", "ppo"))
    field_check(mode, ("ppo_amp", "ppo"))

    dataset_type = str(get_cfg_val("dataset_type", "augmented"))
    field_check(dataset_type, ("augmented", "raw"))

    lazy_load = bool(get_cfg_val("lazy_load", False))
    field_check(lazy_load, (True, False))

    vel_mode = str(get_cfg_val("vel_mode", "central"))
    field_check(vel_mode, ("central", "backward"))

    robot_name = str(get_cfg_val("robot"))
    field_check(robot_name, ("g1", "g1_amp"))

    path = str(get_cfg_val("dataset_path"))

    num_amp_obs_steps = int(get_cfg_val("num_amp_obs_steps"))

    print(f"[DataManager] Initializing dataset type: '{mode}' from {path}")

    if mode == "ppo":
        return G1HybridPriorDataset(
            file_path=path,
            robot=robot_name,
            lazy_load=lazy_load,
            vel_mode=vel_mode,
            dataset_type=dataset_type,
        )

    elif mode == "ppo_amp":
        # Il dataset veloce per AMP (Discriminator training)
        return G1AMPDataset(
            file_path=path,
            device=device,
            robot=robot_name,
            num_amp_obs_steps=num_amp_obs_steps,
        )

    else:
        raise ValueError(f"[DataManager] Unknown dataset type: {mode}")


def field_check(field, options: Tuple) -> None:
    if field not in options:
        raise ValueError(
            f"[DataManager] Unknown field: {field}. \
            Available options: {options}"
        )
