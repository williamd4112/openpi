"""Compute normalization statistics for a config, optionally excluding tasks.

Like ``compute_norm_stats.py``, but supports ``--exclude-task`` (repeatable) to
drop whole tasks (by their LeRobot task string) from the statistics, and
``--output-dir`` to control where ``norm_stats.json`` is written.

Excluded tasks are dropped at the dataset level via ``LeRobotDataset(episodes=...)``
so their frames never enter the running statistics.

Example:
  HF_LEROBOT_HOME=/.../RLinf/data uv run python scripts/compute_norm_stats_filtered.py \
      pi05_fr3vla_pde_2cam \
      --exclude-task "put the green container on the bottom rack" \
      --output-dir checkpoints/torch/pi05_base_hf/jmarangola/pde_real_sft_403ep
"""

import json
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

# RLinf registers the FR3VLA / custom TrainConfigs in its own registry (not
# openpi's), so resolve config names through RLinf's get_openpi_config.
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _episode_task_strings(root: pathlib.Path) -> dict[int, set[str]]:
    """Map episode_index -> set of task strings, from the dataset meta files."""
    tasks_by_idx: dict[int, str] = {}
    tasks_path = root / "meta" / "tasks.jsonl"
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                tasks_by_idx[int(d["task_index"])] = d["task"]

    ep_tasks: dict[int, set[str]] = {}
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            ep_idx = int(e["episode_index"])
            strs: set[str] = set()
            if "tasks" in e and e["tasks"] is not None:
                # newer LeRobot format stores task strings directly
                strs.update(str(t) for t in e["tasks"])
            if "task_index" in e and e["task_index"] is not None:
                strs.add(tasks_by_idx[int(e["task_index"])])
            ep_tasks[ep_idx] = strs
    return ep_tasks


def _kept_episodes(root: pathlib.Path, exclude_tasks: list[str]) -> list[int]:
    ep_tasks = _episode_task_strings(root)
    excluded = set(exclude_tasks)
    kept = sorted(ep for ep, strs in ep_tasks.items() if not (strs & excluded))
    dropped = sorted(ep for ep, strs in ep_tasks.items() if (strs & excluded))
    print(f"[filter] total episodes: {len(ep_tasks)}")
    print(f"[filter] excluding tasks: {exclude_tasks}")
    print(f"[filter] dropped {len(dropped)} episodes, keeping {len(kept)}")
    return kept


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    exclude_tasks: list[str],
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    # Load the FULL dataset (contiguous episode indexing). NOTE: passing a
    # non-contiguous episodes=[...] subset to this LeRobot version corrupts
    # frame indexing (ep_idx out of bounds), so we subset by frame index below.
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        tolerance_s=1e6,
        # Force pyav: the default torchcodec backend fails to load libtorchcodec
        # (missing libavutil) in this env; pyav bundles its own ffmpeg.
        video_backend="pyav",
    )

    if exclude_tasks:
        kept_eps = set(_kept_episodes(pathlib.Path(dataset_meta.root), exclude_tasks))
        edi = dataset.episode_data_index
        frame_idx: list[int] = []
        for ep in sorted(kept_eps):
            frame_idx.extend(range(int(edi["from"][ep]), int(edi["to"][ep])))
        dataset = torch.utils.data.Subset(dataset, frame_idx)

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset, [transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Strings are not supported by JAX and are not needed for norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    print(f"[filter] frames in stats pass: {len(dataset)} ({num_batches} batches of {batch_size})")
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    exclude_task: list[str] = [],  # noqa: B006 — tyro fills this per-flag
    output_dir: str | None = None,
    max_frames: int | None = None,
):
    config = get_openpi_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("Filtered norm stats only supports LeRobot (torch) datasets.")

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        list(exclude_task),
        max_frames,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = pathlib.Path(output_dir) if output_dir else (config.assets_dirs / data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
