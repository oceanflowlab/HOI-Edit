#!/usr/bin/env python3
"""
本地 Wan2.2 I2V A14B 批量生成（参考 https://github.com/Wan-Video/Wan2.2 官方 generate.py）。

与 wan22_generate_from_enhanced_prompts.py 相同输入（enhanced_prompts JSON + 原图），
但调用本地 checkpoint + WanI2V 管线，而非 DashScope API。

依赖:
  - 已 clone Wan2.2 仓库（--wan-repo，默认仓库根目录 ../Wan2.2）
  - 已下载 Wan2.2-I2V-A14B 权重（--ckpt-dir，默认 ../Wan2.2-I2V-A14B）
  - CUDA GPU 推荐单卡 + offload_model；Ascend NPU 请使用本机已适配的 Wan2.2 fork

模式:
  inprocess  — 一次加载模型，循环生成（推荐，CUDA / 已适配 NPU fork）
  subprocess — 每条样本调用 Wan2.2/generate.py（兼容复杂 NPU 启动参数，但很慢）

示例:
  python wan22_local_i2v_a14b_generate.py \\
    --enhanced-prompts-json ../output/enhanced_prompts.json \\
    --image-dir-l1l2 /path/to/data_v7_L12 \\
    --output-dir-l1l2 ../output/wan22_local_a14b/L1L2 \\
    --split L1L2 --limit 2

  # NPU 多卡 subprocess（与 start.sh / batch_generate_L1L2.py 类似）
  WAN22_LAUNCHER=torchrun WAN22_NPROC=8 \\
    python wan22_local_i2v_a14b_generate.py --mode subprocess ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASIC_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BASIC_ROOT.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wan22_prompt_tasks import (
    build_final_prompt,
    has_existing_video,
    load_enhanced_tasks,
    tasks_for_split,
)


def default_wan_repo() -> Path:
    env = os.getenv("WAN22_REPO", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "Wan2.2").resolve()


def default_ckpt_dir() -> Path:
    env = os.getenv("WAN22_CKPT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "Wan2.2-I2V-A14B").resolve()


def resolve_device_id() -> int:
    return int(os.getenv("WAN22_DEVICE_ID", os.getenv("LOCAL_RANK", "0")))


def detect_device_kind() -> str:
    forced = os.getenv("WAN22_DEVICE", "").strip().lower()
    if forced in ("cuda", "npu", "cpu"):
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import torch_npu  # noqa: F401

        return "npu"
    except ImportError:
        return "cuda"


class Wan22LocalI2VPipeline:
    """单次加载 WanI2V，批量生成（对齐官方 generate.py i2v-A14B 路径）。"""

    def __init__(
        self,
        wan_repo: Path,
        ckpt_dir: Path,
        size: str,
        frame_num: Optional[int],
        sample_steps: Optional[int],
        sample_shift: Optional[float],
        sample_guide_scale: Optional[float],
        sample_solver: str,
        offload_model: bool,
        t5_cpu: bool,
        convert_model_dtype: bool,
        device_id: int,
    ) -> None:
        self.wan_repo = wan_repo
        self.ckpt_dir = ckpt_dir
        self.size = size
        self.frame_num = frame_num
        self.sample_steps = sample_steps
        self.sample_shift = sample_shift
        self.sample_guide_scale = sample_guide_scale
        self.sample_solver = sample_solver
        self.offload_model = offload_model
        self.t5_cpu = t5_cpu
        self.convert_model_dtype = convert_model_dtype
        self.device_id = device_id
        self.device_kind = detect_device_kind()

        self.cfg: Any = None
        self.wan_i2v: Any = None
        self.max_area: int = 0

    def _ensure_wan_import(self) -> None:
        repo = str(self.wan_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def load(self) -> None:
        if not self.wan_repo.is_dir():
            raise FileNotFoundError(f"Wan2.2 仓库不存在: {self.wan_repo}")
        if not self.ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint 目录不存在: {self.ckpt_dir}")

        self._ensure_wan_import()

        import torch
        from PIL import Image  # noqa: F401 — 供 generate 使用

        import wan
        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

        if self.device_kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA，请设置 WAN22_DEVICE=npu 或使用 --mode subprocess")

        cfg = WAN_CONFIGS["i2v-A14B"]
        self.cfg = cfg
        self.max_area = MAX_AREA_CONFIGS[self.size]

        if self.frame_num is None:
            self.frame_num = cfg.frame_num
        if self.sample_steps is None:
            self.sample_steps = cfg.sample_steps
        if self.sample_shift is None:
            self.sample_shift = cfg.sample_shift
        if self.sample_guide_scale is None:
            self.sample_guide_scale = cfg.sample_guide_scale

        logging.info(
            "加载 WanI2V A14B | repo=%s ckpt=%s device=%s:%d",
            self.wan_repo,
            self.ckpt_dir,
            self.device_kind,
            self.device_id,
        )

        self.wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=str(self.ckpt_dir),
            device_id=self.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=self.t5_cpu,
            convert_model_dtype=self.convert_model_dtype,
        )

        if self.device_kind == "npu":
            dev = torch.device(f"npu:{self.device_id}")
            self.wan_i2v.low_noise_model.to(dev)
            self.wan_i2v.high_noise_model.to(dev)
            if hasattr(self.wan_i2v, "device"):
                self.wan_i2v.device = dev

    def generate_one(
        self,
        image_path: Path,
        prompt: str,
        save_path: Path,
        seed: int,
    ) -> bool:
        from PIL import Image
        from wan.utils.utils import save_video

        if self.wan_i2v is None or self.cfg is None:
            raise RuntimeError("请先调用 load()")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(image_path).convert("RGB")
        video = self.wan_i2v.generate(
            prompt,
            img,
            max_area=self.max_area,
            frame_num=self.frame_num,
            shift=self.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.sample_steps,
            guide_scale=self.sample_guide_scale,
            seed=seed,
            offload_model=self.offload_model,
        )
        if video is None:
            return False

        save_video(
            tensor=video[None],
            save_file=str(save_path),
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return save_path.is_file()


def run_subprocess_one(
    *,
    wan_repo: Path,
    ckpt_dir: Path,
    image_path: Path,
    prompt: str,
    save_path: Path,
    size: str,
    frame_num: int,
    sample_steps: int,
    sample_shift: float,
    sample_guide_scale: float,
    sample_solver: str,
    base_seed: int,
    extra_args: List[str],
    launcher: str,
    nproc: int,
) -> Tuple[bool, str]:
    generate_py = wan_repo / "generate.py"
    if not generate_py.is_file():
        return False, f"未找到 {generate_py}"

    cmd: List[str]
    if launcher == "torchrun":
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(nproc),
            str(generate_py),
        ]
    else:
        cmd = [sys.executable, str(generate_py)]

    cmd.extend(
        [
            "--task",
            "i2v-A14B",
            "--ckpt_dir",
            str(ckpt_dir),
            "--size",
            size,
            "--frame_num",
            str(frame_num),
            "--sample_steps",
            str(sample_steps),
            "--sample_solver",
            sample_solver,
            "--sample_shift",
            str(sample_shift),
            "--sample_guide_scale",
            str(sample_guide_scale),
            "--image",
            str(image_path),
            "--prompt",
            prompt,
            "--save_file",
            str(save_path),
            "--base_seed",
            str(base_seed),
        ]
    )
    cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(wan_repo),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            return False, f"exit {proc.returncode}: {tail}"
        if not save_path.is_file():
            return False, "generate.py 完成但 save_file 不存在"
        return True, str(save_path)
    except Exception as e:
        return False, str(e)


def run_batch(
    *,
    name: str,
    tasks: List[Dict[str, str]],
    image_dir: Path,
    output_dir: Path,
    pipeline: Optional[Wan22LocalI2VPipeline],
    wan_repo: Path,
    ckpt_dir: Path,
    mode: str,
    skip_if_exists: bool,
    skip_first: int,
    limit: Optional[int],
    sleep_seconds: float,
    base_seed: int,
    subprocess_extra: List[str],
    launcher: str,
    nproc: int,
    frame_num: int,
    sample_steps: int,
    sample_shift: float,
    sample_guide_scale: float,
    sample_solver: str,
    size: str,
    manifest_path: Optional[Path],
) -> Dict[str, Any]:
    if skip_first:
        tasks = tasks[skip_first:]
    if limit is not None:
        tasks = tasks[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "name": name,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "total": len(tasks),
        "generated": 0,
        "skipped": 0,
        "failed": 0,
    }

    mf = open(manifest_path, "a", encoding="utf-8") if manifest_path else None
    print(f"\n=== {name}: {len(tasks)} tasks | mode={mode} -> {output_dir} ===")

    try:
        for idx, task in enumerate(tqdm(tasks, desc=name)):
            image_name = task["image_name"]
            image_path = image_dir / image_name
            stem = Path(image_name).stem
            out_path = output_dir / f"{stem}_local_a14b.mp4"

            if not image_path.exists():
                summary["failed"] += 1
                tqdm.write(f"  ✗ missing image: {image_name}")
                continue

            if skip_if_exists and (has_existing_video(image_name, output_dir) or out_path.is_file()):
                summary["skipped"] += 1
                continue

            prompt = build_final_prompt(task["enhanced_prompt"])
            seed = base_seed + idx

            if mode == "inprocess" and pipeline is not None:
                ok = pipeline.generate_one(image_path, prompt, out_path, seed)
                msg = str(out_path) if ok else "inprocess generate failed"
            else:
                ok, msg = run_subprocess_one(
                    wan_repo=wan_repo,
                    ckpt_dir=ckpt_dir,
                    image_path=image_path,
                    prompt=prompt,
                    save_path=out_path,
                    size=size,
                    frame_num=frame_num,
                    sample_steps=sample_steps,
                    sample_shift=sample_shift,
                    sample_guide_scale=sample_guide_scale,
                    sample_solver=sample_solver,
                    base_seed=seed,
                    extra_args=subprocess_extra,
                    launcher=launcher,
                    nproc=nproc,
                )

            if ok:
                summary["generated"] += 1
                if mf:
                    mf.write(
                        json.dumps(
                            {
                                "split": task.get("split"),
                                "image_name": image_name,
                                "instruction": task.get("instruction"),
                                "enhanced_prompt": task["enhanced_prompt"],
                                "final_prompt": prompt,
                                "video_path": msg,
                                "seed": seed,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    mf.flush()
            else:
                summary["failed"] += 1
                tqdm.write(f"  ✗ {image_name}: {msg}")

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if mf:
            mf.close()

    print(
        f"[{name}] generated={summary['generated']} "
        f"skipped={summary['skipped']} failed={summary['failed']}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local Wan2.2 I2V A14B batch generation from enhanced_prompts JSON."
    )
    p.add_argument(
        "--enhanced-prompts-json",
        type=Path,
        default=BASIC_ROOT / "output/enhanced_prompts.json",
    )
    p.add_argument("--image-dir-l1l2", type=Path, default=None)
    p.add_argument("--image-dir-l3", type=Path, default=None)
    p.add_argument(
        "--output-dir-l1l2",
        type=Path,
        default=BASIC_ROOT / "output/wan22_local_a14b/L1L2",
    )
    p.add_argument(
        "--output-dir-l3",
        type=Path,
        default=BASIC_ROOT / "output/wan22_local_a14b/L3",
    )
    p.add_argument("--split", choices=["all", "L1L2", "L3"], default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-first", type=int, default=0)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--base-seed", type=int, default=0)

    p.add_argument("--wan-repo", type=Path, default=None, help="Wan2.2 源码目录")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="Wan2.2-I2V-A14B 权重目录")
    p.add_argument(
        "--mode",
        choices=["inprocess", "subprocess"],
        default=os.getenv("WAN22_RUN_MODE", "inprocess"),
    )
    p.add_argument("--size", default=os.getenv("WAN22_SIZE", "1280*720"))
    p.add_argument("--frame-num", type=int, default=None)
    p.add_argument("--sample-steps", type=int, default=None)
    p.add_argument("--sample-shift", type=float, default=None)
    p.add_argument("--sample-guide-scale", type=float, default=None)
    p.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    p.add_argument("--offload-model", action="store_true", default=True)
    p.add_argument("--no-offload-model", action="store_false", dest="offload_model")
    p.add_argument("--t5-cpu", action="store_true", default=False)
    p.add_argument("--convert-model-dtype", action="store_true", default=False)
    p.add_argument(
        "--subprocess-extra",
        default=os.getenv("WAN22_SUBPROCESS_EXTRA", ""),
        help="传给 generate.py 的额外参数，如 '--dit_fsdp --t5_fsdp --cfg_size 2'",
    )
    p.add_argument(
        "--launcher",
        default=os.getenv("WAN22_LAUNCHER", "python"),
        choices=["python", "torchrun"],
    )
    p.add_argument("--nproc", type=int, default=int(os.getenv("WAN22_NPROC", "1")))
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument(
        "--summary-json",
        type=Path,
        default=BASIC_ROOT / "output/wan22_local_a14b/run_summary.json",
    )
    p.add_argument("--manifest", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    args = parse_args()

    wan_repo = (args.wan_repo or default_wan_repo()).expanduser().resolve()
    ckpt_dir = (args.ckpt_dir or default_ckpt_dir()).expanduser().resolve()
    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data"))).expanduser().resolve()

    image_l1l2 = (
        args.image_dir_l1l2.expanduser().resolve()
        if args.image_dir_l1l2
        else data_root / "data_v7_L12"
    )
    image_l3 = (
        args.image_dir_l3.expanduser().resolve()
        if args.image_dir_l3
        else data_root / "data_v7_L3"
    )

    ep_path = args.enhanced_prompts_json.expanduser().resolve()
    all_tasks = load_enhanced_tasks(ep_path)
    skip_if_exists = not args.no_skip_existing

    subprocess_extra = args.subprocess_extra.split() if args.subprocess_extra.strip() else []

    pipeline: Optional[Wan22LocalI2VPipeline] = None
    frame_num = args.frame_num or 81
    sample_steps = args.sample_steps or 40
    sample_shift = args.sample_shift if args.sample_shift is not None else 5.0
    sample_guide_scale = (
        args.sample_guide_scale if args.sample_guide_scale is not None else 3.5
    )

    if args.mode == "inprocess":
        pipeline = Wan22LocalI2VPipeline(
            wan_repo=wan_repo,
            ckpt_dir=ckpt_dir,
            size=args.size,
            frame_num=args.frame_num,
            sample_steps=args.sample_steps,
            sample_shift=args.sample_shift,
            sample_guide_scale=args.sample_guide_scale,
            sample_solver=args.sample_solver,
            offload_model=args.offload_model,
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
            device_id=resolve_device_id(),
        )
        pipeline.load()
        frame_num = pipeline.frame_num
        sample_steps = pipeline.sample_steps
        sample_shift = pipeline.sample_shift
        if isinstance(pipeline.sample_guide_scale, tuple):
            sample_guide_scale = pipeline.sample_guide_scale[0]
        else:
            sample_guide_scale = float(pipeline.sample_guide_scale)

    summaries: Dict[str, Any] = {
        "backend": "wan2.2-local-i2v-a14b",
        "wan_repo": str(wan_repo),
        "ckpt_dir": str(ckpt_dir),
        "mode": args.mode,
        "source_json": str(ep_path),
        "splits": {},
    }

    if args.split in ("all", "L1L2"):
        manifest = args.manifest or (args.output_dir_l1l2 / "manifest.jsonl")
        summaries["splits"]["L1L2"] = run_batch(
            name="L1L2",
            tasks=tasks_for_split(all_tasks, "L1L2"),
            image_dir=image_l1l2,
            output_dir=args.output_dir_l1l2.expanduser().resolve(),
            pipeline=pipeline,
            wan_repo=wan_repo,
            ckpt_dir=ckpt_dir,
            mode=args.mode,
            skip_if_exists=skip_if_exists,
            skip_first=args.skip_first,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            base_seed=args.base_seed,
            subprocess_extra=subprocess_extra,
            launcher=args.launcher,
            nproc=args.nproc,
            frame_num=frame_num,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_solver=args.sample_solver,
            size=args.size,
            manifest_path=manifest,
        )

    if args.split in ("all", "L3"):
        manifest = args.manifest or (args.output_dir_l3 / "manifest.jsonl")
        summaries["splits"]["L3"] = run_batch(
            name="L3",
            tasks=tasks_for_split(all_tasks, "L3"),
            image_dir=image_l3,
            output_dir=args.output_dir_l3.expanduser().resolve(),
            pipeline=pipeline,
            wan_repo=wan_repo,
            ckpt_dir=ckpt_dir,
            mode=args.mode,
            skip_if_exists=skip_if_exists,
            skip_first=args.skip_first if args.split == "L3" else 0,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
            base_seed=args.base_seed,
            subprocess_extra=subprocess_extra,
            launcher=args.launcher,
            nproc=args.nproc,
            frame_num=frame_num,
            sample_steps=sample_steps,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_solver=args.sample_solver,
            size=args.size,
            manifest_path=manifest,
        )

    totals = {"generated": 0, "skipped": 0, "failed": 0}
    for s in summaries["splits"].values():
        for k in totals:
            totals[k] += s.get(k, 0)
    summaries["total"] = totals

    summary_path = args.summary_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n=== LOCAL WAN2.2 A14B DONE ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
