"""Train a YOLO model from a Roboflow dataset.

This script keeps the project changes isolated to a single new file.
Use it with a Roboflow dataset export in YOLO format and any valid
Ultralytics checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect.train import DetectionTrainer


load_dotenv()

ULTRALYTICS_PRETRAINED_WEIGHTS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
}


def _clean_roboflow_value(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip().strip('"').strip("'")
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered.startswith("your_") or "placeholder" in lowered:
        return None

    return cleaned


def _find_local_dataset_yaml(project: str, version: int) -> Path | None:
    candidate_names = [
        Path(f"{project}-{version}") / "data.yaml",
        Path(f"{project.replace('_', '-')}-{version}") / "data.yaml",
        Path(f"{project.replace('_', ' ').title().replace(' ', '-')}-{version}") / "data.yaml",
    ]

    for candidate in candidate_names:
        if candidate.exists():
            return candidate

    for candidate in Path.cwd().glob(f"**/*-{version}/data.yaml"):
        return candidate

    return None


def _load_yolo_model(model: str) -> YOLO:
    model_path = Path(model)

    if model_path.exists():
        try:
            return YOLO(str(model_path))
        except RuntimeError as exc:
            remote_weight = ULTRALYTICS_PRETRAINED_WEIGHTS.get(model_path.name)
            if remote_weight is None:
                raise

            print(
                f"Local checkpoint '{model_path}' is invalid. Falling back to {remote_weight}."
            )
            return YOLO(remote_weight)

    remote_weight = ULTRALYTICS_PRETRAINED_WEIGHTS.get(model)
    if remote_weight is not None:
        try:
            return YOLO(model)
        except RuntimeError:
            print(f"Falling back to remote pretrained weight: {remote_weight}")
            return YOLO(remote_weight)

    return YOLO(model)


def train_yolov26_from_roboflow(
    data_yaml: str | None = None,
    *,
    api_key: str | None = None,
    workspace: str | None = None,
    project: str | None = None,
    version: int = 1,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int | None = 4,
    device: str | int | None = 0,
    project_dir: str = "runs/detect",
    run_name: str = "loteria_yolo_26",
    save_path: str = "best_yolov26.pt",
):
    """Train YOLO on a Roboflow dataset.

    If `data_yaml` is not provided, the function downloads the dataset
    from Roboflow and uses the exported `data.yaml` path.
    """

    dataset_yaml = data_yaml

    if dataset_yaml is None:
        api_key = _clean_roboflow_value(api_key or os.getenv("ROBOFLOW_API_KEY"))
        workspace = _clean_roboflow_value(workspace or os.getenv("ROBOFLOW_WORKSPACE"))
        project = _clean_roboflow_value(project or os.getenv("ROBOFLOW_PROJECT"))

        if project:
            local_dataset_yaml = _find_local_dataset_yaml(project, version)
            if local_dataset_yaml is not None:
                dataset_yaml = str(local_dataset_yaml)

        if dataset_yaml is None and (not api_key or not workspace or not project):
            raise ValueError(
                "Provide data_yaml or set real ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and ROBOFLOW_PROJECT values. "
                "Your .env currently contains placeholder workspace/project values."
            )

        if dataset_yaml is None:
            try:
                from roboflow import Roboflow
            except ImportError as exc:
                raise ImportError(
                    "roboflow is required to download datasets. Install it with `pip install roboflow`."
                ) from exc

            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            dataset = project_obj.version(version).download("yolo26")
            dataset_yaml = str(Path(dataset.location) / "data.yaml")

    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    if workers is None:
        cpu_count = os.cpu_count() or 2
        workers = max(2, min(8, cpu_count // 2))

    model_obj = _load_yolo_model(model)
    os.environ["YOLO_VERBOSE"] = "False"

    original_logger_info = LOGGER.info

    def filtered_logger_info(message, *args, **kwargs):
        if isinstance(message, str) and "GPU_mem" in message and "Instances" in message:
            return
        if isinstance(message, str) and any(x in str(message) for x in ["box_loss", "cls_loss", "%" , "━", "─"]):
            return
        return original_logger_info(message, *args, **kwargs)

    LOGGER.info = filtered_logger_info
    try:
        train_result = model_obj.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            verbose=False,
        )
    finally:
        LOGGER.info = original_logger_info

    best_weights = Path(project_dir) / run_name / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, save_path)

    return train_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model using a Roboflow dataset export."
    )
    parser.add_argument("--data-yaml", default=None, help="Path to Roboflow data.yaml")
    parser.add_argument("--api-key", default=None, help="Roboflow API key")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace name")
    parser.add_argument("--project", default=None, help="Roboflow project name")
    parser.add_argument("--version", type=int, default=9, help="Roboflow dataset version")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base checkpoint to start from (local path or Ultralytics weight name)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader workers (auto if omitted)",
    )
    parser.add_argument("--device", default=0, help="cuda, cpu, mps, or device index")
    parser.add_argument("--project-dir", default="runs/detect")
    parser.add_argument("--run-name", default="loteria_yolo")
    parser.add_argument(
        "--save-path",
        default="best_yolov26.pt",
        help="Path to copy the trained best weights into",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_yolov26_from_roboflow(
        data_yaml=args.data_yaml,
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project_dir=args.project_dir,
        run_name=args.run_name,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()