import sys
import time
import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.detector import *
from src.train_eval import *
from src.kitti_bdd100k import *
from src.Lhawk import *
from src.color_stripe.trigger_generation import generate_trigger_tensor
from src.color_stripe.laser_trigger import (
    LaserCalibration,
    LaserParams,
    build_laser_param_grid,
    generate_laser_trigger_tensor,
    parse_float_spec,
)
from utils.parser import ConfigParser, logger
from src.patch_train import train, eval, score_trigger_loss
from src.classifier import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # program root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

time_str = time.strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='./configs/TA-C.yaml',
                    help='HA, CA, TA-D, TA-C')
parser.add_argument('--exp_dir', type=str, default="exp")
parser.add_argument('--attack_type', type=str, default=None)
parser.add_argument('--target', type=str, default=None)
parser.add_argument('--origin', type=str, default="stop sign")
parser.add_argument('--det', type=str, default=None)
parser.add_argument('--eval-det', type=str, default=None,
                    help="Optional separate model used only for evaluation/transfer tests.")
parser.add_argument('--run-tag', type=str, default=None,
                    help="Optional label appended to the output directory for experiment tracking.")
parser.add_argument('--experiment-name', type=str, default=None,
                    help="Optional manifest experiment name recorded in run_config.json.")
parser.add_argument('--profile', type=str, default=None,
                    help="Optional manifest profile recorded in run_config.json.")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--train-batch', type=int, default=None,
                    help="Optional max train batches per epoch for debugging. Defaults to full dataloader epoch.")
parser.add_argument('--eval-batch', type=int, default=None)
parser.add_argument('--repeat', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None,
                    help="Training dataloader batch size. Overrides DETECTOR.BATCH_SIZE.")
parser.add_argument('--eval-dataset', choices=("kitti", "bdd100k", "coco"), default="coco")
parser.add_argument('--content-pretrained', action="store_true",
                    help='Use torchvision pretrained VGG19 for content loss; may require cached/downloaded weights.')
parser.add_argument('--trigger-source', choices=("none", "fixed", "laser"), default="fixed")
parser.add_argument('--laser-model', choices=("linear", "sigmoid", "gaussian"), default="linear")
parser.add_argument('--laser-color', choices=("green", "red", "white"), default="green")
parser.add_argument('--laser-power', default="29",
                    help="Laser power grid in mW. Supports '29', '10,30,70', or '10:70:10'.")
parser.add_argument('--laser-distance', default="30",
                    help="Attack distance grid in meters.")
parser.add_argument('--laser-angle', default="18",
                    help="Incidence angle grid in degrees.")
parser.add_argument('--ambient-light', default="1000",
                    help="Ambient light grid in Lux.")
parser.add_argument('--trigger-height', type=int, default=None,
                    help="Default synthetic trigger row thickness. Defaults to 50 for laser triggers.")
parser.add_argument('--trigger-width', type=int, default=None,
                    help="Synthetic trigger row thickness for width sweeps. Defaults to --trigger-height.")
parser.add_argument('--trigger-position', type=float, default=0.5,
                    help="Vertical stripe position in [0,1], where 0 is top and 1 is bottom.")
parser.add_argument('--trigger-noise-std', type=float, default=0.0,
                    help="Max probability of lens-imperfection snowflake noise on the trigger.")
parser.add_argument('--laser-k1', type=float, default=12.0)
parser.add_argument('--laser-k2', type=float, default=2.0e-5)
parser.add_argument('--laser-k3', type=float, default=4.0)
parser.add_argument('--laser-k4', type=float, default=1.0e-5)
parser.add_argument('--trigger-selection', choices=("random", "epoch-search", "async-joint"), default="random")
parser.add_argument('--trigger-search-objective', choices=("loss", "metric"), default=None,
                    help="Trigger candidate objective. Defaults to loss for async-joint and metric for epoch-search.")
parser.add_argument('--trigger-search-metric', choices=("ASR", "Triggered", "No_triggered"), default="ASR")
parser.add_argument('--trigger-search-batch', type=int, default=None,
                    help="Number of eval samples used to score each trigger. Defaults to one eval dataloader batch.")
parser.add_argument('--trigger-param-samples', type=int, default=32,
                    help="K: number of laser parameter tuples randomly sampled for each async-joint trigger space.")
parser.add_argument('--async-trigger-zeta', type=float, default=1.0,
                    help="Weight of triggered attack loss in async-joint trigger optimization.")
parser.add_argument('--async-trigger-psi', type=float, default=None,
                    help="Weight of benign loss in async-joint trigger optimization. Defaults to ATTACKER.ALPHA.")
parser.add_argument('--async-power-radius', type=float, default=10.0,
                    help="Power search radius around the best trigger for async-joint.")
parser.add_argument('--async-distance-radius', type=float, default=5.0,
                    help="Distance search radius around the best trigger for async-joint.")
parser.add_argument('--async-angle-radius', type=float, default=5.0,
                    help="Incidence-angle search radius around the best trigger for async-joint.")
parser.add_argument('--async-light-radius', type=float, default=200.0,
                    help="Ambient-light search radius around the best trigger for async-joint.")
parser.add_argument('--async-min-power', type=float, default=10.0)
parser.add_argument('--async-max-power', type=float, default=70.0)
parser.add_argument('--async-min-distance', type=float, default=10.0)
parser.add_argument('--async-max-distance', type=float, default=50.0)
parser.add_argument('--async-min-angle', type=float, default=0.0)
parser.add_argument('--async-max-angle', type=float, default=30.0)
parser.add_argument('--async-min-light', type=float, default=0.0)
parser.add_argument('--async-max-light', type=float, default=2000.0)
parser.add_argument('--patch-size', type=int, default=None,
                    help="Override the attack patch size with a square patch.")
parser.add_argument('--patch-top', type=int, default=None,
                    help="Fix patch top coordinate for position sweeps.")
parser.add_argument('--patch-left', type=int, default=None,
                    help="Fix patch left coordinate for position sweeps.")
parser.add_argument('--swanlab', action='store_true',
                    help="Enable SwanLab experiment tracking.")
parser.add_argument('--swanlab-project', default='l-hawk',
                    help="SwanLab project name.")
parser.add_argument('--swanlab-workspace', default=None,
                    help="Optional SwanLab workspace name.")
parser.add_argument('--swanlab-mode', choices=('online', 'local', 'offline'), default='online',
                    help="SwanLab run mode.")
parser.add_argument('--swanlab-log-dir', default='swanlog',
                    help="Directory used for SwanLab local run data.")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain(value):
    if hasattr(value, "__dict__"):
        return {k: to_plain(v) for k, v in value.__dict__.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def write_run_metadata(save_path, cfg, args, time_str):
    metadata = {
        "time": time_str,
        "args": vars(args),
        "config": {
            "DATA": to_plain(cfg.DATA),
            "DETECTOR": to_plain(cfg.DETECTOR),
            "ATTACKER": to_plain(cfg.ATTACKER),
            "EVAL": to_plain(cfg.EVAL),
            "target_index": cfg.target_index,
            "origin_index": cfg.origin_index,
        },
    }
    with open(os.path.join(save_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def append_metrics(save_path, metrics):
    path = os.path.join(save_path, "metrics.csv")
    fieldnames = [
        "epoch", "samples", "success", "no_triggered_success",
        "triggered_success", "ASR", "No_triggered", "Triggered",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(metrics)


def append_trigger_selection(save_path, row):
    path = os.path.join(save_path, "trigger_selection.csv")
    fieldnames = [
        "epoch", "phase", "loss_value",
        "p_mw", "d_m", "theta_deg", "l_lux",
        "async_epoch",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def async_joint_config_row(args, cfg):
    psi = cfg.ATTACKER.ALPHA if args.async_trigger_psi is None else args.async_trigger_psi
    return {
        "trigger_space_size": args.trigger_param_samples,
        "trigger_search_batch": args.trigger_search_batch,
        "trigger_selection_objective": args.trigger_search_objective,
        "async_trigger_zeta": args.async_trigger_zeta,
        "async_trigger_psi": psi,
        "power_range": f"[{args.async_min_power}, {args.async_max_power}]",
        "distance_range": f"[{args.async_min_distance}, {args.async_max_distance}]",
        "theta_range": f"[{args.async_min_angle}, {args.async_max_angle}]",
        "light_range": f"[{args.async_min_light}, {args.async_max_light}]",
        "power_radius": args.async_power_radius,
        "distance_radius": args.async_distance_radius,
        "angle_radius": args.async_angle_radius,
        "light_radius": args.async_light_radius,
    }


def write_async_joint_config(save_path, row):
    path = os.path.join(save_path, "async_joint_config.csv")
    fieldnames = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_trigger_candidates(save_path, rows):
    if not rows:
        return
    with open(os.path.join(save_path, "trigger_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def init_swanlab(save_path, cfg, args):
    if not args.swanlab:
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise SystemExit(
            "SwanLab tracking requested but the package is not installed. "
            "Install it with `pip install swanlab`."
        ) from exc

    init_kwargs = {
        "project": args.swanlab_project,
        "name": os.path.basename(save_path),
        "config": {
            "args": vars(args),
            "attack": to_plain(cfg.ATTACKER),
            "model": to_plain(cfg.DETECTOR),
            "eval": to_plain(cfg.EVAL),
        },
        "mode": args.swanlab_mode,
        "log_dir": args.swanlab_log_dir,
        "tags": [str(cfg.ATTACKER.TYPE), slugify(cfg.DETECTOR.NAME)],
    }
    if args.swanlab_workspace:
        init_kwargs["workspace"] = args.swanlab_workspace
    return swanlab.init(**init_kwargs)


def media_epochs(total_epochs, limit=10):
    count = min(limit, total_epochs)
    if count <= 0:
        return set()
    return set(np.linspace(1, total_epochs, num=count, dtype=int).tolist())


def tensor_to_pil(image):
    return tv.transforms.ToPILImage()(image.clamp(0, 1))


def add_image_header(image, text):
    header_height = 28
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    ImageDraw.Draw(canvas).text((6, 7), text, fill="black")
    return canvas


def imagenet_class_name(index):
    categories = tv.models.VGG16_BN_Weights.IMAGENET1K_V1.meta["categories"]
    index = int(index)
    return categories[index] if 0 <= index < len(categories) else f"class_{index}"


def detector_class_name(index, model_name):
    converter = LabelConverter()
    index = int(index)
    categories = converter.category91 if model_name == "faster_rcnn" else converter.category80
    return categories[index] if 0 <= index < len(categories) else f"class_{index}"


def annotate_detections(image, predictions, model_name):
    annotated = tensor_to_pil(image)
    draw = ImageDraw.Draw(annotated)
    if predictions.numel() == 0:
        return add_image_header(annotated, "Detections: none")
    predictions = predictions[predictions[:, 4].argsort(descending=True)][:20]
    for detection in predictions:
        x1, y1, x2, y2, confidence, class_index = detection.tolist()
        label = f"{detector_class_name(class_index, model_name)} {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(text_box, fill="red")
        draw.text((x1, y1), label, fill="white")
    return add_image_header(annotated, f"Detections: {len(predictions)} shown")


def prepare_swanlab_sample(sample, attack_type, model_name):
    if attack_type == "TA-C":
        clean_index = int(sample["clean_prediction"].reshape(-1)[0])
        attacked_index = int(sample["attacked_prediction"].reshape(-1)[0])
        clean_name = imagenet_class_name(clean_index)
        attacked_name = imagenet_class_name(attacked_index)
        clean_image = add_image_header(
            tensor_to_pil(sample["clean_image"]),
            f"Clean prediction: {clean_name} ({clean_index})",
        )
        attacked_image = add_image_header(
            tensor_to_pil(sample["attacked_image"]),
            f"Attack: {clean_name} -> {attacked_name}",
        )
    else:
        clean_image = annotate_detections(
            sample["clean_image"], sample["clean_prediction"], model_name)
        attacked_image = annotate_detections(
            sample["attacked_image"], sample["attacked_prediction"], model_name)
    return clean_image, attacked_image


def log_swanlab_epoch(run, metrics, epoch, attack_type, model_name,
                      train_losses=None,
                      patch_path=None, sample=None):
    if run is None:
        return
    import swanlab

    payload = {
        "metrics/ASR": metrics["ASR"],
        "metrics/No_triggered": metrics["No_triggered"],
        "metrics/Triggered": metrics["Triggered"],
    }
    if train_losses is not None:
        payload.update({
            "loss/total": train_losses["total"],
            "loss/loss1_triggered_attack": train_losses["loss1_triggered_attack"],
            "loss/loss2_aux_no_trigger": train_losses["loss2_aux_no_trigger"],
            "loss/loss3_tv": train_losses["loss3_tv"],
            "loss/loss4_content": train_losses["loss4_content"],
            "loss/loss5_nps": train_losses["loss5_nps"],
        })
    media_images = []
    if patch_path is not None:
        media_images.append(swanlab.Image(patch_path, caption="patch"))
    if sample is not None:
        clean_image, attacked_image = prepare_swanlab_sample(
            sample, attack_type, model_name)
        media_images.extend([
            swanlab.Image(clean_image, caption="clean_input"),
            swanlab.Image(attacked_image, caption="attacked_input"),
        ])
    if media_images:
        payload["media/epoch_images"] = media_images
    swanlab.log(payload, step=epoch)


def log_swanlab_async_config(run, row):
    if run is None or not row:
        return
    import swanlab
    from swanlab.data.modules.custom_charts.echarts import Table
    t = Table().add(headers=list(row.keys()), rows=[list(row.values())])
    swanlab.log({"async/config": t})


def log_swanlab_async_history(run, history):
    """Log accumulated per-epoch async selection history as a table."""
    if run is None or not history:
        return
    import swanlab
    from swanlab.data.modules.custom_charts.echarts import Table
    columns = list(history[0].keys())
    rows = [[h.get(k) for k in columns] for h in history]
    t = Table().add(headers=columns, rows=rows)
    swanlab.log({"async/history": t})


def log_swanlab_batch(run, batch_records, global_step_offset):
    """Log per-batch training losses, gradient norm, and update L2 as scalar curves."""
    if run is None or not batch_records:
        return
    import swanlab
    for rec in batch_records:
        step = global_step_offset + rec["batch"] + 1
        swanlab.log(
            {
                "loss/total": rec["total"],
                "loss/loss1_triggered_attack": rec["loss1_triggered_attack"],
                "loss/loss2_aux_no_trigger": rec["loss2_aux_no_trigger"],
                "loss/loss3_tv": rec["loss3_tv"],
                "loss/loss4_content": rec["loss4_content"],
                "loss/loss5_nps": rec["loss5_nps"],
                "loss/patch_gradient": rec["patch_gradient"],
                "loss/patch_update_l2": rec["patch_update_l2"],
            },
            step=step,
        )


def _bounded_interval(center, radius, lower, upper):
    center = float(center)
    radius = max(float(radius), 0.0)
    low = min(max(center - radius, lower), upper)
    high = min(max(center + radius, lower), upper)
    return (min(low, high), max(low, high))


def _sample_uniform(bounds):
    low, high = bounds
    if abs(high - low) < 1e-12:
        return float(low)
    return float(np.random.uniform(low, high))


def sample_laser_params(count, power_bounds, distance_bounds, angle_bounds, light_bounds):
    return [
        LaserParams(
            power_mw=_sample_uniform(power_bounds),
            distance_m=_sample_uniform(distance_bounds),
            angle_deg=_sample_uniform(angle_bounds),
            ambient_lux=_sample_uniform(light_bounds),
        )
        for _ in range(count)
    ]


def make_trigger_metadata(trigger_params, args, async_epoch=None):
    rows = []
    for idx, params in enumerate(trigger_params):
        row = {
            "index": idx,
            "power_mw": params.power_mw,
            "distance_m": params.distance_m,
            "angle_deg": params.angle_deg,
            "ambient_lux": params.ambient_lux,
            "model": args.laser_model,
            "color": args.laser_color,
            "trigger_height": args.trigger_height or 50,
            "trigger_width": args.trigger_width,
            "trigger_position": args.trigger_position,
        }
        if async_epoch is not None:
            row["async_epoch"] = async_epoch
        rows.append(row)
    return rows


def generate_laser_space(trigger_params, args, is_detector_attack, device):
    return generate_laser_trigger_tensor(
        params_grid=trigger_params,
        isdetector=is_detector_attack,
        model=args.laser_model,
        color=args.laser_color,
        trigger_height=args.trigger_height or 50,
        trigger_width=args.trigger_width,
        position=args.trigger_position,
        calibration=LaserCalibration(args.laser_k1, args.laser_k2, args.laser_k3, args.laser_k4),
        noise_std=args.trigger_noise_std,
        device=str(device),
    )


def configure_laser_trigger_layout(cfg, args, is_detector_attack):
    image_h = 640 if is_detector_attack else 224
    stripe_rows = args.trigger_width if args.trigger_width is not None else (args.trigger_height or 50)
    stripe_rows = max(1, min(int(stripe_rows), image_h))
    source_top = int(round((image_h - stripe_rows) * min(max(args.trigger_position, 0.0), 1.0)))
    cfg.ATTACKER.TRIGGER_STRIPE_TOP = source_top
    cfg.ATTACKER.TRIGGER_STRIPE_ROWS = stripe_rows


def build_initial_async_laser_space(args, is_detector_attack, device):
    trigger_params = sample_laser_params(
        args.trigger_param_samples,
        (args.async_min_power, args.async_max_power),
        (args.async_min_distance, args.async_max_distance),
        (args.async_min_angle, args.async_max_angle),
        (args.async_min_light, args.async_max_light),
    )
    trigger_metadata = make_trigger_metadata(
        trigger_params, args, async_epoch=1)
    trigger_mask = generate_laser_space(trigger_params, args, is_detector_attack, device)
    return trigger_mask, trigger_metadata


def build_async_laser_space(best_meta, args, is_detector_attack, device, next_epoch):
    trigger_params = sample_laser_params(
        args.trigger_param_samples,
        _bounded_interval(
            best_meta["power_mw"], args.async_power_radius,
            args.async_min_power, args.async_max_power),
        _bounded_interval(
            best_meta["distance_m"], args.async_distance_radius,
            args.async_min_distance, args.async_max_distance),
        _bounded_interval(
            best_meta["angle_deg"], args.async_angle_radius,
            args.async_min_angle, args.async_max_angle),
        _bounded_interval(
            best_meta["ambient_lux"], args.async_light_radius,
            args.async_min_light, args.async_max_light),
    )
    trigger_metadata = make_trigger_metadata(
        trigger_params, args, async_epoch=next_epoch)
    trigger_mask = generate_laser_space(trigger_params, args, is_detector_attack, device)
    return trigger_mask, trigger_metadata


def select_trigger_candidate(
    cfg,
    model,
    relpos,
    relpos3,
    patch,
    patch2,
    trigger_mask,
    quick_load,
    evaluate_dataloader,
    epoch,
    save_path,
    metric,
    search_batch,
    trigger_metadata=None,
    phase="epoch_search",
    return_selection=False,
    objective="metric",
    trigger_zeta=1.0,
    trigger_psi=None,
):
    best_index = 0
    best_metrics = None
    best_loss = None
    best_value = None
    for idx in range(trigger_mask.size(0)):
        candidate = torch.index_select(
            trigger_mask, 0, torch.tensor([idx], device=trigger_mask.device))
        if objective == "loss":
            loss_value = score_trigger_loss(
                cfg, model, relpos, relpos3, patch, patch2, candidate, quick_load,
                evaluate_dataloader, candidate.device, epoch, search_batch,
                zeta=trigger_zeta, psi=trigger_psi, quiet=True)
            if best_loss is None or loss_value < best_loss:
                best_index = idx
                best_loss = loss_value
                best_value = -loss_value
        else:
            with torch.no_grad():
                metrics = eval(
                    cfg, model, relpos, relpos3, patch, patch2, candidate, quick_load,
                    evaluate_dataloader, epoch, eval_batch=search_batch, quiet=True)
            metric_value = metrics[metric]
            if best_value is None or metric_value > best_value:
                best_index = idx
                best_metrics = metrics
                best_value = metric_value
    if objective == "loss":
        with torch.no_grad():
            selected_candidate = torch.index_select(
                trigger_mask, 0, torch.tensor([best_index], device=trigger_mask.device))
            metrics = eval(
                cfg, model, relpos, relpos3, patch, patch2, selected_candidate, quick_load,
                evaluate_dataloader, epoch, eval_batch=search_batch, quiet=True)
        best_metrics = metrics
    selected_meta = trigger_metadata[best_index] if trigger_metadata else {}
    selection_info = {
        "epoch": epoch,
        "phase": phase,
        "loss_value": best_loss,
        "p_mw": selected_meta.get("power_mw"),
        "d_m": selected_meta.get("distance_m"),
        "theta_deg": selected_meta.get("angle_deg"),
        "l_lux": selected_meta.get("ambient_lux"),
        "async_epoch": selected_meta.get("async_epoch"),
    }
    append_trigger_selection(save_path, selection_info)
    if objective == "loss":
        print(
            f"Selected trigger {best_index} by trigger_loss={best_loss} "
            f"(ASR={best_metrics['ASR']}, Triggered={best_metrics['Triggered']})"
        )
    else:
        print(
            f"Selected trigger {best_index} by {metric}={best_value} "
            f"(ASR={best_metrics['ASR']}, Triggered={best_metrics['Triggered']})"
        )
    selected_mask = torch.index_select(
        trigger_mask, 0, torch.tensor([best_index], device=trigger_mask.device))
    if return_selection:
        return selected_mask, selected_meta, best_metrics, selection_info
    return selected_mask


def slugify(value):
    if isinstance(value, (list, tuple)):
        value = "-".join(str(v) for v in value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "none"


set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type != "cuda":
    sys.exit("CUDA is required for the digital experiments in this project.")
if args.target != None:
    try:
        args.target = int(args.target)
    except ValueError:
        pass
if args.trigger_selection == "async-joint" and args.trigger_source != "laser":
    sys.exit("--trigger-selection async-joint requires --trigger-source laser")
if args.trigger_search_objective is None:
    args.trigger_search_objective = "loss" if args.trigger_selection == "async-joint" else "metric"
if args.trigger_param_samples < 1:
    raise ValueError("--trigger-param-samples must be at least 1")
cfg = ConfigParser(args, time_str)
if args.attack_type != None:
    cfg.ATTACKER.TYPE = args.attack_type
if args.det != None:
    if cfg.ATTACKER.TYPE == "TA-C":
        cfg.DETECTOR.NAME = [args.det]
    else:
        cfg.DETECTOR.NAME = args.det
if args.target != None:
    cfg.ATTACKER.TARGET_LABEL = args.target
if args.epochs is not None:
    cfg.ATTACKER.EPOCH = args.epochs
if args.train_batch is not None:
    cfg.ATTACKER.TRAIN_BATCH = args.train_batch
if args.eval_batch is not None:
    cfg.ATTACKER.EVAL_BATCH = args.eval_batch
if args.repeat is not None:
    cfg.ATTACKER.REPEAT = args.repeat
if args.batch_size is not None:
    cfg.DETECTOR.BATCH_SIZE = args.batch_size
if cfg.DETECTOR.BATCH_SIZE < 1:
    raise ValueError("DETECTOR.BATCH_SIZE must be at least 1")
if args.trigger_search_batch is None:
    args.trigger_search_batch = cfg.DETECTOR.BATCH_SIZE
if args.trigger_search_batch < 1:
    raise ValueError("--trigger-search-batch must be at least 1")
cfg.ATTACKER.FIXED_TOP = args.patch_top
cfg.ATTACKER.FIXED_LEFT = args.patch_left

print(f"Start Time: {time_str}_{cfg.ATTACKER.TYPE}_{cfg.DETECTOR.NAME}_{cfg.ATTACKER.TARGET_LABEL}")
logger(cfg, args)

if cfg.ATTACKER.TYPE != "TA-C":
    train_dataloader = load_coco(
        cfg.DATA.TRAIN.IMG_DIR, cfg.DATA.TRAIN.LAB_DIR, batch_size=cfg.DETECTOR.BATCH_SIZE)
    if args.eval_dataset == "coco":
        evaluate_dataloader = load_coco(
            cfg.DATA.TRAIN.IMG_DIR, cfg.DATA.TRAIN.LAB_DIR, batch_size=cfg.DETECTOR.BATCH_SIZE)
    elif args.eval_dataset == "bdd100k":
        evaluate_dataloader = load_bdd100k(cfg=cfg, batch_size=cfg.DETECTOR.BATCH_SIZE)
    else:
        evaluate_dataloader = load_kitti(cfg=cfg, batch_size=cfg.DETECTOR.BATCH_SIZE)
    model = get_det_model(device, cfg.DETECTOR.NAME)
else:
    train_dataloader = load_imagenet_val(
        cfg.DATA.TRAIN.IMG_DIR, batch_size=cfg.DETECTOR.BATCH_SIZE)
    evaluate_dataloader = load_imagenet_one_per_class_val(
        cfg.DATA.TRAIN.IMG_DIR, batch_size=cfg.DETECTOR.BATCH_SIZE)
    if args.eval_batch is None:
        cfg.ATTACKER.EVAL_BATCH = 1000
    model = get_cls_ens_model(device, cfg.DETECTOR.NAME)
if args.eval_det is not None:
    if cfg.ATTACKER.TYPE == "TA-C":
        eval_model = get_cls_ens_model(device, [args.eval_det])
    else:
        eval_model = get_det_model(device, args.eval_det)
else:
    eval_model = model

# Size and Location of Images Initialization
configured_patch_size = (
    int(getattr(cfg.ATTACKER.PATCH, "HEIGHT", 200)),
    int(getattr(cfg.ATTACKER.PATCH, "WIDTH", 200)),
)
bgsize = configured_patch_size  # Size of Background
bgsize_TA_Cls = configured_patch_size
psize = configured_patch_size
if cfg.ATTACKER.TYPE == "HA":
    bgsize = (200, 200)
if args.patch_size is not None:
    if cfg.ATTACKER.TYPE == "TA-C":
        bgsize_TA_Cls = (args.patch_size, args.patch_size)
    elif cfg.ATTACKER.TYPE == "HA":
        psize = (args.patch_size, args.patch_size)
    else:
        bgsize = (args.patch_size, args.patch_size)

def _center_pos(container_size, item_size):
    return (
        max(0, (container_size[0] - item_size[0]) // 2),
        max(0, (container_size[1] - item_size[1]) // 2),
    )

if not cfg.ATTACKER.DOUBLE_APPLY:
    relpos = _center_pos(bgsize, psize)
else:
    relpos = (
        max(0, min(bgsize[0] - psize[0], bgsize[0] // 4 - psize[0] // 2)),
        max(0, (bgsize[1] - psize[1]) // 2),
    )
relpos3 = (
    max(0, min(bgsize[0] - psize[0], 3 * bgsize[0] // 4 - psize[0] // 2)),
    max(0, (bgsize[1] - psize[1]) // 2),
)

# Adversarial Patch Initialization
patch2 = LHawk(bgsize[0], bgsize[1], cfg.target_index, device, eot=cfg.ATTACKER.PATCH.EOT,
                eot_scale=cfg.ATTACKER.PATCH.SCALE, eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)
resize = tv.transforms.Resize(bgsize)
quick_load = lambda x: resize(patch2.pil2tensor(Image.open(x))).unsqueeze(0).to(device)
patch2.data = quick_load("assets/stop_sign.png")
patch2.load_mask("assets/stop_sign_mask.png")
patch2.rotate_mask = resize(patch2.rotate_mask)

folder_path = "src/color_stripe/trigger"  # Replace with your actual folder path
is_detector_attack = cfg.ATTACKER.TYPE != "TA-C"
cfg.ATTACKER.ALIGN_TRIGGER_TO_PATCH = args.trigger_source == "laser"
if cfg.ATTACKER.ALIGN_TRIGGER_TO_PATCH:
    configure_laser_trigger_layout(cfg, args, is_detector_attack)
trigger_metadata = []
if args.trigger_source == "fixed":
    if is_detector_attack:
        trigger_mask = generate_trigger_tensor(folder_path)
    else:
        trigger_mask = generate_trigger_tensor(folder_path, isdetector=False)
elif args.trigger_source == "none":
    trigger_shape = (1, 3, 640, 640) if is_detector_attack else (1, 3, 224, 224)
    trigger_mask = torch.zeros(trigger_shape, device=device)
    print(f"Generated zero trigger tensor with shape: {trigger_mask.shape}")
else:
    if args.trigger_selection == "async-joint":
        trigger_mask, trigger_metadata = build_initial_async_laser_space(
            args, is_detector_attack, device)
    else:
        trigger_params = build_laser_param_grid(
            powers=parse_float_spec(args.laser_power),
            distances=parse_float_spec(args.laser_distance),
            angles=parse_float_spec(args.laser_angle),
            lights=parse_float_spec(args.ambient_light),
        )
        trigger_metadata = make_trigger_metadata(trigger_params, args)
        trigger_mask = generate_laser_space(trigger_params, args, is_detector_attack, device)

# Cal Content Loss through VGG19 Network proposed by TPatch
if args.content_pretrained:
    a = tv.models.vgg19(weights=tv.models.VGG19_Weights.DEFAULT).to(device)
else:
    a = tv.models.vgg19(weights=None).to(device)
content_loss = ContentLoss(a.features, cfg.ATTACKER.PATCH.CONTENT, device, extract_layer=11)
tv_loss = TVLoss()

# Cal NPS Loss
if cfg.ATTACKER.TYPE == "HA":
    nps_loss = NPS_Loss("src/printability/30values.txt", psize).to(device)
elif cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
    nps_loss = NPS_Loss("src/printability/30values.txt", bgsize).to(device)
elif cfg.ATTACKER.TYPE == "TA-C":
    nps_loss = NPS_Loss("src/printability/30values.txt", bgsize_TA_Cls).to(device)

run_parts = [time_str]
if args.run_tag:
    run_parts.append(slugify(args.run_tag))
run_parts.extend([
    slugify(cfg.ATTACKER.TYPE),
    slugify(cfg.DETECTOR.NAME),
    slugify(cfg.ATTACKER.TARGET_LABEL),
])
save_path = os.path.join(args.exp_dir, "train_{}".format("_".join(run_parts)))
if not os.path.exists(save_path):
    os.makedirs(save_path)
write_run_metadata(save_path, cfg, args, time_str)
write_trigger_candidates(save_path, trigger_metadata)
async_config = None
if args.trigger_selection == "async-joint":
    async_config = async_joint_config_row(args, cfg)
    write_async_joint_config(save_path, async_config)
swanlab_run = init_swanlab(save_path, cfg, args)
log_swanlab_async_config(swanlab_run, async_config)
swanlab_patch_epochs = media_epochs(cfg.ATTACKER.EPOCH)
best_asr_so_far = float("-inf")
async_history = []

if cfg.ATTACKER.TYPE == "HA":
    patch = LHawk(psize[0], psize[1], cfg.target_index, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=0.97, eot_angle=math.pi / 60)
elif cfg.ATTACKER.TYPE == "CA" or cfg.ATTACKER.TYPE == "TA-D":
    patch = LHawk(bgsize[0], bgsize[1], cfg.target_index, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=cfg.ATTACKER.PATCH.SCALE,
                   eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)
elif cfg.ATTACKER.TYPE == "TA-C":
    patch = LHawk(bgsize_TA_Cls[0], bgsize_TA_Cls[1], cfg.ATTACKER.TARGET_LABEL, device=device, lr=cfg.ATTACKER.LR, momentum=cfg.ATTACKER.MOMENTUM,
                   eot=cfg.ATTACKER.PATCH.EOT, eot_scale=cfg.ATTACKER.PATCH.SCALE,
                   eot_angle=cfg.ATTACKER.PATCH.ANGLE, p=1)

for e in range(1, cfg.ATTACKER.EPOCH + 1):
    async_info = None
    if args.trigger_selection == "epoch-search" and trigger_mask.size(0) > 1:
        train_trigger_mask = select_trigger_candidate(
            cfg, model, relpos, relpos3, patch, patch2, trigger_mask, quick_load,
            evaluate_dataloader, e, save_path, args.trigger_search_metric,
            args.trigger_search_batch, trigger_metadata=trigger_metadata,
            objective=args.trigger_search_objective,
            trigger_zeta=args.async_trigger_zeta,
            trigger_psi=args.async_trigger_psi)
    elif args.trigger_selection == "async-joint":
        train_trigger_mask = trigger_mask
    else:
        train_trigger_mask = trigger_mask
    train_losses, batch_records = train(
        cfg, model, relpos, relpos3, patch, patch2, train_trigger_mask,
        content_loss, tv_loss, nps_loss, quick_load, train_dataloader, device, e)
    batch_step_offset = (e - 1) * cfg.ATTACKER.TRAIN_BATCH
    log_swanlab_batch(swanlab_run, batch_records, batch_step_offset)
    with torch.no_grad():
        if args.trigger_selection == "async-joint":
            selected_mask, selected_meta, selected_metrics, async_info = select_trigger_candidate(
                cfg, eval_model, relpos, relpos3, patch, patch2, trigger_mask, quick_load,
                evaluate_dataloader, e, save_path, args.trigger_search_metric,
                args.trigger_search_batch, trigger_metadata=trigger_metadata,
                phase="trigger_step", return_selection=True,
                objective=args.trigger_search_objective,
                trigger_zeta=args.async_trigger_zeta,
                trigger_psi=args.async_trigger_psi)
            print(
                "Async-joint trigger step selected "
                f"p={selected_meta.get('power_mw')}mW, "
                f"d={selected_meta.get('distance_m')}m, "
                f"theta={selected_meta.get('angle_deg')}deg, "
                f"l={selected_meta.get('ambient_lux')}Lux."
            )
        elif train_trigger_mask.size(0) == 1:
            selected_mask = train_trigger_mask
            print("Use selected trigger mask for Eval.")
        else:
            random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
            print(f"Select {random_index.item()} mask for Eval.")
            selected_mask = torch.index_select(trigger_mask, 0, random_index)
        should_capture_media = swanlab_run is not None and e in swanlab_patch_epochs
        if should_capture_media:
            metrics, visualization_sample = eval(
                cfg, eval_model, relpos, relpos3, patch, patch2, selected_mask,
                quick_load, evaluate_dataloader, e, capture_sample=True)
        else:
            metrics = eval(
                cfg, eval_model, relpos, relpos3, patch, patch2, selected_mask,
                quick_load, evaluate_dataloader, e)
            visualization_sample = None
        is_best_asr = metrics["ASR"] > best_asr_so_far
        if is_best_asr:
            best_asr_so_far = metrics["ASR"]
        if swanlab_run is not None and is_best_asr and visualization_sample is None:
            _, visualization_sample = eval(
                cfg, eval_model, relpos, relpos3, patch, patch2, selected_mask,
                quick_load, evaluate_dataloader, e, eval_batch=1, quiet=True, capture_sample=True)
        should_upload_media = swanlab_run is not None and (should_capture_media or is_best_asr)
        append_metrics(save_path, metrics)
        if args.trigger_selection == "async-joint" and e < cfg.ATTACKER.EPOCH:
            trigger_mask, trigger_metadata = build_async_laser_space(
                selected_meta, args, is_detector_attack, device, next_epoch=e + 1)
            write_trigger_candidates(save_path, trigger_metadata)
    patch_path = os.path.join(save_path, f"p_epoch{e}.png")
    patch.save(patch_path)
    if cfg.ATTACKER.TYPE == "HA":
        patch2.save(os.path.join(save_path, f"p2_white_epoch{e}.png"))
    if e % cfg.ATTACKER.DECAY_EPOCH == 0:
        patch.opt.lr *= cfg.ATTACKER.STEP_LR
    if async_info is not None:
        async_history.append(dict(async_info))
    log_swanlab_epoch(
        swanlab_run,
        metrics,
        e,
        cfg.ATTACKER.TYPE,
        args.eval_det or args.det,
        train_losses=train_losses,
        patch_path=patch_path if should_upload_media else None,
        sample=visualization_sample if should_upload_media else None,
    )

log_swanlab_async_history(swanlab_run, async_history)
if swanlab_run is not None:
    import swanlab
    swanlab.finish()
