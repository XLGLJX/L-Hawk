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
    build_laser_param_grid,
    generate_laser_trigger_tensor,
    parse_float_spec,
)
from utils.parser import ConfigParser, logger
from src.patch_train import train, eval
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
parser.add_argument('--train-batch', type=int, default=None)
parser.add_argument('--eval-batch', type=int, default=None)
parser.add_argument('--repeat', type=int, default=None)
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
                    help="Synthetic trigger stripe height. Defaults to 50 for laser triggers.")
parser.add_argument('--trigger-width', type=int, default=None,
                    help="Synthetic trigger stripe width. Defaults to the whole image width.")
parser.add_argument('--trigger-position', type=float, default=0.5,
                    help="Vertical stripe position in [0,1], where 0 is top and 1 is bottom.")
parser.add_argument('--trigger-noise-std', type=float, default=0.0)
parser.add_argument('--laser-k1', type=float, default=12.0)
parser.add_argument('--laser-k2', type=float, default=2.0e-5)
parser.add_argument('--laser-k3', type=float, default=4.0)
parser.add_argument('--laser-k4', type=float, default=1.0e-5)
parser.add_argument('--trigger-selection', choices=("random", "epoch-search", "async-joint"), default="random")
parser.add_argument('--trigger-search-metric', choices=("ASR", "Triggered", "No_triggered"), default="ASR")
parser.add_argument('--trigger-search-batch', type=int, default=8)
parser.add_argument('--async-power-radius', type=float, default=10.0,
                    help="Power search radius around the best trigger for async-joint.")
parser.add_argument('--async-distance-radius', type=float, default=5.0,
                    help="Distance search radius around the best trigger for async-joint.")
parser.add_argument('--async-angle-radius', type=float, default=5.0,
                    help="Incidence-angle search radius around the best trigger for async-joint.")
parser.add_argument('--async-light-radius', type=float, default=200.0,
                    help="Ambient-light search radius around the best trigger for async-joint.")
parser.add_argument('--async-shrink', type=float, default=0.75,
                    help="Radius decay applied after each async-joint epoch.")
parser.add_argument('--async-min-power', type=float, default=1.0)
parser.add_argument('--async-max-power', type=float, default=100.0)
parser.add_argument('--async-min-distance', type=float, default=1.0)
parser.add_argument('--async-max-distance', type=float, default=80.0)
parser.add_argument('--async-min-angle', type=float, default=-60.0)
parser.add_argument('--async-max-angle', type=float, default=60.0)
parser.add_argument('--async-min-light', type=float, default=0.0)
parser.add_argument('--async-max-light', type=float, default=5000.0)
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
        "epoch", "phase", "selected_index", "metric", "metric_value",
        "ASR", "No_triggered", "Triggered",
        "power_mw", "distance_m", "angle_deg", "ambient_lux",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
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


def _bounded_triplet(center, radius, lower, upper):
    center = float(center)
    radius = float(radius)
    values = [center] if radius <= 0 else [center - radius, center, center + radius]
    clipped = []
    for value in values:
        value = min(max(value, lower), upper)
        if not any(abs(value - old) < 1e-9 for old in clipped):
            clipped.append(value)
    return clipped


def make_trigger_metadata(trigger_params, args, async_epoch=None, async_radius_scale=None):
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
            row["async_radius_scale"] = async_radius_scale
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


def build_async_laser_space(best_meta, args, is_detector_attack, device, next_epoch):
    radius_scale = args.async_shrink ** max(next_epoch - 1, 0)
    powers = _bounded_triplet(
        best_meta["power_mw"], args.async_power_radius * radius_scale,
        args.async_min_power, args.async_max_power)
    distances = _bounded_triplet(
        best_meta["distance_m"], args.async_distance_radius * radius_scale,
        args.async_min_distance, args.async_max_distance)
    angles = _bounded_triplet(
        best_meta["angle_deg"], args.async_angle_radius * radius_scale,
        args.async_min_angle, args.async_max_angle)
    lights = _bounded_triplet(
        best_meta["ambient_lux"], args.async_light_radius * radius_scale,
        args.async_min_light, args.async_max_light)
    trigger_params = build_laser_param_grid(powers, distances, angles, lights)
    trigger_metadata = make_trigger_metadata(
        trigger_params, args, async_epoch=next_epoch, async_radius_scale=radius_scale)
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
):
    best_index = 0
    best_metrics = None
    best_value = None
    with torch.no_grad():
        for idx in range(trigger_mask.size(0)):
            candidate = torch.index_select(
                trigger_mask, 0, torch.tensor([idx], device=trigger_mask.device))
            metrics = eval(
                cfg, model, relpos, relpos3, patch, patch2, candidate, quick_load,
                evaluate_dataloader, epoch, eval_batch=search_batch, quiet=True)
            metric_value = metrics[metric]
            if best_value is None or metric_value > best_value:
                best_index = idx
                best_metrics = metrics
                best_value = metric_value
    selected_meta = trigger_metadata[best_index] if trigger_metadata else {}
    append_trigger_selection(save_path, {
        "epoch": epoch,
        "phase": phase,
        "selected_index": best_index,
        "metric": metric,
        "metric_value": best_value,
        "ASR": best_metrics["ASR"],
        "No_triggered": best_metrics["No_triggered"],
        "Triggered": best_metrics["Triggered"],
        "power_mw": selected_meta.get("power_mw"),
        "distance_m": selected_meta.get("distance_m"),
        "angle_deg": selected_meta.get("angle_deg"),
        "ambient_lux": selected_meta.get("ambient_lux"),
    })
    print(
        f"Selected trigger {best_index} by {metric}={best_value} "
        f"(ASR={best_metrics['ASR']}, Triggered={best_metrics['Triggered']})"
    )
    selected_mask = torch.index_select(
        trigger_mask, 0, torch.tensor([best_index], device=trigger_mask.device))
    if return_selection:
        return selected_mask, selected_meta, best_metrics
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
cfg.ATTACKER.FIXED_TOP = args.patch_top
cfg.ATTACKER.FIXED_LEFT = args.patch_left

print(f"Start Time: {time_str}_{cfg.ATTACKER.TYPE}_{cfg.DETECTOR.NAME}_{cfg.ATTACKER.TARGET_LABEL}")
logger(cfg, args)

if cfg.ATTACKER.TYPE != "TA-C":
    train_dataloader = load_coco(cfg.DATA.TRAIN.IMG_DIR, cfg.DATA.TRAIN.LAB_DIR)
    if args.eval_dataset == "coco":
        evaluate_dataloader = load_coco(cfg.DATA.TRAIN.IMG_DIR, cfg.DATA.TRAIN.LAB_DIR)
    elif args.eval_dataset == "bdd100k":
        evaluate_dataloader = load_bdd100k(cfg=cfg)
    else:
        evaluate_dataloader = load_kitti(cfg=cfg)
    model = get_det_model(device, cfg.DETECTOR.NAME)
else:
    train_dataloader = load_imagenet_val(cfg.DATA.TRAIN.IMG_DIR)
    evaluate_dataloader = load_imagenet_one_per_class_val(cfg.DATA.TRAIN.IMG_DIR)
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
bgsize = (200, 200)  # Size of Background
bgsize_TA_Cls = (100, 100)
if not cfg.ATTACKER.DOUBLE_APPLY:
    psize = (70, 170)
    relpos = (65, 15)
else:
    psize = (40, 110)
    relpos = (28, 45)
relpos3 = (132, 45)
if args.patch_size is not None:
    if cfg.ATTACKER.TYPE == "TA-C":
        bgsize_TA_Cls = (args.patch_size, args.patch_size)
    elif cfg.ATTACKER.TYPE == "HA":
        psize = (args.patch_size, args.patch_size)
    else:
        bgsize = (args.patch_size, args.patch_size)

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
swanlab_run = init_swanlab(save_path, cfg, args)
swanlab_patch_epochs = media_epochs(cfg.ATTACKER.EPOCH)

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
    if args.trigger_selection == "epoch-search" and trigger_mask.size(0) > 1:
        train_trigger_mask = select_trigger_candidate(
            cfg, model, relpos, relpos3, patch, patch2, trigger_mask, quick_load,
            evaluate_dataloader, e, save_path, args.trigger_search_metric,
            args.trigger_search_batch, trigger_metadata=trigger_metadata)
    elif args.trigger_selection == "async-joint":
        train_trigger_mask = trigger_mask
    else:
        train_trigger_mask = trigger_mask
    train_losses = train(
        cfg, model, relpos, relpos3, patch, patch2, train_trigger_mask,
        content_loss, tv_loss, nps_loss, quick_load, train_dataloader, device, e)
    with torch.no_grad():
        if args.trigger_selection == "async-joint":
            selected_mask, selected_meta, selected_metrics = select_trigger_candidate(
                cfg, eval_model, relpos, relpos3, patch, patch2, trigger_mask, quick_load,
                evaluate_dataloader, e, save_path, args.trigger_search_metric,
                args.trigger_search_batch, trigger_metadata=trigger_metadata,
                phase="trigger_step", return_selection=True)
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
        if swanlab_run is not None and e in swanlab_patch_epochs:
            metrics, visualization_sample = eval(
                cfg, eval_model, relpos, relpos3, patch, patch2, selected_mask,
                quick_load, evaluate_dataloader, e, capture_sample=True)
        else:
            metrics = eval(
                cfg, eval_model, relpos, relpos3, patch, patch2, selected_mask,
                quick_load, evaluate_dataloader, e)
            visualization_sample = None
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
    log_swanlab_epoch(
        swanlab_run,
        metrics,
        e,
        cfg.ATTACKER.TYPE,
        args.eval_det or args.det,
        train_losses=train_losses,
        patch_path=patch_path if e in swanlab_patch_epochs else None,
        sample=visualization_sample,
    )

if swanlab_run is not None:
    import swanlab
    swanlab.finish()
