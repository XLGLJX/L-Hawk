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
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--train-batch', type=int, default=None)
parser.add_argument('--eval-batch', type=int, default=None)
parser.add_argument('--repeat', type=int, default=None)
parser.add_argument('--eval-dataset', choices=("kitti", "bdd100k", "coco"), default="kitti")
parser.add_argument('--content-pretrained', action="store_true",
                    help='Use torchvision pretrained VGG19 for content loss; may require cached/downloaded weights.')
parser.add_argument('--trigger-source', choices=("fixed", "laser"), default="fixed")
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
    train_dataloader = evaluate_dataloader = load_imagenet_val(cfg.DATA.TRAIN.IMG_DIR)
    model = get_cls_ens_model(device, cfg.DETECTOR.NAME)

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
if args.trigger_source == "fixed":
    if is_detector_attack:
        trigger_mask = generate_trigger_tensor(folder_path)
    else:
        trigger_mask = generate_trigger_tensor(folder_path, isdetector=False)
else:
    trigger_params = build_laser_param_grid(
        powers=parse_float_spec(args.laser_power),
        distances=parse_float_spec(args.laser_distance),
        angles=parse_float_spec(args.laser_angle),
        lights=parse_float_spec(args.ambient_light),
    )
    trigger_mask = generate_laser_trigger_tensor(
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

save_path = os.path.join(
    args.exp_dir,
    "train_{}_{}_{}_{}".format(
        time_str,
        slugify(cfg.ATTACKER.TYPE),
        slugify(cfg.DETECTOR.NAME),
        slugify(cfg.ATTACKER.TARGET_LABEL),
    ),
)
if not os.path.exists(save_path):
    os.makedirs(save_path)
write_run_metadata(save_path, cfg, args, time_str)

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
    train(cfg, model, relpos, relpos3, patch, patch2, trigger_mask, content_loss, tv_loss, nps_loss, quick_load, train_dataloader, device, e)
    with torch.no_grad():
        random_index = torch.randint(0, trigger_mask.size(0), (1,), device=device)
        print(f"Select {random_index.item()} mask for Eval.")
        selected_mask = torch.index_select(trigger_mask, 0, random_index)
        metrics = eval(cfg, model, relpos, relpos3, patch, patch2, selected_mask, quick_load, evaluate_dataloader, e)
        append_metrics(save_path, metrics)
    patch.save(os.path.join(save_path, f"p_epoch{e}.png"))
    if cfg.ATTACKER.TYPE == "HA":
        patch2.save(os.path.join(save_path, f"p2_white_epoch{e}.png"))
    if e % cfg.ATTACKER.DECAY_EPOCH == 0:
        patch.opt.lr *= cfg.ATTACKER.STEP_LR
