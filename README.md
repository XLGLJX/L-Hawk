# [NDSS'25] L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target

Our work is accepted by NDSS Symposium 2025.
The paper will appear in the conference proceeding.

We present the Pytorch implementation of digital L-Hawk's optimization and evaluation below.
The work "[Usenix'24] TPatch: A Triggered Physical Adversarial Patch" has been very inspiring to us.

## Environment Installation

`conda create -n l-hawk python=3.8`

`conda activate l-hawk`

`pip install -r requirements.txt`

The CUDA environment (CUDA 11.7) for Pytorch will be installed.
We also successfully run the code under those environments with higher Pytorch and CUDA version.

## Datasets Setting
For this reproduction branch, the digital experiments use ImageNet for
classifier attacks and COCO val2014 for detector attacks. KITTI and BDD100K
paths are still supported by the original loaders, but they are not required for
the Plan-B reproduction commands documented below.

## Victim Models
The target models include `YOLO V3/V5`, `Faster R-CNN`, `VGG-13/16/19`, `ResNet-50/101/152`, `Inception-v3`, and `MobileNet-v2`.
You can download all [model weight](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing) and place them under the folder **detlib/weights**.

## Digital Attack Demo
We present a simple demo: train an adversarial patch based on fixed color stripes we provide.
First, you can initialize the parameters for different attacks in `./configs`.
Then, run `demo.py` to generate and evaluate the patch for HA(Hiding Attack), CA(Creating Attack), TA-D(Targeted Attack Against Detectors), and TA-C(Targeted Attack Against Classifiers).

Examples:

```bash
# TA-C against a single classifier target.
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920

# CA against YOLOv5 on COCO evaluation images.
python demo.py --cfg configs/CA.yaml --attack_type CA --det yolov5 --target "stop sign"

# Quick smoke-sized run. Use this to validate paths before full experiments.
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --epochs 1 --train-batch 1 --eval-batch 1 --repeat 1

# Detector smoke-sized run using COCO as the evaluation source.
python demo.py --cfg configs/CA.yaml --attack_type CA --det yolov5 --target "stop sign" \
  --eval-dataset coco --epochs 1 --train-batch 1 --eval-batch 1 --repeat 1
```

Each run writes its generated patches, `run_config.json`, and `metrics.csv` under `exp/`.
The metrics file records `ASR`, `No_triggered`, and `Triggered` for each epoch.
For TA-C, evaluation defaults to a fixed 1,000-image ImageNet-1K subset with
one deterministic image from every class. Training remains randomly sampled.

### SwanLab Tracking

Install and log in once before using online tracking:

```bash
pip install swanlab
swanlab login
```

Add `--swanlab` to any `demo.py` command to log `ASR`, `No_triggered`, and
`Triggered` after every epoch. A run uploads at most 10 evenly spaced patch
images. At each selected epoch SwanLab receives three images: the learned patch,
a random clean input without patch or laser, and the same input attacked with
both patch and laser trigger. Classifier images are labeled with the clean-to-
attacked class change; detector images include predicted boxes and labels.

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --epochs 20 --train-batch 50 --eval-batch 1000 --repeat 20 \
  --exp_dir exp/tac --swanlab --swanlab-project l-hawk
```

Use `--swanlab-mode offline` when the training host should only save local
SwanLab data for a later `swanlab sync`.

For a small experiment matrix, use:

```bash
python scripts/run_digital_experiments.py \
  --attacks TA-C,CA \
  --classifiers vgg16,res50 \
  --detectors yolov5 \
  --classifier-targets 920 \
  --detector-targets "stop sign" \
  --eval-dataset coco
```

Add `--dry-run` to print the commands without executing them.

### Parameterized Laser Trigger

The default trigger source is the captured fixed color-stripe images under
`src/color_stripe/trigger`. To run the approximate parameterized trigger model
used for the plan-B digital reproduction work, set `--trigger-source laser`.
This implements a configurable approximation of the paper's `S(p,d,theta,l)`
trigger generation function with linear, sigmoid, and Gaussian stripe profiles.
The calibration constants are exposed as CLI parameters because the paper does
not publish sensor-specific `k1..k4` values.

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --trigger-source laser \
  --laser-model linear \
  --laser-color green \
  --laser-power 10:70:10 \
  --laser-distance 30 \
  --laser-angle 18 \
  --ambient-light 1000 \
  --trigger-height 50
```

For red-trigger experiments, use `--laser-color red`. For incidence-shape
experiments, change `--laser-model` to `sigmoid` or `gaussian`.

To approximate the paper's asynchronous trigger/patch optimization loop, use
`--trigger-selection async-joint`. Each epoch trains the patch against the
current trigger space, selects the best trigger parameters with
`--trigger-search-metric`, and rebuilds the next trigger space around that
selection with shrinking search radii. The older `epoch-search` mode only
chooses one candidate trigger before patch training and does not update the
trigger parameter distribution.

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --trigger-source laser \
  --laser-power 10:70:10 \
  --trigger-selection async-joint \
  --trigger-search-metric ASR \
  --trigger-search-batch 8 \
  --async-power-radius 10 \
  --async-distance-radius 5 \
  --async-angle-radius 5 \
  --async-light-radius 200 \
  --async-shrink 0.75
```

Patch-size sweeps can be run by changing `--patch-size`, for example:

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det res50 --target 920 \
  --trigger-source laser \
  --patch-size 64 \
  --epochs 20 --train-batch 50 --eval-batch 1000 --repeat 20
```

Plan-B sweep helper:

```bash
# Laser power sweep: 10 mW to 70 mW.
python scripts/run_plan_b_sweeps.py \
  --sweep power \
  --values 10,20,30,40,50,60,70 \
  --cfg configs/TA-C.yaml \
  --attack TA-C \
  --model res50 \
  --target 920

# Trigger color sweep.
python scripts/run_plan_b_sweeps.py \
  --sweep color \
  --values green,red \
  --cfg configs/CA.yaml \
  --attack CA \
  --model yolov5 \
  --target "stop sign" \
  --eval-dataset coco

# Patch-size sweep.
python scripts/run_plan_b_sweeps.py \
  --sweep patch-size \
  --values 32,48,64,80 \
  --cfg configs/TA-C.yaml \
  --attack TA-C \
  --model res50 \
  --target 920

# Patch lateral-position sweep.
python scripts/run_plan_b_sweeps.py \
  --sweep patch-left \
  --values 0,32,64,96 \
  --cfg configs/TA-C.yaml \
  --attack TA-C \
  --model res50 \
  --target 920

# Laser distance / incidence-angle / ambient-light sweeps can be launched with:
python scripts/run_plan_b_sweeps.py --sweep distance --values 10,15,20,25,30
python scripts/run_plan_b_sweeps.py --sweep angle --values 0,15,30,45
python scripts/run_plan_b_sweeps.py --sweep ambient-light --values 1238,719,461,198,0
```

Collect completed sweep results:

```bash
python scripts/collect_plan_b_results.py \
  --root exp/plan-b \
  --output exp/plan-b/summary.csv
```

Compare completed runs against the paper's digital ASR baselines:

```bash
python scripts/compare_plan_b_to_paper.py \
  --summary exp/plan-b/summary.csv \
  --output exp/plan-b/paper_comparison.csv
```

The sweep helper passes `--seed`, `--experiment-name`, `--profile`, and a
generated `--run-tag` to `demo.py`, so manually launched sweeps are traceable in
the collected summary CSV as well.

Plot collected results:

```bash
# Bar chart for a single sweep axis.
python scripts/plot_plan_b_results.py \
  --summary exp/plan-b/summary.csv \
  --plot bar \
  --x selected_power_mw \
  --y ASR \
  --output exp/plan-b/power_asr.png

# Heatmap for two sweep axes.
python scripts/plot_plan_b_results.py \
  --summary exp/plan-b/summary.csv \
  --plot heatmap \
  --x patch_left \
  --heatmap-y selected_power_mw \
  --y ASR \
  --output exp/plan-b/power_patch_left_asr.png
```

Transfer tests use `--eval-det` to train on one model and evaluate on another:

```bash
# Classifier transfer: optimize on VGG16, evaluate on ResNet-50.
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C \
  --det vgg16 \
  --eval-det res50 \
  --target 920 \
  --trigger-source laser \
  --epochs 20 --train-batch 50 --eval-batch 1000 --repeat 20

# Detector transfer: optimize on YOLOv5, evaluate on YOLOv3.
python demo.py --cfg configs/CA.yaml --attack_type CA \
  --det yolov5 \
  --eval-det yolov3 \
  --target "stop sign" \
  --eval-dataset coco \
  --trigger-source laser \
  --epochs 20 --train-batch 50 --eval-batch 800 --repeat 20
```

Plan-B ablations compare trigger sources, epoch trigger search, and async-joint:

```bash
python scripts/run_plan_b_ablation.py \
  --modes none,fixed,laser-random,laser-epoch-search,laser-async-joint \
  --cfg configs/TA-C.yaml \
  --attack TA-C \
  --model res50 \
  --target 920
```

Each ablation mode receives a distinct run tag and records the same traceability
fields as the manifest runner.

Plan-B manifest runner expands predefined paper-aligned experiment matrices:

```bash
# List available manifest experiments.
python scripts/run_plan_b_manifest.py --list

# Dry-run smoke commands for selected experiments.
python scripts/run_plan_b_manifest.py \
  --profile smoke \
  --experiments overall_tac,factor_power_tac \
  --dry-run

# Full profile uses 20 epochs, 50 training batches, and 800 eval samples.
python scripts/run_plan_b_manifest.py \
  --profile full \
  --experiments factor_power_tac

# Paper-aligned heatmap grids are available as explicit experiments.
python scripts/run_plan_b_manifest.py \
  --profile full \
  --experiments factor_power_patch_left_tac,factor_trigger_position_width_tac
```

Manifest runs pass `--experiment-name`, `--profile`, `--run-tag`, and `--seed`
to `demo.py`. These fields are saved in each run's `run_config.json`, and
`scripts/collect_plan_b_results.py` exports them to the summary CSV for
traceability across smoke and full experiment matrices.

For paper comparison after manifest runs:

```bash
python scripts/compare_plan_b_to_paper.py \
  --summary exp/plan-b-manifest/summary.csv \
  --where profile=full \
  --output exp/plan-b-manifest/paper_comparison.csv
```

Example heatmap plots for manifest outputs:

```bash
python scripts/plot_plan_b_results.py \
  --summary exp/plan-b-manifest/summary.csv \
  --plot heatmap \
  --x patch_left \
  --heatmap-y laser_power \
  --y ASR \
  --where experiment_name=factor_power_patch_left_tac \
  --output exp/plan-b-manifest/power_patch_left_asr.png

python scripts/plot_plan_b_results.py \
  --summary exp/plan-b-manifest/summary.csv \
  --plot heatmap \
  --x trigger_position \
  --heatmap-y trigger_width \
  --y ASR \
  --where experiment_name=factor_trigger_position_width_tac \
  --output exp/plan-b-manifest/trigger_position_width_asr.png

python scripts/plot_plan_b_results.py \
  --summary exp/plan-b-manifest/summary.csv \
  --plot heatmap \
  --x laser_distance \
  --heatmap-y laser_angle \
  --y ASR \
  --where experiment_name=factor_power_distance_angle_tac \
  --where laser_power=50 \
  --title "power=50 mW" \
  --output exp/plan-b-manifest/distance_angle_power50_asr.png
```

## Physical Attack Demo
Physical attack demos (such as, indoor/outdoor attacks, various speed attacks, and end-to-end attacks) are available in [Link](https://drive.google.com/drive/folders/1nnzW85pbG9vF1T1T4Tdw6EagopkG_Dv4?usp=sharing).

**Our Contact:**
Taifeng Liu ([tfliu@gmx.com](tfliu@gmx.com))

## Paper Reference
```
@inproceedings{lhawk2025ndss,
  address   = {San Diego, CA},
  title     = {L-HAWK: A Controllable Physical Adversarial Patch against A Long-Distance Target},
  booktitle = {Network and Distributed System Security Symposium, {NDSS} 2025},
  publisher = {The Internet Society},
  author    = {Taifeng Liu, Yang Liu, Zhuo Ma, Tong Yang, Xinjing Liu, Teng Li, and JianFeng Ma},
  month     = feb,
  year      = {2025}
}
```
