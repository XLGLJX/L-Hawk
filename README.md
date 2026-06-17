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
The detailed three dateset (including KITTI, BDD100K, and ImageNet) building is available in [README](./datasets/README.md).

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

# CA against YOLOv5 on KITTI evaluation images.
python demo.py --cfg configs/CA.yaml --attack_type CA --det yolov5 --target "stop sign"

# Quick smoke-sized run. Use this to validate paths before full experiments.
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --epochs 1 --train-batch 1 --eval-batch 1 --repeat 1

# Detector smoke-sized run using COCO as the evaluation source.
# Use KITTI/BDD100K for the normal detector evaluation once those datasets are linked.
python demo.py --cfg configs/CA.yaml --attack_type CA --det yolov5 --target "stop sign" \
  --eval-dataset coco --epochs 1 --train-batch 1 --eval-batch 1 --repeat 1
```

Each run writes its generated patches, `run_config.json`, and `metrics.csv` under `exp/`.
The metrics file records `ASR`, `No_triggered`, and `Triggered` for each epoch.

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
`--trigger-selection epoch-search`. At the beginning of each epoch, the runner
evaluates the current patch against the generated trigger candidates and trains
that epoch with the best candidate according to `--trigger-search-metric`.

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det vgg16 --target 920 \
  --trigger-source laser \
  --laser-power 10:70:10 \
  --trigger-selection epoch-search \
  --trigger-search-metric ASR \
  --trigger-search-batch 8
```

Patch-size sweeps can be run by changing `--patch-size`, for example:

```bash
python demo.py --cfg configs/TA-C.yaml --attack_type TA-C --det res50 --target 920 \
  --trigger-source laser \
  --patch-size 64 \
  --epochs 20 --train-batch 50 --eval-batch 800 --repeat 20
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
