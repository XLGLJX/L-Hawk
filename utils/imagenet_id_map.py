"""ImageNet 类别编号映射工具
=================================

- 论文/标准 ImageNet 1-indexed 编号 (1..1000)  ↔  WordNet synset ID (n01440764 ...)
- WordNet synset ID  ↔  torchvision ImageFolder 字典序 0-indexed 编号
- 单文件即可从 WordNet synset 集合反推任意映射

使用：
    python imagenet_id_map.py --synset n02951358
    python imagenet_id_map.py --standard 920
    python imagenet_id_map.py --folder 472
"""
import argparse
import sys
from pathlib import Path

# 标准 ILSVRC2012 synset 列表（按 1-indexed 编号 1..1000 顺序）
STANDARD_SYNSETS_FILE = Path(__file__).parent / "imagenet_standard_synsets.txt"


def load_standard_synsets(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def synset_to_standard(synset: str, std_synsets):
    """synset -> 1-indexed 标准 ImageNet 编号"""
    if std_synsets and synset in std_synsets:
        return std_synsets.index(synset) + 1
    return None


def synset_to_folder(synset: str, folder_classes):
    """synset -> ImageFolder 字典序 0-indexed 索引"""
    if synset in folder_classes:
        return folder_classes.index(synset)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synset", help="WordNet synset ID (e.g. n02951358)")
    ap.add_argument("--standard", type=int, help="1-indexed ImageNet class (1..1000)")
    ap.add_argument("--folder", type=int, help="0-indexed ImageFolder dict order (0..999)")
    ap.add_argument("--data-root", default="datasets/ImageNet",
                    help="ImageNet validation root with synset subdirs")
    args = ap.parse_args()

    import torchvision as tv
    ds = tv.datasets.ImageFolder(args.data_root)
    folder_classes = ds.classes  # 0-indexed dict order
    std_synsets = load_standard_synsets(STANDARD_SYNSETS_FILE)

    if args.synset:
        std = synset_to_standard(args.synset, std_synsets)
        fld = synset_to_folder(args.synset, folder_classes)
        print(f"synset   = {args.synset}")
        print(f"standard = {std}  (1-indexed, 论文里使用的编号；缺少标准表时为 None)")
        print(f"folder   = {fld}  (ImageFolder 字典序 0-indexed)")
    elif args.standard is not None:
        if std_synsets is None:
            sys.exit(f"缺少 {STANDARD_SYNSETS_FILE}，无法从 standard 编号反查 synset")
        if not (1 <= args.standard <= 1000):
            sys.exit("standard 必须在 1..1000")
        synset = std_synsets[args.standard - 1]
        fld = synset_to_folder(synset, folder_classes)
        print(f"standard = {args.standard}")
        print(f"synset   = {synset}")
        print(f"folder   = {fld}  (ImageFolder 字典序 0-indexed)")
    elif args.folder is not None:
        if not (0 <= args.folder < len(folder_classes)):
            sys.exit(f"folder 必须在 0..{len(folder_classes)-1}")
        synset = folder_classes[args.folder]
        std = synset_to_standard(synset, std_synsets)
        print(f"folder   = {args.folder}")
        print(f"synset   = {synset}")
        print(f"standard = {std}  (论文/标准 1-indexed 编号；缺少标准表时为 None)")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
