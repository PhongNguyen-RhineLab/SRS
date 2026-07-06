"""
Fetch the sequence file and pre-trained ICSRec checkpoint for a dataset.

Both come from the official ICSRec repository
(https://github.com/QinHsiu/ICSRec) and are downloaded straight from GitHub
raw URLs -- no manual copying needed. Example:

    python fetch_data.py --dataset sports
    python fetch_data.py --dataset toys
    python fetch_data.py --all

After fetching, the usual pipeline order for a new dataset is:

    python build_index.py      --dataset sports
    python pretrain_encoder.py --dataset sports
    python train_rl.py         --dataset sports
    python run_baselines.py    --dataset sports
    python run_test_eval.py    --dataset sports
    python theory_validation.py --dataset sports

Note: the checkpoint files are ~5 MB each, the txt files 0.6-3.5 MB.
"""

import os

import requests
from tqdm import tqdm

from config import DATASET_REGISTRY
from cli import make_parser, resolve_dataset

RAW_BASE = "https://raw.githubusercontent.com/QinHsiu/ICSRec/main"
CKPT_DIR = "icsrec_ckpts"


def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  already present, skipping: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    tmp = dest + ".part"
    with open(tmp, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    os.replace(tmp, dest)


def fetch(dataset: str):
    spec = DATASET_REGISTRY[dataset]
    print(f"[{dataset}] fetching data + checkpoint...")
    _download(f"{RAW_BASE}/data/{spec['txt']}",
              os.path.join(spec["data_dir"], spec["txt"]))
    _download(f"{RAW_BASE}/src/output/{spec['ckpt']}",
              os.path.join(CKPT_DIR, spec["ckpt"]))
    print(f"[{dataset}] done.\n")


def main():
    p = make_parser("Download ICSRec sequence file + checkpoint for a dataset.")
    p.add_argument("--all", action="store_true",
                   help="Fetch every dataset in the registry.")
    args = p.parse_args()

    if args.all:
        for d in DATASET_REGISTRY:
            fetch(d)
    else:
        fetch(resolve_dataset(args.dataset))


if __name__ == "__main__":
    main()
