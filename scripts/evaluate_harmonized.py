"""Evaluate a trained harmonized checkpoint on the shared test episodes, without retraining.

Reloads `results/<run_name>/best_model.pt` (or `--checkpoint`), rebuilds the backbone
named in the config (ResNet-12 or Conv-4) and runs `common.protocol.evaluate`: the
1000 frozen episodes, with and without CL2N. The scores are written to
`results/<run_name>/test_shared.json`.

Examples:
    python scripts/evaluate_harmonized.py --config configs/harmonized/prwn_resnet12.yaml
    python scripts/evaluate_harmonized.py --config configs/harmonized/prwn_conv4_pn.yaml --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common.protocol import evaluate, print_results  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from train_harmonized import build_backbone  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="config of the trained run")
    p.add_argument("--checkpoint", type=Path, help="default: results/<run_name>/best_model.pt")
    p.add_argument("--data-root", type=Path, help="overrides data_root")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="overrides device")
    args = p.parse_args()

    overrides = {}
    if args.data_root:
        overrides["data_root"] = str(args.data_root.resolve())
    if args.device:
        overrides["device"] = args.device
    config = load_config(args.config, overrides)
    device = config.resolve_device()

    checkpoint = args.checkpoint or config.run_dir / "best_model.pt"
    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    # The checkpoint knows which backbone it holds (e.g. a run started with --model conv4).
    trained_model = state.get("config", {}).get("model", config.model)
    if trained_model != config.model:
        print(f"Checkpoint holds a {trained_model} backbone (config says {config.model}): using {trained_model}.")
        config.model = trained_model
    model = build_backbone(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    results = evaluate(model, config.data_root, device, config.extract_batch_size)
    print_results(results, f"{config.run_name} (checkpoint from step {state.get('step', '?')})")

    out = config.run_dir / "test_shared.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(checkpoint), "step": state.get("step"), "test": results}, handle, indent=2)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
