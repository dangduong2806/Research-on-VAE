"""So sánh nhiều cấu hình Vanilla VAE."""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from analysis.common import load_trained_model
from analysis.latent_distribution import collect_latents
from analysis.random_sampling import save_samples
from analysis.reconstruction import denormalize
from src.data.dataloader import build_dataloaders
from src.engine.evaluator import evaluate
from src.losses.kl_divergence import KLDivergenceLoss
from src.losses.reconstruction_loss import ReconstructionLoss


def load_experiments(
    manifest_path: str,
) -> list[dict]:
    """Đọc danh sách thí nghiệm."""

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = yaml.safe_load(file)

    experiments = manifest.get(
        "experiments",
        [],
    )

    if not experiments:
        raise ValueError(
            "Không có experiment trong manifest."
        )

    return experiments


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def evaluate_experiment(
    experiment: dict,
    output_dir: Path,
    num_samples: int,
    seed: int,
) -> dict:
    """Đánh giá một config và checkpoint."""

    name = experiment["name"]
    config_path = experiment["config"]
    checkpoint_path = experiment["checkpoint"]

    config, model, _, device = (
        load_trained_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
    )

    loaders = build_dataloaders(
        config.data,
        seed=config.seed,
    )

    reconstruction_loss_fn = ReconstructionLoss(
        loss_type=config.loss.get(
            "reconstruction",
            "mse",
        ),
        reduction="mean",
    )

    kl_loss_fn = KLDivergenceLoss(
        reduction="mean"
    )

    beta = config.loss.get(
        "beta",
        1.0,
    )

    metrics = evaluate(
        model=model,
        dataloader=loaders.test,
        reconstruction_loss_fn=(
            reconstruction_loss_fn
        ),
        kl_loss_fn=kl_loss_fn,
        device=device,
        beta=beta,
    )

    mu, std, _ = collect_latents(
        model=model,
        dataloader=loaders.test,
        device=device,
    )

    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        samples = model.generate(
            num_samples=num_samples,
            device=device,
        )

    samples = denormalize(
        samples,
        config.data.get(
            "normalization",
            "minus_one_to_one",
        ),
    )

    sample_path = (
        output_dir
        / "samples"
        / f"{name}.png"
    )

    save_samples(
        images=samples,
        output_path=str(sample_path),
        columns=min(8, num_samples),
    )

    return {
        "name": name,
        "config": config_path,
        "checkpoint": checkpoint_path,
        "latent_dim": config.model["latent_dim"],
        "beta": beta,
        "parameters": count_parameters(model),
        "test_samples": metrics["samples"],
        "total_loss": metrics["total"],
        "reconstruction_loss": (
            metrics["reconstruction"]
        ),
        "kl_loss": metrics["kl"],
        "mu_mean": mu.mean().item(),
        "mu_std": mu.std(
            unbiased=False
        ).item(),
        "posterior_std_mean": (
            std.mean().item()
        ),
        "sample_image": str(sample_path),
    }


def save_csv(
    results: list[dict],
    output_path: Path,
) -> None:
    """Lưu bảng ablation dạng CSV."""

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


def run_ablation(
    manifest_path: str,
    output_dir: str,
    num_samples: int,
    seed: int,
) -> None:
    experiments = load_experiments(
        manifest_path
    )

    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for experiment in experiments:
        print(
            f"Evaluating: "
            f"{experiment['name']}"
        )

        result = evaluate_experiment(
            experiment=experiment,
            output_dir=output,
            num_samples=num_samples,
            seed=seed,
        )

        results.append(result)

        print(
            f"  total="
            f"{result['total_loss']:.6f} | "
            f"recon="
            f"{result['reconstruction_loss']:.6f} | "
            f"kl="
            f"{result['kl_loss']:.6f}"
        )

    csv_path = output / "ablation_results.csv"
    json_path = output / "ablation_results.json"

    save_csv(
        results,
        csv_path,
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study cho Vanilla VAE"
        )
    )

    parser.add_argument(
        "--manifest",
        default="configs/ablation.yaml",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/ablation",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run_ablation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()