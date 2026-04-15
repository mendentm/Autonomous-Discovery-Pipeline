"""
Deterministic benchmark for the GoL discovery pipeline.
Runs the same training -> dream -> evaluate steps as main_pipeline.py
but with fixed RNG seeds and structured metric output, so we can
compare before/after a fix on apples-to-apples terms.
"""
import os
import glob
import json
import time
import numpy as np
import torch

from game_of_life_engine import GameOfLifeEngine
from pattern_generator_model import ModelTrainer

SEED = 1337
DATA_DIR = "training_patterns"
GRID_SIZE = 64
GENERATIONS = 400
EPOCHS = 150
NUM_DREAMS = 3000
DISCOVERY_THRESHOLD = 5.0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_training(engine):
    rle_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.rle")))
    grids = []
    loaded_names = []
    for f in rle_files:
        g = engine.load_rle(f)
        if g is None:
            continue
        loaded_names.append((os.path.basename(f), int(g.sum())))
        grids.append(g)
        grids.append(np.rot90(g, 1))
        grids.append(np.flipud(g))
    return np.array(grids), loaded_names


def seed_stats(seeds):
    """Mean alive-cell density and fraction of empty seeds."""
    densities = seeds.reshape(len(seeds), -1).mean(axis=1)
    return {
        "mean_density": float(densities.mean()),
        "min_density": float(densities.min()),
        "max_density": float(densities.max()),
        "empty_fraction": float((densities == 0).mean()),
    }


def run(label):
    set_seed(SEED)
    engine = GameOfLifeEngine(height=GRID_SIZE, width=GRID_SIZE)

    training_grids, loaded = load_training(engine)
    print(f"[{label}] Loaded {len(loaded)} RLE files -> {len(training_grids)} samples (with augmentation)")
    for name, pop in loaded:
        print(f"    {name}: {pop} alive cells parsed")

    trainer = ModelTrainer(device="cpu")

    t0 = time.time()
    trainer.train(training_grids, epochs=EPOCHS)
    train_time = time.time() - t0
    print(f"[{label}] Training: {train_time:.1f}s")

    set_seed(SEED + 1)
    t0 = time.time()
    seeds = trainer.generate_seeds(NUM_DREAMS)
    gen_time = time.time() - t0

    s_stats = seed_stats(seeds)
    print(f"[{label}] Generated {len(seeds)} seeds in {gen_time:.2f}s")
    print(f"    density mean={s_stats['mean_density']:.4f} "
          f"min={s_stats['min_density']:.4f} max={s_stats['max_density']:.4f} "
          f"empty_frac={s_stats['empty_fraction']:.2f}")

    scores = []
    discoveries = 0
    top_growth = 0.0
    t0 = time.time()
    for seed in seeds:
        score, stats = engine.evaluate_pattern(seed, generations=GENERATIONS)
        scores.append(score)
        if score > DISCOVERY_THRESHOLD and stats.get("final_pop", 0) > 0:
            discoveries += 1
            if stats["growth"] > top_growth:
                top_growth = stats["growth"]
    eval_time = time.time() - t0

    scores = np.array(scores)
    top10 = np.sort(scores)[-10:][::-1]
    print(f"[{label}] Evaluated {len(seeds)} in {eval_time:.1f}s")
    print(f"    score mean={scores.mean():.3f} max={scores.max():.3f} median={np.median(scores):.3f}")
    print(f"    top-10 scores: {[f'{s:.2f}' for s in top10]}")
    print(f"    discoveries (score>{DISCOVERY_THRESHOLD}): {discoveries}/{len(seeds)}")
    print(f"    best growth ratio seen: {top_growth:.2f}")

    metrics = {
        "label": label,
        "training_samples": int(len(training_grids)),
        "rle_files_parsed": int(len(loaded)),
        "rle_alive_cells_per_file": {n: p for n, p in loaded},
        "train_time_s": round(train_time, 2),
        "gen_time_s": round(gen_time, 2),
        "eval_time_s": round(eval_time, 2),
        "seed_density_mean": s_stats["mean_density"],
        "seed_empty_fraction": s_stats["empty_fraction"],
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "score_median": float(np.median(scores)),
        "top10_scores": [float(s) for s in top10],
        "discoveries": int(discoveries),
        "best_growth_ratio": float(top_growth),
    }
    return metrics


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    m = run(label)
    out = f"metrics_{label}.json"
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[{label}] metrics written to {out}")
