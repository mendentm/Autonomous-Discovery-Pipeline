"""
Closed-loop latent-space evolutionary search for Game of Life patterns.

A frozen VAE (trained by main_pipeline.py / benchmark.py) serves as a learned
prior over GoL-like patterns. A (mu, lambda) Evolution Strategy searches
directly in the 128-dim latent space, using the GoL simulator's score as the
fitness signal. This closes the feedback loop that the original pipeline was
missing: instead of uniform prior sampling, selection pressure shapes which
regions of latent space get explored.

Why search in latent space instead of pixel space:
    - Pixel-space mutations almost always break patterns (flipping random cells
      in a glider produces noise). The VAE's latent space is continuous and
      smooth by construction, so small latent perturbations map to *related*
      patterns, not random ones.
    - The VAE has already concentrated probability mass on "GoL-looking"
      regions. Starting search from the prior keeps the population in that
      manifold instead of wandering into noise.

Reports a best-so-far curve that should climb monotonically, proving the
feedback loop is doing something real.
"""
import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from game_of_life_engine import GameOfLifeEngine
from pattern_generator_model import ModelTrainer
import main_pipeline  # reuse save_pattern_image / save_rle

SEED = 1337
DATA_DIR = "training_patterns"
MODEL_PATH = "life_generator_vae.pth"
OUTPUT_DIR = "latent_search_results"
GRID_SIZE = 64
GENERATIONS_SIM = 200   # GoL ticks per candidate
POP_SIZE = 64           # lambda
ELITE_SIZE = 16         # mu (top 25%)
N_GENERATIONS = 30      # ES outer loop
INITIAL_SIGMA = 1.0     # matches VAE prior N(0, I)
SIGMA_MIN = 0.4         # floor to prevent premature convergence
SIGMA_MAX = 2.0
IMMIGRATION_FRAC = 0.15 # fraction of each generation drawn fresh from N(0, I)
LATENT_DIM = 128


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_population(trainer, engine, z_pop):
    """Decode each latent vector and score the resulting grid."""
    with torch.no_grad():
        z_t = torch.from_numpy(z_pop).float().to(trainer.device)
        decoded = trainer.model.decode(z_t).cpu().numpy()
        grids = (decoded.squeeze(1) > 0.5).astype(np.float32)

    scores = np.zeros(len(grids), dtype=np.float32)
    growths = np.zeros(len(grids), dtype=np.float32)
    for i, g in enumerate(grids):
        score, stats = engine.evaluate_pattern(g, generations=GENERATIONS_SIM)
        scores[i] = score
        growths[i] = stats.get("growth", 0.0)
    return scores, growths, grids


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    engine = GameOfLifeEngine(height=GRID_SIZE, width=GRID_SIZE)

    # Load the trained VAE (must exist - run main_pipeline.py or benchmark.py first)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"No trained model at {MODEL_PATH}. "
            "Run 'python main_pipeline.py' or 'python benchmark.py after' first."
        )
    trainer = ModelTrainer(device="cpu")
    trainer.load_model(MODEL_PATH)
    print(f"Loaded VAE from {MODEL_PATH}")

    # Initialize the population from the VAE's prior N(0, I)
    population = np.random.randn(POP_SIZE, LATENT_DIM).astype(np.float32)
    sigma = INITIAL_SIGMA

    # Tracking
    best_so_far_score = -np.inf
    best_so_far_growth = 0.0
    best_so_far_grid = None
    best_so_far_z = None
    history = []

    t_start = time.time()
    for gen in range(N_GENERATIONS):
        t0 = time.time()
        scores, growths, grids = evaluate_population(trainer, engine, population)

        # Stats for this generation
        gen_best_idx = int(np.argmax(scores))
        gen_best_score = float(scores[gen_best_idx])
        gen_best_growth = float(growths[gen_best_idx])
        gen_mean_score = float(scores.mean())
        gen_median_score = float(np.median(scores))

        # Update global best
        if gen_best_score > best_so_far_score:
            best_so_far_score = gen_best_score
            best_so_far_growth = gen_best_growth
            best_so_far_grid = grids[gen_best_idx].copy()
            best_so_far_z = population[gen_best_idx].copy()

        # Truncation selection: keep top ELITE_SIZE.
        elite_idx = np.argsort(scores)[-ELITE_SIZE:]
        elites = population[elite_idx]
        elite_mean = elites.mean(axis=0)

        # Recombine. The GoL fitness landscape is bimodal: both methuselahs
        # (high growth, ~60 score) and oscillators (growth=1, ~25 score from
        # activity) coexist. A pure elite-mean centroid gets pulled toward
        # the larger oscillator basin. Anchoring the centroid on the all-time
        # best z and only blending in the elite mean keeps search in the
        # methuselah basin once one is found. This is closer to (1+lambda)-ES
        # with elite-mean drift as a tiebreaker.
        if best_so_far_z is not None:
            centroid = 0.7 * best_so_far_z + 0.3 * elite_mean
        else:
            centroid = elite_mean

        # Adaptive sigma: 1/5 success rule.
        # If more than 1/5 of offspring beat the current best, we're in a
        # productive region -- expand. Otherwise contract to refine.
        improved_fraction = float(np.mean(scores > best_so_far_score - 1e-6))
        if improved_fraction > 0.2:
            sigma = min(sigma * 1.15, SIGMA_MAX)
        else:
            sigma = max(sigma / 1.15, SIGMA_MIN)

        # Generate next population: mutate around the elite centroid,
        # but reserve a fraction for fresh samples from the prior. This
        # immigration prevents premature convergence into a single basin
        # by continuously re-seeding the search with unexplored regions.
        n_immigrants = int(POP_SIZE * IMMIGRATION_FRAC)
        n_offspring = POP_SIZE - n_immigrants
        offspring = centroid[None, :] + sigma * np.random.randn(n_offspring, LATENT_DIM).astype(np.float32)
        immigrants = np.random.randn(n_immigrants, LATENT_DIM).astype(np.float32)
        population = np.concatenate([offspring, immigrants], axis=0)

        dt = time.time() - t0
        history.append({
            "gen": gen,
            "best_score": gen_best_score,
            "best_growth": gen_best_growth,
            "mean_score": gen_mean_score,
            "median_score": gen_median_score,
            "best_so_far_score": float(best_so_far_score),
            "best_so_far_growth": float(best_so_far_growth),
            "sigma": float(sigma),
            "elapsed_s": round(dt, 2),
        })
        print(
            f"Gen {gen+1:2d}/{N_GENERATIONS}  "
            f"best={gen_best_score:7.2f} (growth {gen_best_growth:5.2f}x)  "
            f"mean={gen_mean_score:7.2f}  "
            f"best_so_far={best_so_far_score:7.2f} ({best_so_far_growth:5.2f}x)  "
            f"sigma={sigma:.2f}  "
            f"{dt:.1f}s"
        )

    total_t = time.time() - t_start
    print(f"\nSearch complete in {total_t:.1f}s")
    print(f"Best score:  {best_so_far_score:.2f}")
    print(f"Best growth: {best_so_far_growth:.2f}x")

    # Save artifacts
    with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
        json.dump(
            {
                "config": {
                    "pop_size": POP_SIZE,
                    "elite_size": ELITE_SIZE,
                    "n_generations": N_GENERATIONS,
                    "sim_generations": GENERATIONS_SIM,
                    "initial_sigma": INITIAL_SIGMA,
                    "latent_dim": LATENT_DIM,
                    "seed": SEED,
                },
                "total_time_s": round(total_t, 2),
                "best_score": float(best_so_far_score),
                "best_growth": float(best_so_far_growth),
                "history": history,
            },
            f,
            indent=2,
        )

    # Save best seed as image + rle
    best_name = f"best_score_{best_so_far_score:.1f}_growth_{best_so_far_growth:.1f}"
    main_pipeline.save_pattern_image(
        best_so_far_grid,
        os.path.join(OUTPUT_DIR, f"{best_name}.png"),
        best_so_far_score,
    )
    main_pipeline.save_rle(best_so_far_grid, os.path.join(OUTPUT_DIR, f"{best_name}.rle"))

    # Convergence plot
    gens = np.array([h["gen"] for h in history])
    best_curve = np.array([h["best_so_far_score"] for h in history])
    mean_curve = np.array([h["mean_score"] for h in history])
    gen_best = np.array([h["best_score"] for h in history])

    plt.figure(figsize=(10, 6))
    plt.plot(gens + 1, best_curve, "o-", label="best-so-far", linewidth=2)
    plt.plot(gens + 1, gen_best, "s-", label="generation best", alpha=0.6)
    plt.plot(gens + 1, mean_curve, "^-", label="generation mean", alpha=0.6)
    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.title(
        f"Latent-space ES: {POP_SIZE}x{N_GENERATIONS}  "
        f"best={best_so_far_score:.1f}  growth={best_so_far_growth:.1f}x"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "convergence.png"), dpi=120)
    plt.close()

    print(f"Artifacts written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
