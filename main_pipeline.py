import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from game_of_life_engine import GameOfLifeEngine
from pattern_generator_model import ModelTrainer
import torch

# --- CONFIGURATION ---
DATA_DIR = "training_patterns"
OUTPUT_DIR = "incredible_discoveries"
MODEL_PATH = "life_generator_vae.pth"
GRID_SIZE = 64
GENERATIONS = 200
EPOCHS = 100
NUM_DREAMS = 500 # How many patterns to generate and test
# ---------------------

def ensure_directories():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please place .rle files here.")
        # Create a dummy glider gun RLE if empty so it runs out of the box
        create_dummy_data()
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def create_dummy_data():
    """Creates a simple RLE file if none exist so the pipeline doesn't crash."""
    glider_rle = """#N Glider
x = 3, y = 3
bob$2bo$3o!"""
    with open(os.path.join(DATA_DIR, "glider.rle"), "w") as f:
        f.write(glider_rle)
    print("Added default 'glider.rle' to training data.")

def load_training_data(engine):
    rle_files = glob.glob(os.path.join(DATA_DIR, "*.rle"))
    print(f"Found {len(rle_files)} .rle files.")
    
    grids = []
    for f in rle_files:
        grid = engine.load_rle(f)
        if grid is not None:
            grids.append(grid)
            # Data Augmentation: Rotate and Flip to increase dataset size
            grids.append(np.rot90(grid, 1))
            grids.append(np.flipud(grid))
    
    if not grids:
        print("Warning: No valid patterns found. Generating random noise for training.")
        # Fallback: Random noise training (The model will learn to generate noise, 
        # but the evolutionary filter will still find good stuff)
        for _ in range(100):
            grids.append(np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2]).astype(np.float32))
            
    return np.array(grids)

def save_pattern_image(grid, filename, score):
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='binary')
    plt.title(f"Score: {score:.2f}")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def save_rle(grid, filename):
    """Simple RLE encoder for saving discoveries."""
    rows, cols = grid.shape
    with open(filename, 'w') as f:
        f.write(f"x = {cols}, y = {rows}\n")
        for i in range(rows):
            line = ""
            run_count = 0
            last_val = None
            for j in range(cols):
                val = grid[i, j]
                tag = 'o' if val > 0.5 else 'b'
                
                if tag == last_val:
                    run_count += 1
                else:
                    if last_val:
                        line += (str(run_count) if run_count > 1 else "") + last_val
                    last_val = tag
                    run_count = 1
            line += (str(run_count) if run_count > 1 else "") + last_val + "$"
            f.write(line + "\n")
        f.write("!")

def main():
    print("--- AUTOMOUS LIFE PIPELINE INITIATED ---")
    ensure_directories()
    
    # 1. Initialize Engine
    engine = GameOfLifeEngine(height=GRID_SIZE, width=GRID_SIZE)
    
    # 2. Load Data
    training_grids = load_training_data(engine)
    print(f"Loaded {len(training_grids)} training samples (including augmentations).")
    
    # 3. Initialize and Train Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using computation device: {device}")
    
    trainer = ModelTrainer(device=device)
    
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        trainer.load_model(MODEL_PATH)
    else:
        print("Training new model...")
        trainer.train(training_grids, epochs=EPOCHS)
        trainer.save_model(MODEL_PATH)
        print("Model saved.")
    
    # 4. The Dream Loop (Search Phase)
    print(f"\n--- DREAMING PHASE: Generating {NUM_DREAMS} candidates ---")
    seeds = trainer.generate_seeds(NUM_DREAMS)
    
    best_patterns = []
    
    for i, seed in enumerate(seeds):
        if i % 50 == 0:
            print(f"Simulating pattern {i}/{NUM_DREAMS}...")
            
        score, stats = engine.evaluate_pattern(seed, generations=GENERATIONS)
        
        # Threshold for "Incredible"
        # We look for patterns that grew significantly or had high activity
        if score > 5.0 and stats['final_pop'] > 0:
            print(f"  -> DISCOVERY! ID:{i} Score:{score:.2f} Growth:{stats['growth']:.2f}")
            best_patterns.append((score, seed, stats))
            
    # 5. Save Results
    best_patterns.sort(key=lambda x: x[0], reverse=True)
    top_discoveries = best_patterns[:10] # Keep top 10
    
    print(f"\n--- RESULTS: Found {len(best_patterns)} interesting candidates ---")
    
    for rank, (score, grid, stats) in enumerate(top_discoveries):
        name = f"rank_{rank+1}_score_{score:.1f}_growth_{stats['growth']:.1f}"
        
        # Save Image
        save_pattern_image(grid, os.path.join(OUTPUT_DIR, f"{name}.png"), score)
        
        # Save RLE
        save_rle(grid, os.path.join(OUTPUT_DIR, f"{name}.rle"))
        
        print(f"Saved {name}")
        
    print("\nPipeline complete. Check the 'incredible_discoveries' folder.")

if __name__ == "__main__":
    main()