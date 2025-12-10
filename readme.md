# Game of Life - Autonomous Discovery Pipeline

This project is an autonomous pipeline that uses a Variational Autoencoder (VAE) to discover new and interesting patterns in Conway's Game of Life.

It works by:
1.  **Learning**: Training a neural network on a collection of existing Game of Life patterns (`.rle` files).
2.  **Dreaming**: Using the trained model to generate thousands of new, unique starting patterns.
3.  **Simulating**: Running each "dreamed" pattern through a Game of Life engine for a set number of generations.
4.  **Discovering**: Evaluating the results of the simulation to find patterns that are "interesting" (e.g., they grow, have high population density, or exhibit other complex behaviors).
5.  **Saving**: Automatically saving the most interesting discoveries as both images (`.png`) and reusable `.rle` files.

## How It Works

The core of the project is `main_pipeline.py`, which orchestrates the entire process.

1.  **Data Loading**: The pipeline first looks in the `training_patterns/` directory for `.rle` files. It loads these patterns and augments the dataset by rotating and flipping them. If no patterns are found, it creates a default `glider.rle` to ensure the program can run out-of-the-box.

2.  **Model Training**: A VAE model is trained on the loaded patterns. The model learns a compressed "latent space" representation of what makes a valid Game of Life pattern. If a pre-trained model (`life_generator_vae.pth`) exists, this step is skipped, and the existing model is loaded.

3.  **Generation (The "Dream Loop")**: The trained model is used to generate a large number of new patterns (`seeds`). This is done by sampling points from the learned latent space and decoding them back into a grid format.

4.  **Evaluation**: Each generated seed is passed to the `GameOfLifeEngine`. The engine simulates the pattern's evolution and calculates a `score` based on its growth, final population, and activity.

5.  **Saving Results**: Patterns that exceed a certain score threshold are considered "discoveries." The top 10 discoveries are saved in the `incredible_discoveries/` directory, sorted by their score. Each discovery includes a `.png` image of the starting grid and a `.rle` file for use in other Game of Life software.

## Requirements

This project requires Python 3 and the following libraries:

*   `numpy`
*   `matplotlib`
*   `torch`

You can install them using pip:
```bash
pip install numpy matplotlib torch
```

If you have an NVIDIA GPU, it is highly recommended to install the CUDA-enabled version of PyTorch for significantly faster model training.

## Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **(Optional) Add Training Data:**
    Place any `.rle` (Run-Length Encoded) files of Game of Life patterns you want the model to learn from into the `training_patterns/` directory. The script will create this directory for you if it doesn't exist and add a sample `glider.rle`. More varied and complex input patterns will lead to more interesting generated results.

3.  **Run the Pipeline:**
    Execute the main script from your terminal:
    ```bash
    python main_pipeline.py
    ```

    The script will:
    - Create necessary directories (`training_patterns`, `incredible_discoveries`).
    - Load or train the model (training will happen on the first run).
    - Begin the "Dreaming Phase" to generate and test new patterns.
    - Save the best results to the `incredible_discoveries/` folder.

## Configuration

You can easily tweak the pipeline's parameters by editing the `CONFIGURATION` section at the top of `main_pipeline.py`.

```python
# --- CONFIGURATION ---
DATA_DIR = "training_patterns"
OUTPUT_DIR = "incredible_discoveries"
MODEL_PATH = "life_generator_vae.pth"
GRID_SIZE = 64          # The width and height of the simulation grid.
GENERATIONS = 200      # How many steps to simulate each pattern for.
EPOCHS = 100            # Number of training epochs for the model.
NUM_DREAMS = 500        # How many patterns to generate and test.
# ---------------------
```

---

*This project structure is designed for experimentation. Try changing the scoring function in `game_of_life_engine.py` or the model architecture in `pattern_generator_model.py` to guide the discovery process toward different kinds of patterns!*