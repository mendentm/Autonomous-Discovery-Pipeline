import numpy as np
import os
import re

class GameOfLifeEngine:
    def __init__(self, height=64, width=64):
        self.height = height
        self.width = width

    def load_rle(self, file_path):
        """
        Parses a .rle file and returns a numpy grid centered in the engine's dimensions.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        # Filter comments and header
        data_lines = [line.strip() for line in lines if not line.startswith('#')]
        header = None
        pattern_data = []

        for line in data_lines:
            if line.startswith('x'):
                header = line
                continue
            pattern_data.append(line)
        
        if not header:
            return None

        # Parse header for dimensions (optional check, but we rely on our fixed grid)
        full_pattern = "".join(pattern_data).replace(' ', '')
        
        # Decode RLE
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Parse logic
        rows = full_pattern.split('$')
        current_y = 0
        
        # Center the pattern
        # We start drawing from an offset to center it roughly
        start_y = self.height // 4 
        start_x = self.width // 4

        current_y = start_y
        
        for row in rows:
            current_x = start_x
            if '!' in row:
                row = row.split('!')[0]
            
            # Find all runs like '3o' or 'b'
            matches = re.findall(r'(\d*)([ob])', row)
            
            for count_str, tag in matches:
                count = int(count_str) if count_str else 1
                
                if tag == 'o': # Alive
                    # Boundary check
                    if current_y < self.height and current_x + count < self.width:
                        grid[current_y, current_x:current_x+count] = 1.0
                    elif current_y < self.height and current_x < self.width:
                         # Handle edge clipping
                         remaining = self.width - current_x
                         grid[current_y, current_x:current_x+remaining] = 1.0
                
                current_x += count
            
            # Handle empty rows if encoded like '3$' (though typically split by $)
            # Simple parser assumes $ is newline. 
            current_y += 1
            if current_y >= self.height:
                break
                
        return grid

    def run_simulation(self, initial_grid, generations=200):
        """
        Runs the simulation using NumPy convolution operations for speed.
        Returns:
            - final_grid
            - history (list of population counts)
            - activity_score (metric for how 'interesting' it was)
        """
        grid = initial_grid.copy()
        history = []
        activity = 0
        
        # 3x3 kernel for counting neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])

        for _ in range(generations):
            pop = np.sum(grid)
            history.append(pop)
            
            if pop == 0:
                break

            # Count neighbors using convolution (padded to handle edges as dead)
            # We manually implement a fast neighbor count for pure numpy without scipy
            padded = np.pad(grid, 1, mode='constant')
            # Vectorized neighbor counting
            # Shift operations
            nbrs = (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
                    padded[1:-1, :-2]                + padded[1:-1, 2:] +
                    padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:])
            
            # Apply Game of Life Rules
            # 1. Survival: Cell is 1 AND (2 or 3 neighbors)
            survive = ((grid == 1) & ((nbrs == 2) | (nbrs == 3)))
            # 2. Birth: Cell is 0 AND (3 neighbors)
            birth = ((grid == 0) & (nbrs == 3))
            
            new_grid = (survive | birth).astype(np.float32)
            
            # Calculate activity (hamming distance between frames)
            diff = np.abs(new_grid - grid)
            activity += np.sum(diff)
            
            grid = new_grid
            
        return grid, history, activity

    def evaluate_pattern(self, initial_grid, generations=200):
        """
        Determines if a pattern is 'Incredible'.
        Metrics:
        1. Growth: Max population / Initial population
        2. Longevity: How long it stays active
        3. Entropy: Does it just turn into a block (low entropy) or chaos?
        """
        final_grid, history, total_activity = self.run_simulation(initial_grid, generations)
        
        if len(history) == 0 or history[0] == 0:
            return 0.0, {}

        initial_pop = history[0]
        max_pop = max(history)
        final_pop = history[-1]
        
        # 1. Growth Score
        growth_ratio = max_pop / (initial_pop + 1)
        
        # 2. Survival Score (did it die out?)
        survival_ratio = len(history) / generations
        if final_pop == 0:
            survival_ratio = 0.1 # Penalty for extinction
            
        # 3. Dynamic Score (Activity per generation normalized by population)
        # Prevents rewarding static blocks (high pop, 0 activity)
        avg_activity = total_activity / generations
        
        # Composite Score
        # We want high growth, sustained activity.
        score = (growth_ratio * 2.0) + (avg_activity * 0.05)
        
        if final_pop > 0 and survival_ratio == 1.0:
            score *= 1.5 # Bonus for surviving the whole run
            
        stats = {
            "initial_pop": initial_pop,
            "final_pop": final_pop,
            "max_pop": max_pop,
            "growth": growth_ratio,
            "activity": total_activity
        }
        
        return score, stats