# vrp-genetic-algorithm

Genetic algorithm implementation to solve the Vehicle Routing Problem (VRP), aiming to minimize the total distance traveled from a central depot. It uses classic evolutionary operators such as tournament selection, balanced crossover, move mutation, and 2-opt local optimization. Includes result visualization and performance analysis.

---

## Key Features

- Reads instances in `.dat` format (TSPLIB-style).
- Represents solutions as lists of routes per vehicle.
- Evolutionary operators:
  - Tournament selection (k = 3)
  - Balanced crossover without duplicates
  - Move-type mutation
  - 2-opt local optimization per route
- Penalties for omitted or repeated cities.
- Visualization of:
  - Cost evolution per generation (`mejor_costos.png`)
  - Assigned routes (`rutas.png`)
- Export of results to text and CSV files.

---

## Included Files

- `codigo_vrp.py`: Source code of the algorithm.
- `A045-03f.dat`: Problem instance from VRP-REP.
- `final/`: Automatically generated folder containing results.

---

## How to Run

Make sure you have Python 3 installed, then run:

```bash
pip install numpy matplotlib pandas tqdm
python codigo_vrp.py
