# Cell states
ALIVE = 1
DEAD = 0
DIVIDING = 2
SENESCENT = 3
EMPTY = -1  # New constant for empty spots

# EMT states
E = 'E'  # Epithelial
H = 'H'  # Hybrid
M = 'M'  # Mesenchymal

# # Parameters for probabilities
# division_probability = 0.8  # Probability of a cell dividing if space is available
# hybrid_migration_probability = 0.5  # Probability of migration happening during division
# mesenchymal_migration_probability = 0.8
# hybrid_senescence_migration_probability = 0.5
# death_probability = 0.01  # Probability of death happening per step
# constant_senescence_probability = [0.1, 0.01, 0.001, 0.00001]  # Probability of senescence happening during division

division_probability = 0.03
hybrid_migration_probability = 0.9
mesenchymal_migration_probability = 0.99
hybrid_senescence_migration_probability = 0.45
death_probability = 0.0003
constant_senescence_probability =[0.1, 0.2, 0.3, 0.4, 0.5]

# division_probability = 0.0278  # Probability of a cell dividing if space is available
# migration_probability = 0.9  # Probability of migration happening during division
# senescence_migration_probability = 0.45
# death_probability = 0.0003  # Probability of death happening per step
# # constant_senescence_probability = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]  # Probability of senescence happening during division
# constant_senescence_probability =[0.1]

# Probability of mesenchymal cells turn into epithelial cell
M_to_E_probability = 0.1

# Size of the grid
grid_size_x = 100
grid_size_y = 100

# Number of steps
NUM_STEPS = 150
