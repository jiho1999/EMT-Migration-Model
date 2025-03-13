# cell_actions.py

import random
from constants import E, H, M, grid_size_x, grid_size_y
from constants import EMPTY, ALIVE, DEAD, DIVIDING, SENESCENT

# Function to check room for division (without periodic boundary and correct boundary checks)
def check_room_in_grid(x, y, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, ''):
                return True
    return False

# Function to check if room is available in the list of new positions
def check_room_in_new_positions(x, y, new_positions, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if (nx, ny) not in new_positions:
                return True
    return False

# Check if neighboring grid is all empty
def all_neighbors_empty(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and empty
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) != (EMPTY, ''):
                return False  # If any neighbor is not EMPTY, return False
    
    return True  # All neighbors are EMPTY

# Check if neighboring grid is all empty or mesenchymal cell
def all_neighbors_empty_mesenchymal(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and empty
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) != (EMPTY, '')) and (grid[nx, ny]["emt_state"] != M):
                return False  # If any neighbor is not EMPTY, return False
    
    return True  # All neighbors are EMPTY or Mesenchymal

# Check if neighboring grid is all occupied
def all_neighbors_occupied(x, y, grid, grid_size_x, grid_size_y, EMPTY):
    neighbors = [
        (x+1, y+0), (x-1, y+0), (x+0, y+1), (x+0, y-1),  # Edge neighbors
        (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)   # Corner neighbors
    ]
    
    # Iterate through each neighbor and check if they are within bounds and occupied (not EMPTY)
    for nx, ny in neighbors:
        if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:  # Ensure within grid bounds
            if (grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, ''):
                return False  # If any neighbor is EMPTY, return False
    
    return True  # All neighbors are occupied (not EMPTY)

# Function to move cells to an available empty neighboring spot
def move_cells(x, y, new_positions, grid):
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(neighbors)
    for dx, dy in neighbors:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and check_room_in_new_positions(x, y, new_positions, grid):
                return nx, ny
    return x, y # Since we use move_cells function when we know there is a open spot, code will not reach return x, y

# Function to move Hybrid cells to an avaialbe empty neighboring spot where cells can remain the contact
def move_H_cell(x, y, new_positions, grid):
    # Assuming there is TGFb in the middle of the wound (index between 49 and 50)
    if y <= 49:
        neighbors = [(-1, 1), (0, 1), (1, 1)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and check_room_in_new_positions(x, y, new_positions, grid) and not(all_neighbors_empty_mesenchymal(nx, ny, grid, grid_size_x, grid_size_y, EMPTY)):
                    return nx, ny
    if y >= 50:
        neighbors = [(-1, -1), (0, -1), (1, -1)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and check_room_in_new_positions(x, y, new_positions, grid) and not(all_neighbors_empty_mesenchymal(nx, ny, grid, grid_size_x, grid_size_y, EMPTY)):
                    return nx, ny
    
    # # When there is no directional movement
    # neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # random.shuffle(neighbors)
    # for dx, dy in neighbors:
    #     nx, ny = x + dx, y + dy
    #     if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
    #         if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and check_room_in_new_positions(x, y, new_positions, grid) and not(all_neighbors_empty(nx, ny, grid, grid_size_x, grid_size_y, EMPTY)):
    #             return nx, ny
    return x, y # Since we use move_cells function when we know there is a open spot, code will not reach return x, y

# Define a function for cell division
def check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions):
    if random.random() < division_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        #new_states.append((DIVIDING, state[1]))  # Enter dividing state
        new_states.append((DIVIDING, H))  # Enter dividing state
        new_positions.append((x, y))  # Keep the original cell's position
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return True  # Division occurred
    return False  # Division didn't happen

# Define a function for cell death
def check_death(x, y, state, new_positions, new_states, death_probability, wound_positions):
    if random.random() < death_probability:  # Chance to die
        new_states.append((DEAD, ''))
        new_positions.append((x, y))  # Keep the dead cell in the grid for this cycle
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return True  # Death occurred
    return False  # Death didn't happen

# Define a function for cell migration (modifies migration_count)
def check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions):
    if state[1] == H and random.random() < hybrid_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        migration_count += 1  # Increment migration count
        new_x, new_y = move_H_cell(x, y, new_positions, grid)  # Move cell to a new position
        new_states.append((ALIVE, H))
        new_positions.append((new_x, new_y))

        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[x, y]['primary_state'] = EMPTY
        grid[x, y]['emt_state'] = ''

        grid[new_x, new_y]['primary_state'] = ALIVE
        grid[new_x, new_y]['emt_state'] = H

        if 30 <= new_x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((new_x, new_y))

        return True, migration_count  # Migration occurred
        
    if state[1] == M and random.random() < mesenchymal_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        migration_count += 1  # Increment migration count
        new_x, new_y = move_cells(x, y, new_positions, grid)  # Move cell to a new position
        new_states.append((ALIVE, M))
        new_positions.append((new_x, new_y))

        # Update the grid promptly in order to reflect the current grid status for next cells' division and migration in a single update step
        grid[x, y]['primary_state'] = EMPTY
        grid[x, y]['emt_state'] = ''

        grid[new_x, new_y]['primary_state'] = ALIVE
        grid[new_x, new_y]['emt_state'] = M

        if 30 <= new_x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((new_x, new_y))

        return True, migration_count  # Migration occurred

    return False, migration_count  # Migration didn't happen

def check_senescence_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_senescence_migration_probability, wound_positions):
    # H is the only senescent cells that can migrate. E cannot move and M cannot be a senescent cell.
    if state[1] == H and random.random() < hybrid_senescence_migration_probability and check_room_in_grid(x, y, grid) and check_room_in_new_positions(x, y, new_positions, grid):
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        random.shuffle(neighbors)
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if ((grid[nx, ny]["primary_state"], grid[nx, ny]["emt_state"]) == (EMPTY, '')) and not(all_neighbors_empty_mesenchymal(nx, ny, grid, grid_size_x, grid_size_y, EMPTY)):
                    new_states.append((SENESCENT, H))
                    new_positions.append((nx, ny))
                    grid[x, y]['primary_state'] = EMPTY
                    grid[x, y]['emt_state'] = ''

                    grid[nx, ny]['primary_state'] = SENESCENT
                    grid[nx, ny]['emt_state'] = H
                    if 30 <= nx <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((nx, ny))
                    migration_count += 1  # Increment migration count
                    return True, migration_count                            
                else:
                    grid[x, y]['primary_state'] = SENESCENT
                    grid[x, y]['emt_state'] = H
                    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
                        wound_positions.add((x, y))
                    return False, migration_count
    else:
        grid[x, y]['primary_state'] = SENESCENT
        grid[x, y]['emt_state'] = state[1]
        if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
            wound_positions.add((x, y))
        return False, migration_count

# Define a function for keeping a cell alive
def check_alive(x, y, state, new_positions, new_states, wound_positions):
    new_states.append((ALIVE, state[1]))
    new_positions.append((x, y))  # Keep the original position
    if 30 <= x <= 69:  # If the cell moves into the wound region, mark the wound position as updated
        wound_positions.add((x, y))
    return True  # Cell stays alive

# Function to choose a random action for each cell
def alive_random_action(x, y, state, grid, new_positions, new_states, migration_count, division_probability, death_probability, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions):
    # If the cell is senescent, it remains in its state and is not processed further
    if ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, H)) or ((grid[x, y]["primary_state"], grid[x, y]["emt_state"]) == (SENESCENT, E)):
        new_states.append((SENESCENT, state[1]))
        new_positions.append((x, y))
        if 30 <= x <= 69:  # Mark the wound position as updated if applicable
            wound_positions.add((x, y))
        return migration_count

    if state[1] == E:
        actions = [
            lambda: (check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions), migration_count),
            lambda: (check_death(x, y, state, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    if state[1] == H:
        actions = [
            lambda: (check_division(x, y, state, grid, new_positions, new_states, division_probability, wound_positions), migration_count),
            lambda: (check_death(x, y, state, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    if state[1] == M:
        actions = [
            lambda: (check_death(x, y, state, new_positions, new_states, death_probability, wound_positions), migration_count,),
            lambda: check_migration(x, y, state, grid, new_positions, new_states, migration_count, hybrid_migration_probability, mesenchymal_migration_probability, wound_positions),
            lambda: (check_alive(x, y, state, new_positions, new_states, wound_positions), migration_count)
        ]

    random.shuffle(actions)

    for action in actions:
        success, migration_count = action()
        if success:
            break

    return migration_count  # Return the updated migration_count
