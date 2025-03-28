# -*- coding: utf-8 -*-
"""simulatedAnnealing.ipynb

 Probability Function
This function determines whether a worse state (higher cost) will be accepted based on the Metropolis criterion:

Logic:
If the new state's cost (new_cost) is lower than the current state's cost (old_cost), accept it unconditionally.

If the new state's cost is higher, accept it with a probability that decreases as the cost difference and temperature change.
"""

# def acceptance_probability(old_cost, new_cost, temperature):
#     if new_cost < old_cost:
#         # Always accept better states
#         return 1.0
#     else:
#         # Accept worse states with a probability
#         return math.exp(-(new_cost - old_cost) / temperature)

"""Explanation:
The worse the new state is (i.e., the higher the cost difference), the lower the acceptance probability.

As the temperature decreases, the algorithm becomes less likely to accept worse states, simulating a gradual "cooling."
"""



import math
import random
import time

# Define the heuristic functions
def h1(state, goal):
    # Number of displaced tiles
    return sum([1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal[i][j]])

def h2(state, goal):
    # Manhattan distance
    total_distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, 3)
                total_distance += abs(x - i) + abs(y - j)
    return total_distance

def h3(state, goal):
    # Combined heuristic h3 = h1 * h2
    return h1(state, goal) * h2(state, goal)


# Generate neighbors
def generate_neighbors(state):
    neighbors = []
    zero_x, zero_y = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        x, y = zero_x + dx, zero_y + dy
        if 0 <= x < 3 and 0 <= y < 3:
            new_state = [row[:] for row in state]
            new_state[zero_x][zero_y], new_state[x][y] = new_state[x][y], new_state[zero_x][zero_y]
            neighbors.append(new_state)
    return neighbors

# Probability function (Metropolis criterion)
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp(-(new_cost - old_cost) / temperature)

# Simulated Annealing Function
def simulated_annealing(start, goal, heuristic, temperature, cooling_rate):
    current = start
    current_cost = heuristic(current, goal)
    states_explored = 0
    start_time = time.time()
    path = [start]  # Store the intermediate states
    random_walk_count = 0  # Count consecutive random walks
    pure_best_cost = current_cost
    final_state = current

    while temperature > 1e-3:
        neighbors = generate_neighbors(current)
        next_state = random.choice(neighbors)
        next_cost = heuristic(next_state, goal)
        states_explored += 1

        # Detect random walk scenario: Cost remains the same
        if current_cost == next_cost:
            random_walk_count += 1
        else:
            random_walk_count = 0  # Reset if the cost changes (progress is made)

        if random.uniform(0, 1) < acceptance_probability(current_cost, next_cost, temperature):
            if next_cost <= pure_best_cost:
              path.append(current)  # Append the new state to the path
              pure_best_cost = next_cost

            current = next_state
            current_cost = next_cost


        temperature *= cooling_rate  # Cooling schedule

        # Check if random walk persists for too long
        if random_walk_count > 10:  # Threshold: Adjust based on your requirements
            # print("Random Walk Scenario Detected: Reheating...")
            temperature *= 1.5  # Reheat to escape local optima
            random_walk_count = 0  # Reset the counter after reheating
        final_state = current

    end_time = time.time()
    success = current == goal
    return {
        "success": success,
        "path": path,
        "states_explored": states_explored,
        "time_taken": end_time - start_time,
        "random_walk": random_walk_count > 0,  # Did a random walk ever occur?
        "final_state":final_state
    }

# Main Function
if __name__ == "__main__":
    # Initialize start and goal states
    start = [
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ]
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    # Define parameters
    heuristics = [h1, h2, h3]  # List of all heuristics
    temperature = 10000.0  # Initial temperature
    cooling_rate = 0.95  # Cooling rate
    print(f"Initial Temperature: {temperature}")
    print(f"Cooling Function: Exponential Decay (Rate = {cooling_rate})")

    for heuristic in heuristics:
        print(f"\nHeuristic Chosen: {heuristic.__name__}")
        # Run Simulated Annealing for each heuristic
        result = simulated_annealing(start, goal, heuristic, temperature, cooling_rate)

        # Output results for each heuristic
        print("\nResults:")
        print(f"Initial State:")
        for row in start:
            print(row)
        print(f"Goal State: ")
        for row in goal:
            print(row)
        print(f"Success: {result['success']}")
        print(f"\n(Sub)Optimal Path:")
        for state in result['path']:
            for row in state:
                print(row)
            print()  # Blank line for better readability
        print("Final State:")
        for row in result['final_state']:
            print(row)
        print(f"Total States Explored: {result['states_explored']}")
        print(f"Time Taken: {result['time_taken']} seconds")
        print(f"Random Walk Scenario: {'Occurred' if result['random_walk'] else 'Did Not Occur'}")
        print("************************************************************************************************************")




# " Another way "





import math
import random
import time

# Define the heuristic functions
def h1(state, goal):
    # Number of displaced tiles
    return sum([1 for i in range(3) for j in range(3) if state[i][j] != 0 and state[i][j] != goal[i][j]])

def h2(state, goal):
    # Manhattan distance
    total_distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, 3)
                total_distance += abs(x - i) + abs(y - j)
    return total_distance

def h3(state, goal):
    # Combined heuristic h3 = h1 * h2
    return h1(state, goal) * h2(state, goal)

# Generate neighbors
def generate_neighbors(state):
    neighbors = []
    zero_x, zero_y = [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0][0]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dx, dy in directions:
        x, y = zero_x + dx, zero_y + dy
        if 0 <= x < 3 and 0 <= y < 3:
            new_state = [row[:] for row in state]
            new_state[zero_x][zero_y], new_state[x][y] = new_state[x][y], new_state[zero_x][zero_y]
            neighbors.append(new_state)
    return neighbors

# Probability function (Metropolis criterion)
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    else:
        return math.exp(-(new_cost - old_cost) / temperature)

# Simulated Annealing Function
def simulated_annealing(start, goal, heuristic, temperature, cooling_rate):
    current = start
    current_cost = heuristic(current, goal)
    states_explored = 0
    start_time = time.time()
    path = [start]  # Store the intermediate states
    random_walk_count = 0  # Count consecutive random walks
    pure_best_cost = current_cost
    final_state = current

    while temperature > 1e-3:
        neighbors = generate_neighbors(current)
        next_state = random.choice(neighbors)
        next_cost = heuristic(next_state, goal)
        states_explored += 1

        # Detect random walk scenario: Cost remains the same
        if current_cost == next_cost:
            random_walk_count += 1
        else:
            random_walk_count = 0  # Reset if the cost changes (progress is made)

        if random.uniform(0, 1) < acceptance_probability(current_cost, next_cost, temperature):
            if next_cost <= pure_best_cost:
                path.append(current)  # Append the new state to the path
                pure_best_cost = next_cost

            current = next_state
            current_cost = next_cost

        temperature *= cooling_rate  # Cooling schedule

        # Check if random walk persists for too long
        if random_walk_count > 10:  # Threshold: Adjust based on your requirements
            # print("Random Walk Scenario Detected: Reheating...")
            temperature *= 1.5  # Reheat to escape local optima
            random_walk_count = 0  # Reset the counter after reheating
        final_state = current  # Updated final state after each iteration

    end_time = time.time()
    success = current == goal
    return {
        "success": success,
        "path": path,
        "states_explored": states_explored,
        "time_taken": end_time - start_time,
        "random_walk": random_walk_count > 0,  # Did a random walk ever occur?
        "final_state": final_state
    }

def read_matrix_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(int, line.split())) for line in lines]
    return matrix

# Example usage
file_name = "input.txt"  # Replace with the path to your file
matrix = read_matrix_from_file(file_name)

# Print the matrix
for row in matrix:
    print(row)

# Main Function
if __name__ == "__main__":
    # Initialize start and goal states
    start = [
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ]
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    # Define parameters
    heuristics = [h1, h2, h3]  # List of all heuristics
    temperature = 10000.0  # Initial temperature
    cooling_rate = 0.95  # Cooling rate
    print(f"Initial Temperature: {temperature}")
    print(f"Cooling Function: Exponential Decay (Rate = {cooling_rate})")

    for heuristic in heuristics:
        print(f"\nHeuristic Chosen: {heuristic.__name__}")
        # Run Simulated Annealing for each heuristic
        result = simulated_annealing(start, goal, heuristic, temperature, cooling_rate)

        # Output results for each heuristic
        print("\nResults:")
        print(f"Initial State:")
        for row in start:
            print(row)
        print(f"Goal State: ")
        for row in goal:
            print(row)
        print(f"Success: {result['success']}")
        print(f"\n(Sub)Optimal Path:")
        for state in result['path']:
            for row in state:
                print(row)
            print()  # Blank line for better readability
        print("Final State:")
        for row in result['final_state']:
            print(row)
        print(f"Total States Explored: {result['states_explored']}")
        print(f"Time Taken: {result['time_taken']} seconds")
        print(f"Random Walk Scenario: {'Occurred' if result['random_walk'] else 'Did Not Occur'}")
        print("************************************************************************************************************")

4 # Main Function
if __name__ == "__main__":

    start = []

    print("Enter the elements row-wise:")

    for i in range(3):
      row = list(map(int, input().split()))  # Taking space-separated input for each row
      start.append(row)

    print("Start State:")
    for row in start:
      print(row)

    goal = []

    print("Enter the elements row-wise:")

    for i in range(3):
      row = list(map(int, input().split()))  # Taking space-separated input for each row
      goal.append(row)


    print("Goal State:")
    for row in goal:
      print(row)

    # temperature = 10000.0  # Initial temperature
    # cooling_rate = 0.95  # Cooling rate
    print("Enter the temperature value:")
    temperature = float(input())
    print("Enter the cooling rate value between 0 and 1:")
    cooling_rate = float(input())
    print(f"Initial Temperature: {temperature}")
    print(f"Cooling Rate : {cooling_rate}")
    print()


    print("1 : for h1(n) (Number of displaced tiles)")
    print("2 : for h2(n) (Total Manhattan Distance)")
    print("3 : for h3(n) = h1(n) * h2(n)")

    heuristic_choice = int(input("Enter your choice: "))
    if heuristic_choice == 1:
      heuristic = h1
    elif heuristic_choice == 2:
      heuristic = h2
    elif heuristic_choice == 3:
      heuristic = h3

    print(f"\nHeuristic Chosen: {heuristic.__name__}")
        # Run Simulated Annealing for each heuristic
    result = simulated_annealing(start, goal, heuristic, temperature, cooling_rate)

        # Output results for each heuristic
    print("\nResults:")
    print(f"Initial State:")
    for row in start:
        print(row)
    print(f"Goal State: ")
    for row in goal:
        print(row)
    print(f"Success: {result['success']}")
    print(f"\n(Sub)Optimal Path:")
    for state in result['path']:
      for row in state:
        print(row)
      print()  # Blank line for better readability
    print("Final State:")
    for row in result['final_state']:
        print(row)
    print(f"Total States Explored: {result['states_explored']}")
    print(f"Time Taken: {result['time_taken']} seconds")
    print(f"Random Walk Scenario: {'Occurred' if result['random_walk'] else 'Did Not Occur'}")



"""a. Check whether the heuristics are admissible
A heuristic is admissible if it never overestimates the cost to reach the goal.

For h1(n) (Number of displaced tiles):
Admissible?: Yes. h1 counts only the number of misplaced tiles but does not consider how far the tiles need to move. It underestimates the true cost and is therefore admissible.

For h2(n) (Total Manhattan Distance):
Admissible?: Yes. h2 calculates the sum of the minimum number of moves required for each tile to reach its correct position, ignoring obstacles like other tiles. Hence, it never overestimates and is admissible.
"""



"""**What happens if we make a new heuristic**

h3(n) = h1(n) * h2(n)?
Implications:

h3(n) may guide the search faster because it introduces a stricter penalty for misplaced tiles that are far away, creating a more aggressive exploration.

However, since it is inadmissible, the solution may not be optimal, as overestimation could cause the algorithm to ignore certain promising paths.

Recommendation: If you're optimizing for speed rather than guaranteed optimality, h3(n) could be useful. However, in cases where optimality is required, stick to h1 or h2.
"""



"""**c. What happens if you consider the blank tile as another tile?**

By default, the blank tile (zero) is not included in heuristic calculations. Treating it as another tile changes the heuristic calculations significantly:

For h1(n):
If the blank tile is treated as a normal tile, the count of misplaced tiles will increase whenever the blank tile is not in its designated position.

This could make h1 inadmissible, as the blank tile does not affect the actual cost to solve the puzzle.

For h2(n):
Including the blank tile's Manhattan distance could inflate the heuristic value unnecessarily, leading to overestimations and inadmissibility.

Implications:
Admissibility: Heuristics that include the blank tile are likely to become inadmissible.

Performance: Including the blank tile may slow the search by penalizing irrelevant moves involving the blank tile.
"""



"""A local optimum occurs when the algorithm finds a solution that is better than all its immediate neighbors but is not the global best solution.

Ways to Escape Local Optima:
**Reheating Mechanism:**

Increase the temperature temporarily to allow the algorithm to accept worse states and explore other regions of the search space.
"""

# temperature *= 1.5  # Increase temperature temporarily

"""**Random Restarts:**

Restart the algorithm from a new random state if progress stagnates.

current_state = generate_random_state()

"""



