import cvxpy as cp
import numpy as np
import csv
import io # Keep io for potential fallback/testing
import os # For file path handling
import time # To time the solver

# --- Data Preparation ---

# Define the expected path for the data file
data_file_path = 'data/nfl_picks.csv'
print(f"Attempting to read data from: {os.path.abspath(data_file_path)}")

try:
    # Read data from the specified CSV file
    with open(data_file_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        try:
            header = next(csv_reader)
            print(f"CSV Header: {header}")
        except StopIteration:
            print(f"Error: File '{data_file_path}' is empty.")
            exit()
        trades = list(csv_reader)
    print(f"Successfully read {len(trades)} trade rows from {data_file_path}")
    if not trades:
        print("Warning: No trade data found in the file after the header.")

except FileNotFoundError:
    print(f"Error: File not found at '{os.path.abspath(data_file_path)}'.")
    print("Please ensure the 'data' directory exists in the same directory as the script,")
    print("and the 'nfl_picks.csv' file is inside the 'data' directory.")
    # --- Fallback to sample data for demonstration if file not found ---
    # print("Using internal sample data for demonstration.")
    # sample_data = """year,picks1,picks2
    # 1992,4,6/28
    # 1992,17/120,19/104
    # 1992,13/71,19/37/104
    # 1992,20,37/64/108
    # 1992,36/121,52/78/163/222/329
    # 1992,40,47/74
    # 1992,47/74,56/58
    # 1992,56,82/109/250
    # """
    # csv_file = io.StringIO(sample_data)
    # csv_reader = csv.reader(csv_file)
    # header = next(csv_reader) # Skip header
    # trades = list(csv_reader)
    # --- End Fallback ---
    # If not using fallback, exit:
    exit()
except Exception as e:
    print(f"An error occurred while reading '{data_file_path}': {e}")
    exit()

# --- Problem Setup ---
max_pick = 0
parsed_trades = []

def parse_picks(picks_str):
    """Parses a string like '6/28' into a list of integers [6, 28]."""
    if not picks_str:
        return []
    try:
        return [int(p) for p in picks_str.strip().split('/') if p.strip()]
    except ValueError:
        print(f"Warning: Could not parse pick value in string: '{picks_str}'. Attempting partial parse.")
        valid_picks = []
        for p in picks_str.strip().split('/'):
            if p.strip():
                try:
                    valid_picks.append(int(p))
                except ValueError:
                    print(f"  - Skipping invalid part: '{p}'")
        return valid_picks

# Parse trades and find the maximum pick number (N)
successful_parses = 0
skipped_rows = 0
for i, trade_row in enumerate(trades):
    if len(trade_row) < 3:
        print(f"Warning: Skipping row {i+2} (1-based index in file) due to insufficient columns: {trade_row}")
        skipped_rows += 1
        continue
    _, picks1_str, picks2_str = trade_row[:3]
    side1 = parse_picks(picks1_str)
    side2 = parse_picks(picks2_str)
    if side1 or side2:
        parsed_trades.append((side1, side2))
        successful_parses += 1
        all_picks = side1 + side2
        if all_picks:
            try:
                current_max = max(all_picks)
                if current_max > max_pick:
                     max_pick = current_max
            except ValueError:
                print(f"Warning: Could not find maximum pick in row {i+2}: {trade_row}")
                skipped_rows +=1
                parsed_trades.pop()
                successful_parses -= 1
    else:
         print(f"Warning: Skipping row {i+2} as no valid picks were found in '{picks1_str}' or '{picks2_str}'.")
         skipped_rows += 1

if max_pick == 0:
    print("Error: No valid picks found in the data. Cannot proceed.")
    print(f"({successful_parses} trades parsed, {skipped_rows} rows skipped)")
    exit()

N = max_pick
print(f"\nMaximum pick number found (N): {N}")
print(f"Total number of variables (pick values): {N}")
print(f"Number of trades successfully parsed for objective function: {len(parsed_trades)}")
if skipped_rows > 0:
     print(f"Number of rows skipped due to formatting issues: {skipped_rows}")

# --- Print Human-Readable Objective Function (using original string representation) ---
print("\nObjective Function (Human-Readable Representation):")
objective_terms_str = []
max_terms_to_print = 20
terms_printed = 0
for side1, side2 in parsed_trades:
    if terms_printed < max_terms_to_print:
        side1_str_parts = [f"pick_{p}" for p in side1]
        side1_str = " + ".join(side1_str_parts) if side1_str_parts else "0"
        side2_str_parts = [f"pick_{p}" for p in side2]
        side2_str = " + ".join(side2_str_parts) if side2_str_parts else "0"
        term_str = f"(({side1_str}) - ({side2_str}))**2"
        objective_terms_str.append(term_str)
        terms_printed += 1
    elif terms_printed == max_terms_to_print:
        objective_terms_str.append("...")
        terms_printed += 1
full_objective_str = "Minimize:\n  " + " + \n  ".join(objective_terms_str)
if len(parsed_trades) > max_terms_to_print:
    full_objective_str += f"\n  (Plus {len(parsed_trades) - max_terms_to_print} more terms)"
print(full_objective_str)
# --- End Human-Readable Objective ---

# --- Define CVXPY Problem ---
print("\nDefining the CVXPY problem...")

# 1. Variables: Define the pick values as a CVXPY variable vector
# Using pos=True enforces v >= 0, simplifying constraints later
# v = cp.Variable(N, name="pick_values", pos=True)
# If pos=True gives issues with specific solvers, use explicit constraint:
v = cp.Variable(N, name="pick_values")

# 2. Objective Function: Sum of squared differences for each trade
objective_terms = []
for side1, side2 in parsed_trades:
    # Create CVXPY expressions for the sum of values on each side.
    # Ensure picks are within the valid range [1, N] and use 0-based index for v.
    # cp.sum works correctly on empty lists (evaluates to 0).
    sum1_expr = cp.sum([v[p-1] for p in side1 if 1 <= p <= N])
    sum2_expr = cp.sum([v[p-1] for p in side2 if 1 <= p <= N])

    # Calculate the squared difference for the trade
    term = cp.square(sum1_expr - sum2_expr)
    objective_terms.append(term)

# Define the objective to minimize the sum of these terms
objective = cp.Minimize(cp.sum(objective_terms))

# 3. Constraints
constraints = []

# Add Non-negativity constraint explicitly if not using pos=True
constraints.append(v >= 0)

# Anchor: pick_1 = 3000 (v[0] in 0-based index)
if N >= 1:
    constraints.append(v[0] == 3000)
else:
    print("Warning: N=0, cannot apply anchor constraint.")

# Monotonicity: v_i >= v_{i+1} (vectorized: v[0]>=v[1], v[1]>=v[2], ...)
if N >= 2:
    # v[:-1] includes elements 0 to N-2
    # v[1:] includes elements 1 to N-1
    constraints.append(v[:-1] >= v[1:])

# Convexity: v_i - 2*v_{i+1} + v_{i+2} >= 0
if N >= 3:
    # v[:-2] includes elements 0 to N-3
    # v[1:-1] includes elements 1 to N-2
    # v[2:] includes elements 2 to N-1
    constraints.append(v[:-2] - 2 * v[1:-1] + v[2:] >= 0)

# 4. Create the Problem object
try:
    problem = cp.Problem(objective, constraints)
    # Check if the problem is Disciplined Convex Programming compliant
    print(f"Problem created. Is DCP: {problem.is_dcp()}")
    if not problem.is_dcp():
        print("Warning: Problem is not DCP. CVXPY might not be able to solve it.")
        # You might need to investigate why it's not DCP if this occurs.
except Exception as e:
    print(f"Error creating CVXPY problem object: {e}")
    exit()


# --- Solve the CVXPY Problem ---
print("\nSolving the CVXPY problem...")
start_time = time.time()
# You can specify solvers known to handle QPs well.
# Examples: OSQP, SCS, CLARABEL, ECOS (might struggle with QP), GUROBI (if installed+license)
# Letting CVXPY choose is usually fine. Add 'verbose=True' for detailed solver output.
try:
    problem.solve(solver=cp.PIQP, verbose=True)
    #problem.solve(solver=cp.CLARABEL, verbose=True)
    #problem.solve(solver=cp.CPLEX, verbose=True)
    #problem.solve(solver=cp.MOSEK, verbose=True)
    #problem.solve(solver=cp.ECOS, verbose=True)
    #problem.solve(solver=cp.OSQP, verbose=True, max_iter=100000)
    #problem.solve(solver=cp.COPT, verbose=True)
    #problem.solve(solver=cp.SCS, verbose=True, max_iters=200000)
    #problem.solve(verbose=False)
except cp.error.SolverError as e:
     print(f"CVXPY Solver Error: {e}")
     print("This might indicate the chosen solver is not installed or cannot handle the problem.")
     print(f"Installed solvers: {cp.installed_solvers()}")
     exit()
except Exception as e:
    print(f"An unexpected error occurred during solving: {e}")
    exit()

end_time = time.time()
print(f"Solver finished in: {end_time - start_time:.2f} seconds")
print(f"Solver status: {problem.status}")


# --- Process Results ---
# Check if the solver found an optimal or near-optimal solution
if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
    status_suffix = "(inaccurate)" if problem.status == cp.OPTIMAL_INACCURATE else ""
    print(f"\nOptimal solution found {status_suffix}.")

    # Retrieve the optimal values. v.value is a NumPy array.
    optimal_pick_values = v.value
    if optimal_pick_values is None:
         print("Error: Solver status optimal/inaccurate, but result vector is None.")
         exit()

    objective_value = problem.objective.value
    if objective_value is not None:
         print(f"Optimal Objective Function Value: {objective_value:.4f}")
    else:
         print("Objective value not available.")

    # --- Print Pick Values to Console (Sample) ---
    print("\nPick Values (sample shown below, full results in CSV):")
    output_lines = []
    for i in range(N):
        val = optimal_pick_values[i] if optimal_pick_values[i] > 1e-7 else 0.0
        output_lines.append(f"  Pick {i+1}: {val:.4f}")
    num_columns = 3
    col_width = 20
    rows_to_print = min(15, (N + num_columns -1) // num_columns)
    for r in range(rows_to_print):
        line = ""
        for c in range(num_columns):
            idx = r + c * ((N + num_columns -1) // num_columns)
            if idx < N:
                line += output_lines[idx].ljust(col_width)
        print(line)
    if N > rows_to_print * num_columns:
        print("  ...")

    # --- Verify Constraints ---
    print("\nVerifying Constraints (approximate due to floating point):")
    pick_1_val = optimal_pick_values[0] if N > 0 else float('nan')
    print(f"  pick_1 = {pick_1_val:.4f} (Expected: 3000, Diff: {abs(pick_1_val - 3000):.4e})")

    monotonic = True
    if N > 1:
         min_mono_diff = np.min(optimal_pick_values[:-1] - optimal_pick_values[1:])
         monotonic = min_mono_diff >= -1e-6
    print(f"  Monotonicity (pick_i >= pick_{{i+1}}): {monotonic} (Min diff: {min_mono_diff:.4e})")

    convex = True
    if N > 2:
        convexity_diffs = (optimal_pick_values[:-2] - optimal_pick_values[1:-1]) - \
                          (optimal_pick_values[1:-1] - optimal_pick_values[2:])
        min_conv_diff = np.min(convexity_diffs)
        convex = min_conv_diff >= -1e-6
    print(f"  Convexity ((v_i-v_{{i+1}}) >= (v_{{i+1}}-v_{{i+2}})): {convex} (Min diff: {min_conv_diff:.4e})")

    non_negative = True
    if N > 0:
        min_nonneg_val = np.min(optimal_pick_values)
        non_negative = min_nonneg_val >= -1e-6
    print(f"  Non-negativity (pick_i >= 0): {non_negative} (Min value: {min_nonneg_val:.4e})")

    # --- Save Results to CSV ---
    # Use a slightly different filename for the CVXPY version
    output_filename = 'pick_table_cvxpy.csv'
    print(f"\nSaving results to {output_filename}...")
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Pick', 'Value']) # Header
            for i in range(N):
                val = optimal_pick_values[i] if optimal_pick_values[i] > 1e-7 else 0.0
                writer.writerow([i + 1, f"{val:.4f}"])
        print(f"Successfully saved results to {os.path.abspath(output_filename)}")
    except IOError as e:
        print(f"Error: Could not write to file {output_filename}. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving CSV: {e}")

# Handle other solver statuses
elif problem.status == cp.INFEASIBLE:
    print("\nProblem Status: INFEASIBLE")
    print("The constraints cannot be satisfied simultaneously.")
    print("Possible causes:")
    print("  - Contradictory constraints (check formulation).")
    print("  - The anchor constraint (v[0] == 3000) might conflict with monotonicity/convexity based on the trade data.")
    print("  - Issues in the input data leading to impossible scenarios.")
elif problem.status == cp.UNBOUNDED:
     print("\nProblem Status: UNBOUNDED")
     print("The objective function can be decreased indefinitely while satisfying constraints.")
     print("Possible causes:")
     print("  - Missing constraints (e.g., non-negativity if not enforced).")
     print("  - Errors in the objective function formulation.")
     print("  - Issues in the input data.")
elif problem.status == cp.INFEASIBLE_INACCURATE:
    print("\nProblem Status: INFEASIBLE_INACCURATE")
    print("Solver determined the problem to likely be infeasible, but results may be inaccurate.")
elif problem.status == cp.UNBOUNDED_INACCURATE:
    print("\nProblem Status: UNBOUNDED_INACCURATE")
    print("Solver determined the problem to likely be unbounded, but results may be inaccurate.")
elif problem.status == cp.SOLVER_ERROR:
     print("\nProblem Status: SOLVER_ERROR")
     print("The solver encountered an error. Try a different solver (e.g., OSQP, SCS, CLARABEL) if installed.")
     print(f"Installed solvers: {cp.installed_solvers()}")
else:
     print(f"\nSolver failed with unhandled status: {problem.status}")
