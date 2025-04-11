import csv
import io
from cvxopt import matrix, solvers
import numpy as np
import os # Import os module for file path handling

# --- Data Preparation ---

# Define the expected path for the data file
data_file_path = 'data/nfl_picks.csv'
print(f"Attempting to read data from: {os.path.abspath(data_file_path)}")

try:
    # Read data from the specified CSV file
    with open(data_file_path, 'r', encoding='utf-8') as f: # Added encoding for broader compatibility
        csv_reader = csv.reader(f)
        # Read header safely
        try:
            header = next(csv_reader)
            print(f"CSV Header: {header}") # Print header for verification
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
    print("and the 'draft_picks.csv' file is inside the 'data' directory.")
    exit() # Exit if the required file is not found
except Exception as e:
    print(f"An error occurred while reading '{data_file_path}': {e}")
    exit() # Exit on other read errors

# --- Problem Setup ---
max_pick = 0
parsed_trades = []

def parse_picks(picks_str):
    """Parses a string like '6/28' into a list of integers [6, 28]."""
    if not picks_str:
        return []
    # Ensure robustness against empty strings or malformed entries
    try:
        # Filter out empty strings that might result from trailing slashes etc.
        return [int(p) for p in picks_str.strip().split('/') if p.strip()]
    except ValueError:
        # Provide more context in warning
        print(f"Warning: Could not parse pick value in string: '{picks_str}'. Skipping pick.")
        # Attempt to parse remaining valid numbers if possible, otherwise return empty
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
    # Check for expected number of columns (ignoring year, needing two pick columns)
    if len(trade_row) < 3:
        print(f"Warning: Skipping row {i+2} (1-based index in file) due to insufficient columns: {trade_row}")
        skipped_rows += 1
        continue

    # Assuming format is year, picks1, picks2
    _, picks1_str, picks2_str = trade_row[:3] # Take only the first 3 potential elements

    side1 = parse_picks(picks1_str)
    side2 = parse_picks(picks2_str)

    # Ensure the trade has picks and is not just empty strings parsed
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
                # This should ideally not happen if parse_picks returns only ints
                print(f"Warning: Could not find maximum pick in row {i+2}: {trade_row}")
                skipped_rows +=1
                parsed_trades.pop() # Remove the partially added trade
                successful_parses -= 1


    else:
        # This case might happen if parse_picks returns empty lists for both sides
        # e.g. if input was "year,,"
         print(f"Warning: Skipping row {i+2} as no valid picks were found in '{picks1_str}' or '{picks2_str}'.")
         skipped_rows += 1


if max_pick == 0:
    print("Error: No valid picks found in the data file. Cannot proceed.")
    print(f"({successful_parses} trades parsed, {skipped_rows} rows skipped)")
    exit()

N = max_pick
print(f"\nMaximum pick number found (N): {N}")
print(f"Total number of variables (pick values): {N}")
print(f"Number of trades successfully parsed for objective function: {len(parsed_trades)}")
if skipped_rows > 0:
     print(f"Number of rows skipped due to formatting issues: {skipped_rows}")

# --- Print Human-Readable Objective Function --- START ---
print("\nObjective Function (Human-Readable Representation):")
objective_terms = []
# Limit printing if there are too many trades
max_terms_to_print = 20
terms_printed = 0

for side1, side2 in parsed_trades:
    if terms_printed < max_terms_to_print:
        # Build string for side 1 sum
        side1_str_parts = [f"pick_{p}" for p in side1]
        side1_str = " + ".join(side1_str_parts) if side1_str_parts else "0"

        # Build string for side 2 sum
        side2_str_parts = [f"pick_{p}" for p in side2]
        side2_str = " + ".join(side2_str_parts) if side2_str_parts else "0"

        # Combine into the squared difference term for this trade
        term_str = f"(({side1_str}) - ({side2_str}))**2"
        objective_terms.append(term_str)
        terms_printed += 1
    elif terms_printed == max_terms_to_print:
        objective_terms.append("...") # Indicate more terms exist
        terms_printed += 1


# Join all trade terms with " + "
full_objective_str = "Minimize:\n  " + " + \n  ".join(objective_terms)
if len(parsed_trades) > max_terms_to_print:
    full_objective_str += f"\n  (Plus {len(parsed_trades) - max_terms_to_print} more terms)"
print(full_objective_str)
# --- Print Human-Readable Objective Function --- END ---


# --- Build CVXOPT Matrices ---
print("\nBuilding CVXOPT matrices...")
# Objective function: minimize (1/2)x'Px + q'x
# x is the vector of pick values [pick_1, pick_2, ..., pick_N]
P = matrix(0.0, (N, N))
q = matrix(0.0, (N, 1))

# Build the P matrix from trades
for side1, side2 in parsed_trades:
    a = matrix(0.0, (N, 1))
    for pick in side1:
        if 1 <= pick <= N:
            a[pick - 1] += 1.0 # pick_i corresponds to index i-1
        else:
            print(f"Warning: Pick {pick} from side 1 is out of range [1, {N}]. Check data.") # Should not happen if max_pick is correct
    for pick in side2:
        if 1 <= pick <= N:
            a[pick - 1] -= 1.0
        else:
             print(f"Warning: Pick {pick} from side 2 is out of range [1, {N}]. Check data.")
    P += 2.0 * (a * a.T)

# Constraints: Gx <= h and Ax = b

G_list = []
h_list = []
num_constraints = 0 # Keep track for reporting

# 1. Non-negativity: pick_i >= 0  => -pick_i <= 0
for i in range(N):
    row = [0.0] * N
    row[i] = -1.0
    G_list.append(row)
    h_list.append(0.0)
num_constraints += N

# 2. Monotonicity: pick_i >= pick_{i+1} => pick_{i+1} - pick_i <= 0
for i in range(N - 1):
    row = [0.0] * N
    row[i] = -1.0
    row[i+1] = 1.0
    G_list.append(row)
    h_list.append(0.0)
num_constraints += (N - 1)

# 3. Convexity: (pick_i - pick_{i+1}) >= (pick_{i+1} - pick_{i+2})
#    => -pick_i + 2*pick_{i+1} - pick_{i+2} <= 0
for i in range(N - 2):
    row = [0.0] * N
    row[i] = -1.0
    row[i+1] = 2.0
    row[i+2] = -1.0
    G_list.append(row)
    h_list.append(0.0)
num_constraints += (N - 2)

# Convert G and h to cvxopt matrices
try:
    G_np = np.array(G_list, dtype=float) # Ensure float type
    G = matrix(G_np) # Shape (m, N)
    h = matrix(h_list)           # Shape (m, 1)
except Exception as e:
    print(f"Error creating G/h matrices: {e}")
    print(f"Expected G shape: ({num_constraints}, {N})")
    print(f"Actual G_list length: {len(G_list)}")
    if G_list:
        print(f"Actual row length: {len(G_list[0])}")
    exit()


# 4. Equality constraint: pick_1 = 3000
try:
    A = matrix(0.0, (1, N))
    A[0, 0] = 1.0
    b = matrix([3000.0])        # Shape (1, 1)
except IndexError:
     print(f"Error setting equality constraint: N={N}. Is N at least 1?")
     exit()
except Exception as e:
     print(f"Error creating A/b matrices: {e}")
     exit()

print(f"\nMatrix dimensions:")
print(f"  N (variables): {N}")
print(f"  m (inequality constraints): {num_constraints}")
print(f"  P: {P.size}")
print(f"  q: {q.size}")
print(f"  G: {G.size}")
print(f"  h: {h.size}")
print(f"  A: {A.size}")
print(f"  b: {b.size}")

# --- Solve the QP Problem ---
print("\nSolving the Quadratic Program...")
# Configure solver options (optional, can help with convergence/numerical issues)
solvers.options['show_progress'] = False # Set to True to see detailed solver output
# solvers.options['abstol'] = 1e-7
# solvers.options['reltol'] = 1e-6
# solvers.options['feastol'] = 1e-7

try:
    solution = solvers.qp(P, q, G, h, A, b)
    if solution['x'] is not None:
        optimal_pick_values = solution['x']
    else:
        print("Error: Solver did not return a solution vector ('x' is None).")
        print(f"Solver status: {solution.get('status', 'N/A')}") # Use .get for safety
        exit()


    # --- Output Results ---
    if solution.get('status') == 'optimal': # Check status safely
        print("\nOptimal solution found.")
        # Optionally print the objective value
        objective_value = solution.get('primal objective') # Use .get for safety
        if objective_value is not None:
             print(f"Optimal Objective Function Value: {objective_value:.4f}")

        print("\nPick Values:")
        output_lines = []
        for i in range(N):
            # Handle potential floating point inaccuracies near zero
            val = optimal_pick_values[i] if optimal_pick_values[i] > 1e-7 else 0.0
            output_lines.append(f"  Pick {i+1}: {val:.4f}")

        # Print in columns for better readability if many picks
        num_columns = 3
        col_width = 20 # Adjust as needed
        num_rows = (N + num_columns - 1) // num_columns
        for r in range(num_rows):
            line = ""
            for c in range(num_columns):
                idx = r + c * num_rows
                if idx < N:
                    line += output_lines[idx].ljust(col_width)
            print(line)


        # Verify constraints (optional)
        print("\nVerifying Constraints (approximate due to floating point):")
        pick_1_val = optimal_pick_values[0] if N > 0 else float('nan')
        print(f"  pick_1 = {pick_1_val:.4f} (Expected: 3000)")

        monotonic = True
        if N > 1:
             monotonic = all(optimal_pick_values[i] >= optimal_pick_values[i+1] - 1e-6 for i in range(N-1))
        print(f"  Monotonicity (pick_i >= pick_{{i+1}}): {monotonic}")

        convex = True
        if N > 2:
            convex = all((optimal_pick_values[i] - optimal_pick_values[i+1]) >= (optimal_pick_values[i+1] - optimal_pick_values[i+2]) - 1e-6 for i in range(N-2))
        print(f"  Convexity ((pick_i - pick_{{i+1}}) >= (pick_{{i+1}} - pick_{{i+2}})): {convex}")

        non_negative = True
        if N > 0:
            non_negative = all(optimal_pick_values[i] >= -1e-6 for i in range(N))
        print(f"  Non-negativity (pick_i >= 0): {non_negative}")

    else:
        print(f"\nSolver finished with status: {solution.get('status', 'N/A')}") # Use .get for safety
        print("Could not find the optimal solution.")
        if solution.get('status') == 'unknown':
             print("  This might be due to numerical issues. Consider adjusting solver options (e.g., 'feastol', 'abstol', 'reltol') or checking data quality.")
        elif solution.get('status') == 'primal infeasible':
             print("  The problem constraints may be contradictory (e.g., pick_1=3000 conflicts with monotonicity/convexity based on the data). Check constraints and data.")
        elif solution.get('status') == 'dual infeasible':
             print("  The objective function might be unbounded below within the feasible region.")


except ValueError as e:
    # CVXOPT can raise ValueError for various reasons, including Rank(A) < p or non-PSD P matrix
    print(f"\nError during optimization (ValueError): {e}")
    print("This might indicate issues with:")
    print("  - Linear dependence in equality constraints (A matrix).")
    print("  - The objective matrix P not being positive semidefinite.")
    print("  - General numerical instability or infeasibility.")
except TypeError as e:
     print(f"\nTypeError during optimization: {e}")
     print("This often indicates incorrect matrix types or dimensions passed to the solver. Check matrix construction.")
except ArithmeticError as e:
     print(f"\nArithmeticError during optimization: {e}")
     print("This might indicate numerical issues like division by zero or overflow within the solver.")
except Exception as e:
    # Catch any other unexpected errors during solving
    print(f"\nAn unexpected error occurred during optimization: {type(e).__name__}: {e}")
