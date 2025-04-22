import cvxpy as cp
import numpy as np
import csv
import os
import time
from datetime import datetime

print(f"Python CVXPY Script Started: {datetime.now()}")
print(f"CVXPY Version: {cp.__version__}")
print(f"NumPy Version: {np.__version__}")

# --- Data Parsing Function ---
def parse_picks(picks_str):
    """Parses a string like '6/28' into a list of integers [6, 28]."""
    if not picks_str:
        return []
    try:
        # Filter out empty strings that might result from trailing slashes etc.
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

# --- Function to Solve the QP for a Fixed k ---
def solve_for_k(k_py, N, parsed_trades):
    """
    Defines and solves the CVXPY convex optimization problem for a given
    transition point k_py (0-based index).

    Args:
        k_py (int): The 0-based index where concavity starts (i.e., k_py = k_user - 1).
                      Index i=0..k_py-1 has convex constraint (>=0).
                      Index i=k_py..N-3 has concave constraint (<=0).
        N (int): Total number of pick variables.
        parsed_trades (list): List of tuples, where each tuple is ([side1_picks], [side2_picks]).

    Returns:
        tuple: (status, objective_value, variable_values)
               Returns (status, None, None) on failure or non-optimal status.
    """
    # 1. Variables
    v = cp.Variable(N, name="pick_values")

    # 2. Objective
    objective_terms = []
    for side1, side2 in parsed_trades:
        # Ensure picks are valid indices (0 to N-1)
        valid_side1_indices = [p - 1 for p in side1 if 1 <= p <= N]
        valid_side2_indices = [p - 1 for p in side2 if 1 <= p <= N]

        # Use list comprehension for slicing if indices exist, else 0
        sum1_expr = cp.sum(v[valid_side1_indices]) if valid_side1_indices else 0.0
        sum2_expr = cp.sum(v[valid_side2_indices]) if valid_side2_indices else 0.0
        term = cp.square(sum1_expr - sum2_expr)
        objective_terms.append(term)

    if not objective_terms: # Handle case with no valid trades
         return "no_trades", None, None

    objective = cp.Minimize(cp.sum(objective_terms))

    # 3. Constraints
    constraints_k = []
    # Non-negativity
    constraints_k.append(v >= 0)
    # Anchor (pick_1 -> v[0])
    if N >= 1:
        constraints_k.append(v[0] == 3000)
    # Monotonicity (v[i] >= v[i+1])
    if N >= 2:
        constraints_k.append(v[:-1] >= v[1:])
    # Convexity/Concavity based on k_py
    # Second difference index i ranges from 0 to N-3
    if N >= 3:
        # Convex part (i = 0 to k_py - 1)
        # This loop runs only if k_py > 0
        for i in range(k_py):
            constraints_k.append(v[i] - 2 * v[i+1] + v[i+2] >= 0)

        # Concave part (i = k_py to N - 3)
        # This loop runs only if k_py <= N-3
        for i in range(k_py, N - 2): # N-2 corresponds to index N-3
            constraints_k.append(v[i] - 2 * v[i+1] + v[i+2] <= 0)

    # 4. Problem Definition
    problem = cp.Problem(objective, constraints_k)

    # 5. Solve
    obj_val = None
    v_val = None
    status = "solver_error" # Default status
    try:
        # Specify a solver if needed (e.g., OSQP, SCS, ECOS)
        # Add 'verbose=True' to see solver output
        # obj_val = problem.solve(solver=cp.SCS, verbose=False)
        obj_val = problem.solve(solver=cp.PIQP, verbose=False)
        status = problem.status
        if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            v_val = v.value
            # Handle case where solver returns optimal but value is None (rare)
            if v_val is None:
                 status = "optimal_but_no_value"
                 obj_val = None
            # Handle cases where objective is NaN or Inf despite optimal status
            elif obj_val is None or not np.isfinite(obj_val):
                 status = "optimal_but_bad_objective"
                 obj_val = None # Treat as non-optimal for selection
                 v_val = None

        else: # Non-optimal status
            obj_val = None
            v_val = None

    except cp.error.SolverError as e:
        print(f" CVXPY Solver Error for k={k_py+1}: {e}") # k_py+1 is the 1-based k
        status = "solver_error"
    except Exception as e:
        print(f" Unexpected Error during solve for k={k_py+1}: {type(e).__name__}: {e}")
        status = "other_error"

    return status, obj_val, v_val


# --- Main Execution Function ---
def main_with_transition():
    # --- Read Data ---
    data_file_path = os.path.join('data', 'nfl_picks.csv')
    print(f"\nAttempting to read data from: {os.path.abspath(data_file_path)}")
    trades = []
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(f"CSV Header: {header}")
            trades = list(csv_reader)
        print(f"Successfully read {len(trades)} trade rows from {data_file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at '{os.path.abspath(data_file_path)}'.")
        print("Please ensure 'data/nfl_picks.csv' exists.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{data_file_path}': {e}")
        return

    # --- Parse Trades & Find N ---
    max_pick = 0
    parsed_trades = []
    successful_parses = 0
    skipped_rows = 0
    print("Parsing trade data...")
    for i, trade_row in enumerate(trades):
        if len(trade_row) < 3:
            # print(f"Warning: Skipping row {i+2} due to insufficient columns: {trade_row}")
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
                    if current_max > max_pick: max_pick = current_max
                except ValueError: pass # Should be handled by parse_picks
        else: skipped_rows += 1
    if max_pick == 0:
        print("Error: No valid picks found. Cannot proceed.")
        return
    N = max_pick
    print(f"\nMaximum pick number found (N): {N}")
    print(f"Number of variables (pick values): {N}")
    print(f"Number of trades successfully parsed: {len(parsed_trades)}")
    if skipped_rows > 0: print(f"Number of rows skipped: {skipped_rows}")

    # Check if N is large enough for second-difference constraints
    if N < 3:
        print(f"\nN={N} is too small for second-difference constraints.")
        print("Cannot run transition point analysis. Consider running original model.")
        # Optionally, call a function here that runs the original model without the transition logic
        return

    # --- Iterate through possible transition points k ---
    best_k = -1         # Store the 1-based k
    best_obj = float('inf')
    best_v = None       # Store the best solution vector np.array
    results_log = []    # Store status/objective for each k

    # k_user (1-based) = k_py (0-based) + 1
    # k_user loops 1 to N-1, so k_py loops 0 to N-2
    num_k_values = N - 1
    print(f"\nIterating through {num_k_values} possible transition points k (1 to {N-1})...")
    start_time_total = time.time()

    for k_py in range(N - 1): # k_py = 0, 1, ..., N-2
        k_user = k_py + 1
        status, obj_val, v_val = solve_for_k(k_py, N, parsed_trades)

        log_entry = {"k": k_user, "status": status, "objective": obj_val if obj_val is not None else "N/A"}
        results_log.append(log_entry)

        # Check if this is the new best valid solution
        if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and obj_val is not None and obj_val < best_obj:
            best_obj = obj_val
            best_k = k_user
            best_v = v_val
            print(f"  k={k_user:<3} -> Status: {status:<20} Objective: {obj_val:<15.4f} * New Best *")
        elif status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and obj_val is not None:
             print(f"  k={k_user:<3} -> Status: {status:<20} Objective: {obj_val:<15.4f}")
        else:
            print(f"  k={k_user:<3} -> Status: {status:<20} Objective: N/A")

    end_time_total = time.time()
    print(f"\nFinished iterating through k values in {end_time_total - start_time_total:.2f} seconds.")

    # --- Process Best Result ---
    if best_k != -1 and best_v is not None:
        print("\n=============================================")
        print(f"Optimal Transition Point found at k = {best_k}")
        print(f"Minimum Objective Value found: {best_obj:.4f}")
        print("=============================================")

        optimal_pick_values = best_v # This is a numpy array

        # --- Print Pick Values (Sample) ---
        print("\nOptimal Pick Values (for k={best_k}, sample shown below):")
        output_lines = []
        for i in range(N):
            val = optimal_pick_values[i] if optimal_pick_values[i] > 1e-7 else 0.0
            output_lines.append(f"  Pick {i+1}: {val:.4f}")
        num_columns = 3; col_width = 20
        num_rows_total = (N + num_columns - 1) // num_columns
        rows_to_print = min(15, num_rows_total)
        for r in range(rows_to_print):
            line = ""
            for c in range(num_columns):
                idx = r + c * num_rows_total
                if idx < N: line += output_lines[idx].ljust(col_width)
            print(line)
        if N > rows_to_print * num_columns: print("  ...")

        # --- Verify Constraints for best_k ---
        print(f"\nVerifying Constraints for k={best_k} (approximate):")
        pick_1_val = optimal_pick_values[0] if N > 0 else float('nan')
        print(f"  pick_1 = {pick_1_val:.4f} (Expected: 3000, Diff: {abs(pick_1_val - 3000):.4e})")
        monotonic = True; min_mono_diff = float('inf')
        if N >= 2:
            mono_diffs = optimal_pick_values[:-1] - optimal_pick_values[1:]
            min_mono_diff = np.min(mono_diffs)
            monotonic = min_mono_diff >= -1e-6
        print(f"  Monotonicity (v_i >= v_{{i+1}}): {monotonic} (Min diff: {min_mono_diff:.4e})")

        # Verify second differences based on best_k
        min_convex_diff = float('inf')
        max_concave_diff = -float('inf')
        convex_ok = True
        concave_ok = True
        if N >= 3:
            second_diffs = optimal_pick_values[:-2] - 2 * optimal_pick_values[1:-1] + optimal_pick_values[2:]
            best_k_py = best_k - 1 # Convert back to 0-based index for checking
            # Check convex part (i=0 to k_py-1)
            if best_k_py > 0:
                 min_convex_diff = np.min(second_diffs[:best_k_py])
                 convex_ok = min_convex_diff >= -1e-6
            # Check concave part (i=k_py to N-3)
            if best_k_py <= N - 3:
                 max_concave_diff = np.max(second_diffs[best_k_py:])
                 concave_ok = max_concave_diff <= 1e-6
        print(f"  Convexity (i < {best_k}): {convex_ok} (Min diff: {min_convex_diff:.4e})")
        print(f"  Concavity (i >= {best_k}): {concave_ok} (Max diff: {max_concave_diff:.4e})")

        non_negative = True; min_nonneg_val = float('inf')
        if N > 0:
            min_nonneg_val = np.min(optimal_pick_values)
            non_negative = min_nonneg_val >= -1e-6
        print(f"  Non-negativity (v_i >= 0): {non_negative} (Min value: {min_nonneg_val:.4e})")

        # --- Save Results to CSV ---
        output_filename = f"pick_table_cvxpy_transition_k{best_k}.csv"
        print(f"\nSaving optimal results (for k={best_k}) to {output_filename}...")
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

    else:
        print("\nNo valid optimal solution found across all possible values of k.")
        print("Review log of results per k:")
        # Optional: Print results_log for detailed debugging
        # for entry in results_log:
        #    print(f"  k={entry['k']}, Status: {entry['status']}, Objective: {entry['objective']}")

# --- Run Main Execution ---
if __name__ == "__main__":
    main_with_transition()
    print(f"\nPython CVXPY Script Finished: {datetime.now()}")
