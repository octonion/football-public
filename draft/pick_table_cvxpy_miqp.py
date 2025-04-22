import cvxpy as cp
import numpy as np
import csv
import os
import time
from datetime import datetime

print(f"Python CVXPY MIQP Script Started: {datetime.now()}")
print(f"CVXPY Version: {cp.__version__}")
print(f"NumPy Version: {np.__version__}")

# --- Data Parsing Function ---
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

# --- Main MIQP Function ---
def main_miqp():
    # --- Read Data ---
    data_file_path = os.path.join('data', 'nfl_picks.csv')
    print(f"\nAttempting to read data from: {os.path.abspath(data_file_path)}")
    trades = []
    try:
        # Check if file exists before attempting to open
        if not os.path.isfile(data_file_path):
             raise FileNotFoundError(f"File not found at '{os.path.abspath(data_file_path)}'.")
        with open(data_file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            print(f"CSV Header: {header}")
            trades = list(csv_reader)
        print(f"Successfully read {len(trades)} trade rows from {data_file_path}")
    except FileNotFoundError as e:
        print(e)
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
        if len(trade_row) < 3: skipped_rows += 1; continue
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
                except ValueError: pass
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
        print("Cannot run MIQP transition point analysis.")
        return

    # --- Define MIQP Problem ---
    print("\nDefining the CVXPY MIQP problem...")

    # Big-M constant (needs to be large enough to not constrain the inactive side)
    # Should be larger than the maximum possible absolute value of v[i]-2v[i+1]+v[i+2]
    # Since max v[i] is 3000, a value around 2*3000 or more should be safe.
    M = 10000
    print(f"Using Big-M value: {M}")

    # 1. Variables
    v = cp.Variable(N, name="pick_values")
    # Binary variables y[i]: y[i]=1 => convex at i; y[i]=0 => concave at i
    # Index i runs from 0 to N-3 (for N-2 second differences)
    y = cp.Variable(N-2, boolean=True, name="convex_indicator")

    # 2. Objective Function (same as before)
    objective_terms = []
    for side1, side2 in parsed_trades:
        valid_side1_indices = [p - 1 for p in side1 if 1 <= p <= N]
        valid_side2_indices = [p - 1 for p in side2 if 1 <= p <= N]
        sum1_expr = cp.sum(v[valid_side1_indices]) if valid_side1_indices else 0.0
        sum2_expr = cp.sum(v[valid_side2_indices]) if valid_side2_indices else 0.0
        term = cp.square(sum1_expr - sum2_expr)
        objective_terms.append(term)

    if not objective_terms:
        print("Error: No objective terms generated from trades.")
        return
    objective = cp.Minimize(cp.sum(objective_terms))

    # 3. Constraints
    constraints = []
    # Basic constraints
    constraints.append(v >= 0)      # Non-negativity
    if N >= 1: constraints.append(v[0] == 3000) # Anchor
    if N >= 2: constraints.append(v[:-1] >= v[1:]) # Monotonicity

    # Constraints linking y variables (ensure transition happens at most once)
    # y[i] >= y[i+1] for i=0..N-4 ensures y starts at 1 and flips to 0 at most once
    if N >= 4: # Need at least y[0] and y[1] for this constraint
         constraints.append(y[:-1] >= y[1:])

    # Big-M constraints for second differences
    # v[i] - 2v[i+1] + v[i+2] >= -M * (1 - y[i])
    # v[i] - 2v[i+1] + v[i+2] <= M * y[i]
    # Index i runs from 0 to N-3
    print(f"Adding {2 * (N-2)} Big-M constraints for second differences...")
    for i in range(N - 2):
        expr_i = v[i] - 2 * v[i+1] + v[i+2]
        # If y[i] is 1 (convex): Need expr_i >= 0. Becomes expr_i >= -M*(0)=0 and expr_i <= M*1=M (loose)
        # If y[i] is 0 (concave): Need expr_i <= 0. Becomes expr_i >= -M*(1)=-M (loose) and expr_i <= M*0=0
        constraints.append(expr_i >= -M * (1 - y[i]))
        constraints.append(expr_i <= M * y[i])

    # 4. Problem Definition
    problem = cp.Problem(objective, constraints)
    print("Problem defined successfully.")

    # --- Solve using CPLEX ---
    print("\nAttempting to solve MIQP using CPLEX...")
    # Check if CPLEX is available
    if cp.CPLEX not in cp.installed_solvers():
        print("Error: CPLEX solver not found by CVXPY.")
        print("Please ensure CPLEX Optimization Studio is installed and the Python API")
        print("is correctly installed in your environment (e.g., via setup.py install or PYTHONPATH).")
        print(f"Installed solvers: {cp.installed_solvers()}")
        return

    # Set CPLEX parameters (optional)
    # Example: Set a relative MIP gap tolerance and a time limit
    cplex_params = {
        "mip.tolerances.mipgap": 0.001,  # Stop when gap is 0.1% (adjust as needed)
        "timelimit": 300                # Stop after 300 seconds (5 minutes)
        # Add other CPLEX parameters here if needed
        # Reference: IBM CPLEX Parameters documentation
    }
    print(f"Using CPLEX parameters: {cplex_params}")

    obj_val = None
    v_val = None
    y_val = None
    status = "solver_error"
    start_time = time.time()

    try:
        # Add 'verbose=True' to see detailed CPLEX log output
        #obj_val = problem.solve(solver=cp.SCIP, verbose=True)
        obj_val = problem.solve(solver=cp.CPLEX, verbose=True, cplex_params=cplex_params)
        status = problem.status
        if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            v_val = v.value
            y_val = y.value # Get the values of the binary variables
            # Handle potential None values even if status is optimal
            if v_val is None or y_val is None:
                status = "optimal_but_no_value"
                obj_val = None
            elif obj_val is None or not np.isfinite(obj_val):
                 status = "optimal_but_bad_objective"
                 obj_val = None; v_val = None; y_val = None
        else:
            obj_val = None # Ensure obj_val is None if not optimal

    except cp.error.SolverError as e:
        print(f"CVXPY Solver Error with CPLEX: {e}")
        status = "solver_error"
    except Exception as e:
        print(f"Unexpected Error during CPLEX solve: {type(e).__name__}: {e}")
        status = "other_error"

    end_time = time.time()
    print(f"\nCPLEX solve finished in {end_time - start_time:.2f} seconds.")
    print(f"Solver status: {status}")

    # --- Process Results ---
    if status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and v_val is not None and y_val is not None:
        status_suffix = "(inaccurate)" if status == cp.OPTIMAL_INACCURATE else ""
        print(f"\nOptimal solution found {status_suffix}.")
        print(f"Minimum Objective Value found: {obj_val:.4f}")

        optimal_pick_values = v_val
        optimal_indicators = np.round(y_val).astype(int) # Round binary results to 0 or 1

        # Determine the transition point k from the y values
        # k is the first 1-based index where y transitions to 0.
        # If all y are 1, the transition doesn't happen (k = N-1).
        best_k = N - 1 # Default to full convexity
        for i in range(N - 2):
            if optimal_indicators[i] == 0:
                best_k = i + 1 # Found first concave point (1-based index)
                break
        print(f"Implied Transition Point at k = {best_k}")
        # print(f"Binary indicators (y): {optimal_indicators}") # Optional: print y values

        # --- Print Pick Values (Sample) ---
        print(f"\nOptimal Pick Values (for k={best_k}, sample shown below):")
        # ...(Same printing logic as before)...
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
        print(f"\nVerifying Constraints for implied k={best_k} (approximate):")
        # ...(Same verification logic as before, using best_k)...
        pick_1_val = optimal_pick_values[0] if N > 0 else float('nan')
        print(f"  pick_1 = {pick_1_val:.4f} (Expected: 3000, Diff: {abs(pick_1_val - 3000):.4e})")
        monotonic = True; min_mono_diff = float('inf')
        if N >= 2:
            mono_diffs = optimal_pick_values[:-1] - optimal_pick_values[1:]
            min_mono_diff = np.min(mono_diffs)
            monotonic = min_mono_diff >= -1e-6
        print(f"  Monotonicity (v_i >= v_{{i+1}}): {monotonic} (Min diff: {min_mono_diff:.4e})")
        min_convex_diff = float('inf'); max_concave_diff = -float('inf')
        convex_ok = True; concave_ok = True
        if N >= 3:
            second_diffs = optimal_pick_values[:-2] - 2 * optimal_pick_values[1:-1] + optimal_pick_values[2:]
            best_k_py = best_k - 1
            if best_k_py > 0: min_convex_diff = np.min(second_diffs[:best_k_py]); convex_ok = min_convex_diff >= -1e-6
            if best_k_py <= N - 3: max_concave_diff = np.max(second_diffs[best_k_py:]); concave_ok = max_concave_diff <= 1e-6
        print(f"  Convexity (i < {best_k}): {convex_ok} (Min diff: {min_convex_diff:.4e})")
        print(f"  Concavity (i >= {best_k}): {concave_ok} (Max diff: {max_concave_diff:.4e})")
        non_negative = True; min_nonneg_val = float('inf')
        if N > 0: min_nonneg_val = np.min(optimal_pick_values); non_negative = min_nonneg_val >= -1e-6
        print(f"  Non-negativity (v_i >= 0): {non_negative} (Min value: {min_nonneg_val:.4e})")


        # --- Save Results to CSV ---
        output_filename = f"pick_table_cvxpy_miqp_k{best_k}.csv"
        print(f"\nSaving optimal results (for implied k={best_k}) to {output_filename}...")
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
        print("\nCPLEX did not find an optimal solution.")
        print("Check solver output and problem formulation.")

# --- Run Main Execution ---
if __name__ == "__main__":
    # Ensure CPLEX prerequisites are understood
    print("--- MIQP Formulation using CVXPY and CPLEX ---")
    print("NOTE: This script requires CPLEX Optimization Studio to be installed")
    print("      and the CPLEX Python API to be configured for your environment.")
    main_miqp()
    print(f"\nPython CVXPY MIQP Script Finished: {datetime.now()}")
