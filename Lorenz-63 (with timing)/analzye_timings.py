import numpy as np
import os
import pickle
import re

# =====================================================================
# Set up
# =====================================================================

root_directory = os.path.dirname(os.path.realpath(__file__))

output_file = os.path.join(root_directory, "L63_optimization_timing_summary.p")


# =====================================================================
# Helper functions
# =====================================================================

def safe_array(values):

    if values is None:
        return np.asarray([], dtype=float)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    return values


def safe_mean(values):

    values = safe_array(values)

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def safe_std(values):

    values = safe_array(values)

    if len(values) <= 1:
        return np.nan

    return float(np.std(values, ddof=1))


def split_by_adaptation(values, adaptation_active_list):

    values = safe_array(values)
    adaptation_active_list = np.asarray(adaptation_active_list, dtype=bool)

    if len(values) != len(adaptation_active_list):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    values_adaptation = values[adaptation_active_list]
    values_reuse = values[~adaptation_active_list]

    return values_adaptation, values_reuse


def read_pickle(filename):

    try:

        return pickle.load(open(filename, "rb"))

    except Exception as e:

        print("Could not read "+filename)
        print(e)

        return None


# =====================================================================
# Collect timing files
# =====================================================================

def collect_timing_rows():

    rows = []

    pattern = re.compile(
        r"TM_filter_N=(\d+)_RS=(\d+)\.p$"
    )

    for path, folders, files in os.walk(root_directory):

        for filename in files:

            match = pattern.match(filename)

            if match is None:
                continue

            full_filename = os.path.join(path, filename)

            result = read_pickle(full_filename)

            if result is None:
                continue

            N = int(match.group(1))
            random_seed = int(match.group(2))

            adaptation_active_list = result.get(
                "adaptation_active_list",
                None
            )

            if adaptation_active_list is None:

                num_adaptation_timesteps = result.get(
                    "num_adaptation_timesteps",
                    100
                )

                duration_list = result.get(
                    "duration_optimization_DA_step_list",
                    result.get("duration_DA_step_list", [])
                )

                adaptation_active_list = [
                    True if t < num_adaptation_timesteps else False
                    for t in range(len(duration_list))
                ]

            # -------------------------------------------------------------
            # Main timing quantities
            # -------------------------------------------------------------

            optimization_list = result.get(
                "duration_optimization_DA_step_list",
                []
            )

            optimization_adaptation_list = result.get(
                "duration_optimization_DA_step_adaptation_list",
                None
            )

            optimization_reuse_list = result.get(
                "duration_optimization_DA_step_reuse_list",
                None
            )

            if optimization_adaptation_list is None or optimization_reuse_list is None:
                optimization_adaptation_list, optimization_reuse_list = split_by_adaptation(
                    optimization_list,
                    adaptation_active_list
                )

            # -------------------------------------------------------------
            # Optional diagnostic timing quantities
            # -------------------------------------------------------------

            outer_list = result.get(
                "duration_outer_optimization_DA_step_list",
                []
            )

            inner_list = result.get(
                "duration_inner_optimization_DA_step_list",
                []
            )

            diagnostics_list = result.get(
                "duration_diagnostics_DA_step_list",
                []
            )

            train_map_list = result.get(
                "duration_train_map_DA_step_list",
                []
            )

            full_DA_step_list = result.get(
                "duration_DA_step_list",
                []
            )

            outer_adaptation_list, outer_reuse_list = split_by_adaptation(
                outer_list,
                adaptation_active_list
            )

            inner_adaptation_list, inner_reuse_list = split_by_adaptation(
                inner_list,
                adaptation_active_list
            )

            diagnostics_adaptation_list, diagnostics_reuse_list = split_by_adaptation(
                diagnostics_list,
                adaptation_active_list
            )

            train_map_adaptation_list, train_map_reuse_list = split_by_adaptation(
                train_map_list,
                adaptation_active_list
            )

            full_adaptation_list, full_reuse_list = split_by_adaptation(
                full_DA_step_list,
                adaptation_active_list
            )

            # -------------------------------------------------------------
            # Store one row per N and random seed
            # -------------------------------------------------------------

            row = {
                "filename"                                      : full_filename,
                "N"                                             : N,
                "random_seed"                                   : random_seed,

                "num_DA_steps"                                  : len(adaptation_active_list),
                "num_adaptation_DA_steps"                       : int(np.sum(adaptation_active_list)),
                "num_reuse_DA_steps"                            : int(np.sum(~np.asarray(adaptation_active_list, dtype=bool))),

                # Main manuscript-facing optimization-only timings
                "optimization_adaptation_seconds_per_DA_step"    : safe_mean(optimization_adaptation_list),
                "optimization_reuse_seconds_per_DA_step"         : safe_mean(optimization_reuse_list),

                # Decomposition of train_map
                "outer_adaptation_seconds_per_DA_step"           : safe_mean(outer_adaptation_list),
                "outer_reuse_seconds_per_DA_step"                : safe_mean(outer_reuse_list),

                "inner_adaptation_seconds_per_DA_step"           : safe_mean(inner_adaptation_list),
                "inner_reuse_seconds_per_DA_step"                : safe_mean(inner_reuse_list),

                "diagnostics_adaptation_seconds_per_DA_step"     : safe_mean(diagnostics_adaptation_list),
                "diagnostics_reuse_seconds_per_DA_step"          : safe_mean(diagnostics_reuse_list),

                "train_map_adaptation_seconds_per_DA_step"       : safe_mean(train_map_adaptation_list),
                "train_map_reuse_seconds_per_DA_step"            : safe_mean(train_map_reuse_list),

                # Original full DA-step timing, for comparison only
                "full_DA_step_adaptation_seconds"                : safe_mean(full_adaptation_list),
                "full_DA_step_reuse_seconds"                     : safe_mean(full_reuse_list)
            }

            rows.append(row)

    return rows


# =====================================================================
# Summarize across random seeds
# =====================================================================

def summarize_rows(rows):

    Ns = sorted(list(set([row["N"] for row in rows])))

    timing_keys = [
        "optimization_adaptation_seconds_per_DA_step",
        "optimization_reuse_seconds_per_DA_step",

        "outer_adaptation_seconds_per_DA_step",
        "outer_reuse_seconds_per_DA_step",

        "inner_adaptation_seconds_per_DA_step",
        "inner_reuse_seconds_per_DA_step",

        "diagnostics_adaptation_seconds_per_DA_step",
        "diagnostics_reuse_seconds_per_DA_step",

        "train_map_adaptation_seconds_per_DA_step",
        "train_map_reuse_seconds_per_DA_step",

        "full_DA_step_adaptation_seconds",
        "full_DA_step_reuse_seconds"
    ]

    summary = {}

    for N in Ns:

        rows_N = [row for row in rows if row["N"] == N]

        summary[N] = {
            "num_random_seeds"             : len(rows_N),
            "num_DA_steps_mean"            : safe_mean([row["num_DA_steps"] for row in rows_N]),
            "num_adaptation_DA_steps_mean" : safe_mean([row["num_adaptation_DA_steps"] for row in rows_N]),
            "num_reuse_DA_steps_mean"      : safe_mean([row["num_reuse_DA_steps"] for row in rows_N])
        }

        for key in timing_keys:

            summary[N][key + "_mean"] = safe_mean([row[key] for row in rows_N])
            summary[N][key + "_std"] = safe_std([row[key] for row in rows_N])

    return summary


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    print("Extracting Lorenz-63 optimization-only timings.")
    print("Root directory: "+root_directory)

    rows = collect_timing_rows()

    if len(rows) == 0:
        raise ValueError("No TM_filter_N=*_RS=*.p files found.")

    summary = summarize_rows(rows)

    output_dictionary = {
        "rows"      : rows,
        "summary"   : summary
    }

    pickle.dump(output_dictionary, open(output_file, "wb"))

    print("")
    print("Lorenz-63 optimization-only timing summary")
    print("")
    print("N      adaptation opt [s/DA step]      reuse opt [s/DA step]      seeds")
    print("--------------------------------------------------------------------------")

    for N in sorted(summary.keys()):

        row = summary[N]

        print(
            str(N).ljust(7)
            + str(round(row["optimization_adaptation_seconds_per_DA_step_mean"], 6)).ljust(32)
            + str(round(row["optimization_reuse_seconds_per_DA_step_mean"], 6)).ljust(28)
            + str(row["num_random_seeds"])
        )

    print("")
    print("Detailed decomposition")
    print("")
    print("N      outer adapt      inner adapt      diagnostics adapt      train_map adapt")
    print("-------------------------------------------------------------------------------")

    for N in sorted(summary.keys()):

        row = summary[N]

        print(
            str(N).ljust(7)
            + str(round(row["outer_adaptation_seconds_per_DA_step_mean"], 6)).ljust(17)
            + str(round(row["inner_adaptation_seconds_per_DA_step_mean"], 6)).ljust(17)
            + str(round(row["diagnostics_adaptation_seconds_per_DA_step_mean"], 6)).ljust(24)
            + str(round(row["train_map_adaptation_seconds_per_DA_step_mean"], 6))
        )

    print("")
    print("N      outer reuse      inner reuse      diagnostics reuse      train_map reuse")
    print("-------------------------------------------------------------------------------")

    for N in sorted(summary.keys()):

        row = summary[N]

        print(
            str(N).ljust(7)
            + str(round(row["outer_reuse_seconds_per_DA_step_mean"], 6)).ljust(17)
            + str(round(row["inner_reuse_seconds_per_DA_step_mean"], 6)).ljust(17)
            + str(round(row["diagnostics_reuse_seconds_per_DA_step_mean"], 6)).ljust(23)
            + str(round(row["train_map_reuse_seconds_per_DA_step_mean"], 6))
        )

    print("")
    print("Wrote "+output_file)