from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
MAX_MEM = 32768

# This is the main method of the file, and also a giant mess.
# Always in flux to graph whatever targeted. Not really worth making consistent, instead marking major blocks.
def smi_analysis(target, mode="hl-smi"):
    habana_frames = []
    utilz = lambda mode: "utilization.aip [%]" if mode.__eq__("hl-smi") else "utilization.gpu [%]"

    ## Major block 1: creating frames
    ## Most common changes are in filename and exclude_frames
    for frame in load_hl_csv(f"hl-smi-csvs/post-procure/normal-batch.csv", mode="hl-smi", exclude_frames=[]).items():
        if frame[1].var()[utilz("hl-smi")] > 1.0: # not in [] # Alternative to exclude_frames
            habana_frames.append(frame)

    ## The same thing but demonstrating distinct frame lists. Which also happens a lot
    theta_frames = []
    for frame in load_hl_csv("theta-csvs/post-procure/theta-smaller.csv", mode="theta", exclude_frames=[]).items():
        if frame[1].var()[utilz("theta")] > 1.0: # not in []
            # ID often changes cause theta runs have worse names, and to distinguish
            theta_frames.append(("theta run", frame[1]))

    ## This is the default version
    # frames = []
    # id_frames = load_hl_csv(f"{mode}-csvs/post-procure/{target}.csv", mode=mode, exclude_frames=[])
    # for frame in id_frames:
        # if frame[1].var()["utilization.gpu [%]"] > 1.0: # not in []
            # frames.append((frame)

    # TODO would be awesome to remove or make a constant
    time_name = "time-diff" if mode.__eq__("hl-smi") else "rough-time"

    metrics = [("power.draw", "[W]")]
    # metrics = [tuple(metric.split(' ')) for metric in next(iter(id_frames.values())) if len(metric.split(' ')) > 1]
   
    ## The other major block, utilization version below
    if ("power.draw", "[W]") in metrics:
        plt.figure(figsize=(10, 4))
        total_power = {}
        # Plot all the frame data that should be shown
        for frame in habana_frames:
            total_power[frame[0]] = calculate_curve(frame[1], "power.draw", "W")
            plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in theta_frames:
            total_power[frame[0]] = calculate_curve(frame[1], "power.draw", "W", mode="theta")
            plt.plot(frame[1]["rough-time"], frame[1]["over baseline"], label=f"{frame[0]} model")
            plt.plot(frame[1]["rough-time"], frame[1]["total draw"], label=f"{frame[0]} total")
        
        # Add legend label and titles
        plt.legend(loc='upper right')
        plt.title("Model power draw and total power draw, batch 64")
        plt.xlabel("Seconds from start of profiler")
        plt.ylabel("W", rotation="horizontal")
        ## Enable and rename once there's a satisfying figure. Be wary to not overwrite past figures.
        # plt.savefig(f"pngs/theta/128-batch-64-power.png")
        plt.show()

        # Print sums cause why not
        print(total_power)

    ## This one looks more standard
    # if (utilz(mode), "[%]") in metrics:
    #     plt.figure(figsize=(10, 4))
    #     for frame in frames:
    #         # print(utilization_vals(frame[1], f"{utilz} [%]", time_name))
    #         plt.plot(frame[1][time_name], frame[1][f"{utilz} [%]"], label=frame[0])

    #     plt.legend(loc='upper right')
    #     plt.xlabel("seconds from start of profiler")
    #     plt.title("Normal Model Utilization ThetaGPU and Habana")
    #     plt.ylabel("%", rotation="horizontal")
    #     # plt.savefig(f"pngs/hl-smi/habana-v-theta-util.png")
    #     plt.show()
    

    # if ('memory.used', '[%]') in metrics: #TODO needs spans of time to be useful doubtful even then
    #     for frame in frames:
    #         print(frame[1].groupby('memory.used [%]').count())


def calculate_curve(frame: pd.DataFrame, metric: str, unit: str, mode="hl-smi"):
    """
    Calculates the overall metric curve across the whole run
    Arguments:
        frame: The dataframe to calculate on.
        metric: The metric name which to calculate.
        unit: The metric's unit.
        mode: Which type of run it is. Replacable once time is consistent
    Returns: a total sum, also adds rows 'total draw' and 'over baseline' to frame.
    """
    # TODO replace this when time changes
    time_name = "time-diff" if mode.__eq__("hl-smi") else "rough-time"
    # TODO move this renaming somewhere else so unit isn't in this function.
    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)

    # Grab the baseline for this function's calcs before calling find_ends.
    inactive = np.float64(frame[metric].mode().squeeze())
    find_ends(frame, metric)
    
    # TODO think can be removed? 
    # frame["running total"] = 0.0
    # Running calculation values and loop setup
    over_baseline = 0.0
    total = 0.0
    rows = frame.iterrows()
    idx, row = next(rows)
    prev_val = row[metric]
    prev_time = row[time_name]
    # Calculate the curve under the whole active run.
    for idx, row in rows:
        # Only use active rows. Does clip off half the very last entry but not a big error.
        if row["indicator"] == 1:
            dt = row[time_name] - prev_time
            over_baseline += dt * ((row[metric] + prev_val) / 2 - inactive)
            total += dt * (row[metric] + prev_val) / 2
        # Add calculations to their points to graph later.
        frame.at[idx, "over baseline"] = over_baseline
        frame.at[idx, "total draw"] = total
        prev_time = row[time_name]
        prev_val = row[metric]
    return total


def find_ends(frame: pd.DataFrame, metric: str, num_bases=1, inplace=True):
    """
    Mostly helper function that finds the active stretch of csv metrics.
    Arguments:
        frame: The dataframe to scrub.
        metric: The metric to scan by. Shouldn't matter that much.
        num_bases: The number of baselines to treat as inactive, default 1.
        inplace: A boolean whether to modify og frame or use a new one, defaults true
    Returns: nothing, or a frame if not inplace.
    """
    # Get the first baseline
    bases = [np.float64(frame[metric].mode().squeeze())]
    if not inplace:
        frame = frame.copy()

    # Get a second baseline that's distinct from the first.
    # Sometimes this is necessary when the machine's default shifts mid-run.
    for i in range(1, num_bases):
        # Count each number's appearances, then use that as an index to file new baselines
        for val in frame[metric].value_counts().index:
            if _within(val, bases):
                bases.append(val)
                break

    # Add an indicator row to the dataframe
    # 0 means outside of run, 1 means inside of run. Default is 1
    frame["indicator"] = 1
    run_start = False

    # Iterate through the rows to find where activity begins to designate a run start
    rows = frame.iterrows()
    for idx, row in rows:

        # Mark every inactive row with a 0
        if not run_start:
            if _within(row[metric], bases):
                frame.at[idx, "indicator"] = 0
            # Start the run, add last_active to track the end
            else:
                run_start = True
                last_active = idx
        # Every new known to be active value, update last_active index
        elif not _within(row[metric], bases):
            last_active = idx

    # Add a last iteration to mark post-run rows
    for i in range(last_active, idx+1):
        frame.at[i, "indicator"] = 0
    
    if not inplace:
        return frame


# Helper method for find_ends. Checks if a value is roughly around any baseline.
def _within(value: int, baselines: List[int], bounds=1) -> bool:
    for baseline in baselines:
        if value <= baseline + bounds and value >= baseline - bounds:
            return True
    return False


def utilization_vals(frame: pd.DataFrame, metric: str, time_name: str):
    """
    Counts overall utilzation of the run. Currently unused. Should use find_ends like calculate_curve.
    Currently treats default utilization as none instead of weighting down.
    Arguments:
        frame: the dataframe to poll the utilization from.
        metric: a string, should be utilization
        time-name: a string that should get removed eventually
    Returns: a tuple, total utilization & max utilization
    """

    # TODO Earlier version of find_ends. Should replace
    baseline = frame[metric].mode().squeeze()
    totals = frame.groupby(metric).count()
    try:
        totals.drop(labels=[baseline, baseline-1], inplace=True)
    except KeyError:
        totals.drop(labels=baseline, inplace=True)
    
    # Calculate and return a basic percentage, based on time and % utils
    sum = 0
    total = 0
    for index, row in totals.iterrows():
        total += row[time_name]
        sum += index * row[time_name]
    if total == 0:
        return 0.0, 0
    # returning non-baseline average, and maximum utilization point.
    return sum / total, index


def load_hl_csv(filename: str, mode="hl-smi", exclude_frames=[]) -> dict[pd.DataFrame]:
    """
    Loads a specific hl csv file, returning a dict of frames, one for each device polled.
    Arguments:
        filename: csv file to read from
        mode: which mode this csv is based on
        exclude_frames: frames to ignore for one reason or another. Most commonly
            because they have been identified to dirty graphs to generate after.
    Returns: a dict of DataFrames
    """
    # Load the data into a single larger dataframe.
    all_data = pd.read_csv(os.path.join(os.path.abspath(__file__), f"{mode}-csv/post-procure/{filename}.csv"))

    # Create dict and frames based off the large one.
    # TODO renameing bus_id to be consistent would simplify, but there may be a better name so postpone that.
    if mode.__eq__("hl-smi"):
        id_frames = {id: pd.DataFrame() for id in all_data['device'].unique() if id not in exclude_frames}
    elif mode.__eq__("theta"):
        id_frames = {id: pd.DataFrame() for id in all_data['bus_id'].unique() if id not in exclude_frames}
    
    # populate each frame
    for id in id_frames.keys():
        # TODO see above
        # Use devices for ids then remove from the dataframe.
        if mode.__eq__("hl-smi"):
            id_frames[id] = all_data[all_data["device"] == id].copy()
        elif mode.__eq__("theta"):
            id_frames[id] = all_data[all_data["bus_id"] == id].copy()
        id_frames[id].drop(columns=["device", "bus_id"], inplace=True, errors='ignore')

        # For each of the metrics use a try/except block, as they're not in every file.
        try:
            id_frames[id]["memory.used [MiB]"] = id_frames[id]["memory.used [MiB]"] / MAX_MEM
            id_frames[id].rename(columns={"memory.used [MiB]": "memory.used [%]"}, inplace=True)
        except KeyError:
            pass
        try:
            id_frames[id]["power.draw [W]"] = id_frames[id]["power.draw [W]"]
        except KeyError:
            pass
        id_frames[id].index = np.arange(len(id_frames[id]))

    return id_frames
