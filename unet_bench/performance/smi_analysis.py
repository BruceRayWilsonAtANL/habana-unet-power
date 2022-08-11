from gettext import find
from typing import Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
MAX_MEM = 32768
HABANA_CSV = "csvs/habana02_large_batch"
THETA_CSV = "csvs/theta"

# This is the main method of the file, and also a giant mess.
# Always in flux to graph whatever targeted. Not really worth making consistent, instead marking major blocks.
def smi_analysis(mode="hl-smi"):
    frame_avgs = []
    habana_frames = []
    utilz = lambda mode: "utilization.aip [%]" if mode.__eq__("hl-smi") else "utilization.gpu [%]"
    metrics = [("power.draw", "[W]"), ("utilization.aip", "[%]")]
    run = "64-max"

    ## Major block 1: creating frames
    ## Most common changes are in filename and exclude_frames
    for frame in load_hl_csv(run, mode="hl-smi", exclude_frames=[]).items():
        if frame[1].var()[utilz("hl-smi")] > 1.0: # not in [] # Alternative to exclude_frames
            # epoch_slice(frame[1], "128-duped-1", 1)
            # smoothed_curve(frame[1], "utilization.aip", "%")
            habana_frames.append((f"64max-{frame[0]}", frame[1]))
    
    # frame_avgs.append(("habana", average_run(habana_frames, metrics)))
    
    single_frames = []
    # for frame in load_hl_csv("duped-128", mode="hl-smi", exclude_frames=[]).items():
        # if frame[1].var()[utilz("hl-smi")] > 1.0: # not in [] # Alternative to exclude_frames
            # single_frames.append(frame)
    # frame_avgs.append(("single-card", average_run(single_frames, metrics)))
            

    # frame_avgs.append((32, average_run(frames_32, metrics)))
    # frame_avgs.append(("duped 64", average_run(duped_64, metrics)))

    ## The same thing but demonstrating distinct frame lists. Which also happens a lot
    theta_frames = []
    for frame in load_hl_csv(run, mode="theta", exclude_frames=[]).items():
        if frame[1].var()[utilz("theta")] > 1.0: # not in []
            # ID often changes cause theta runs have worse names, and to distinguish

            # smoothed_curve(frame[1], "utilization.gpu", "%")
            theta_frames.append(("theta", frame[1]))
            # theta_frames.append(("theta", frame[1]))

    ## This is the default version
    frames = []
    # id_frames = load_hl_csv(f"{target}.csv", mode=mode, exclude_frames=[])
    # for frame in id_frames:
        # if frame[1].var()["utilization.gpu [%]"] > 1.0: # not in []
            # frames.append((frame)

    time_name = "time-diff"

    metrics = [("power.draw", "[W]")]
    # metrics = [tuple(metric.split(' ')) for metric in next(iter(id_frames.values())) if len(metric.split(' ')) > 1]
   
    ## The other major block, utilization version below
    if ("power.draw", "[W]") in metrics:
        plt.figure(figsize=(10, 4))
        total_power = {}
        # Plot all the frame data that should be shown
        for frame in habana_frames:
            calculate_curve(frame[1], "power.draw", "W")

            delete_indicated_start(frame[1], boundary=1)
            delete_indicated_end(frame[1], boundary=1)
            # plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            # plt.plot(frame[1][time_name], frame[1]["recent util"], label=f"{frame[0]} recency")
            plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in frame_avgs:
            # if frame[0].__eq__("duped 128"):
            #     calculate_curve(frame[1], "power.draw", "W", num_bases=3)
            # else:
            #     calculate_curve(frame[1], "power.draw", "W", num_bases=2)
            calculate_curve(frame[1], "power.draw", "W")
            # print(frame[1])
            # exit()
            delete_indicated_start(frame[1])
            delete_indicated_end(frame[1])
            # plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            # plt.plot(frame[1][time_name], frame[1]["recent power"], label=f"{frame[0]} recency")
            # plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in theta_frames:
            ## Re-enable to see if the theta should increase num bases. Can get very positive or negative results if left at 1.
            calculate_curve(frame[1], "power.draw", "W", num_bases=2)
            delete_indicated_start(frame[1])
            delete_indicated_end(frame[1])
            # plt.plot(frame[1][time_name], frame[1]["recent util"], label=f"{frame[0]} recency")
            # plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        
        # Add legend label and titles
        plt.legend(loc='upper left')
        plt.title("Model power usage, 50x dataset, 2 epochs, size 64")
        plt.xlabel("Seconds from start of profiler")
        plt.ylabel("W", rotation="horizontal")
        ## Enable and rename once there's a satisfying figure. Be wary to not overwrite past figures.
        # plt.savefig(f"pngs/upscale/all-large-habana-combined.png")
        plt.show()

        # Print these sums later
        # print(total_power)

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


# Helper method for the main function, averages frames together.
def average_run(frames: List[pd.DataFrame], metrics: List[str]) -> pd.DataFrame:
    dup = frames[0][1].copy()
    for frame in frames[1:]:
        dup["power.draw [W]"] += frame[1]["power.draw [W]"]
    dup["power.draw [W]"] /= len(frames)
    dup.dropna(inplace=True)
    return dup


def delete_indicated_start(frame: pd.DataFrame, inplace=True, boundary=10) -> pd.DataFrame:
    """
    Deletes everything briefly before the indicator starts for the first time
    Arguments:
        frame: The dataframe to delete from.
        inplace: A boolean for whether it should be inplace.
    Returns: the new frame, if not inplace, else nothing.
    """
    start = frame[frame["indicator"] == 1]["time-diff"].iloc[0]
    mask = (frame["indicator"] == 0) & (frame["time-diff"] < start - boundary)
    if not inplace:
        frame = frame.copy()
    frame.drop(index=frame[mask].index, inplace=True)
    frame["time-diff"] = frame["time-diff"] - start
    if not inplace:
        return frame


def delete_indicated_end(frame: pd.DataFrame, inplace=True, boundary=10) -> pd.DataFrame:
    """
    Deletes everything briefly after the indicator shuts off for the last time
    Arguments:
        frame: The dataframe to delete from.
        inplace: A boolean for whether it should be inplace.
    Returns: the new frame, if not inplace, else nothing.
    """
    end = frame[frame["indicator"] == 1]["time-diff"].iloc[-1]
    mask = (frame["indicator"] == 0) & (frame["time-diff"] > end + boundary)
    dup = frame.drop(index=frame[mask].index, inplace=inplace)
    if not inplace:
        return dup


def smoothed_curve(frame: pd.DataFrame, metric: str, unit: str, num_bases=1):
    time_name = "time-diff"
    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)
    find_ends(frame, metric, num_bases=num_bases)
    current = 0.0
    powers = []
    rows = frame.iterrows()
    idx, row = next(rows)
    prev_time = row[time_name]
    for idx, row in rows:
        dt = row[time_name] - prev_time
        prev_time = row[time_name]
        powers.append(dt * row[metric])
        frame.at[idx, "recent util"] = sum(powers[-10:-1]) / min(len(powers), 10)


def calculate_curve(frame: pd.DataFrame, metric: str, unit: str, num_bases=1):
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
    time_name = "time-diff"
    # TODO move this renaming somewhere else so unit isn't in this function.
    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)

    # Grab the baseline for this function's calcs before calling find_ends.
    inactive = np.float64(frame[metric].mode().squeeze())
    find_ends(frame, metric, num_bases=num_bases)
    bases = find_bases(frame, metric, num_bases=num_bases)
    
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
    # TODO need to actively change the inactive time value
    for idx, row in rows:
        if _within(row[metric], bases):
            inactive = row[metric]
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
    # exit()
    return total


def find_bases(frame: pd.DataFrame, metric: str, num_bases=1, boundary=1):
    # Get the first baseline
    bases = [np.float64(frame[metric].mode().squeeze())]

    # Get a second baseline that's distinct from the first.f
    # Sometimes this is necessary when the machine's default shifts mid-run.
    for i in range(1, num_bases):
        # Count each number's appearances, then use that as an index to file new baselines
        for val in frame[metric].value_counts().index:
            if not _within(val, bases):
                bases.append(val)
                break
    return bases


def find_ends(frame: pd.DataFrame, metric: str, num_bases=1, inplace=True, boundary=1):
    """
    Mostly helper function that finds the active stretch of csv metrics.
    Arguments:
        frame: The dataframe to scrub.
        metric: The metric to scan by. Shouldn't matter that much.
        num_bases: The number of baselines to treat as inactive, default 1.
        inplace: A boolean whether to modify og frame or use a new one, defaults true
    Returns: nothing, or a frame if not inplace.
    """
    bases = find_bases(frame, metric, num_bases=num_bases)
    if not inplace:
        frame = frame.copy()

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
    loc = os.path.dirname(os.path.abspath(__file__))
    all_data = pd.read_csv(os.path.join(loc, f"{mode}-csvs/post-procure/{filename}.csv"))

    # Create dict and frames based off the large one.
    id_frames = {id: pd.DataFrame() for id in all_data['device'].unique() if id not in exclude_frames}
    
    # populate each frame
    for id in id_frames.keys():
        # TODO see above
        # Use devices for ids then remove from the dataframe.
        id_frames[id] = all_data[all_data["device"] == id].copy()
        id_frames[id].drop(columns=["device"], inplace=True)

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
