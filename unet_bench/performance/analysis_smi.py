from gettext import find
from typing import Tuple, List, Callable
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

MAX_MEM = 32768
#HABANA_CSV = "csvs/habana02_theta_comapre"
#HABANA_CSV = "csvs/habana02_theta_compare"
#THETA_CSV = "csvs/theta"
OUT_DIR = "pngs/theta-v-habana"
OVERWRITE = False # Included since it's very easy to overwrite old, useful data.

# Always in flux to graph whatever targeted. Not really worth making consistent,
# instead marking major blocks.
def smi_analysis(mode_="all"):
    """
    The main function of this file. Runs everything, and is very mutable (read: messy).
    This is the function to change when trying to output more data.
    Arguments:
        mode:   Whether to chart model, total (baseline + model), or both sets of values.
                    Choices: ["model", "total", "all"]
    Returns:    Nothing. Generates a plt figure and sometimes a png.
    """

    # These are the general frame lists to create and view.
    frame_avgs = []
    habana_frames = []
    theta_frames = []

    metrics = ["power.draw", "utilization.aip"] # <- Utilization is .gpu not .aip for Theta. Be careful.
    units = ["W", "%"]

    # Change these variables in every run, as well as which frames are getting produced and viewed.
    # TODO: Change the 'run' variable.
    run = "resnet50_1"

    input_run = input(f"Enter run name [{run}]: ")
    if len(input_run) > 0:
        run = input_run


    plot_title = f"{mode_.capitalize()} power usage"

    input_plot_title = input(f"Enter plot title [{plot_title}]:")
    if len(input_plot_title) > 0:
        plot_title = input_plot_title

    outfile_name = f"{OUT_DIR}/{run}-{mode_}.png"
    print(f"\nsmi_analysis.outfile: {outfile_name}")

    # These lists get commented and uncommented, and the function values changed, with every dataset to display.
    # Hence a number of things currently commented out

    name = run.replace('_', ' ')
    run_name = run
    mode = "hl-smi"
    excludeFrames = [0]  # This removes a good frame so, do not do it.
    excludeFrames = []
    habana_frames = filter_frames(name, run_name, mode, exclude_frames=excludeFrames)
    print(f"smi_analysis.habana_frames: {habana_frames}")

    original = False
    if original:
        filtered_frames = filter_frames("64 max", "64-max", "hl-smi")
        averaged_run = average_run(filtered_frames, metrics, units)

        frame_avgs.append(("habana", averaged_run))

        # single_frames = filter_frames("single card", "duped-128", "hl-smi")
        # frame_avgs.append(("single-card", average_run(single_frames, metrics, units)))

        theta_frames = filter_frames("theta", run, "theta")
        for frame in theta_frames: # Theta device names are ugly
            frame = ("theta", frame[1])

        # Utilization actually isn't all that useful. For now just track power.
        # This expansion is useful for if there's more/other metrics. Right now, there aren't really.
        # metrics = [tuple(metric.split(' ')) for metric in next(iter(id_frames.values())) if len(metric.split(' ')) > 1]

    # The block of code that generates the figure(s).
    # Could (and previously has been) a loop for every metric. Since it's currently just power, didn't bother.
    time_name = "time-diff"
    if "power.draw" in metrics:
        plt.figure(figsize=(10, 4))
        # Plot all the frame data that should be shown.
        # Comment out any of these if their data isn't desired (or in existence) for the chart.
        # Example: for the example png, habana_frames just adds clutter. Would comment out usually,
        #habana_frames = []

        # Create the curve for "power.draw".
        metric = "power.draw"
        unit   = "W"

        for frameNum, frame in enumerate(habana_frames):
            #csvOutPath = f'frame_{frameNum}'
            clean_curve(frame[0], frame[1], metric, unit, boundary=1)
            if mode_ in ["all", "model"]:
                plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            if mode_ in ["all", "total"]:
                plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in theta_frames:
            clean_curve(frame[0], frame[1], metric, unit, num_bases=2)
            if mode_ in ["all", "model"]:
                plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            if mode_ in ["all", "total"]:
                plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in frame_avgs:
            clean_curve(frame[0], frame[1], metric, unit)
            if mode_ in ["all", "model"]:
                plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            if mode_ in ["all", "total"]:
                plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")

        # Add legend label and titles. Most of these should be modified in variables at the start.
        plt.legend(loc='upper left')
        plt.title(plot_title)
        plt.xlabel("Time since start of profiler (S)")
        plt.ylabel("Power (W)", rotation="horizontal")
        ## Enable and rename once there's a satisfying figure. Be wary to not overwrite past figures.
        if not os.path.isfile(outfile_name) or OVERWRITE:
            plt.savefig(outfile_name)
        plt.show()


def filter_frames(name: str, run_name: str, mode: str, utilz=None, exclude_frames=[]) -> Tuple[str, pd.DataFrame]:
    """
    Loads and filters the frames from the noisy data.
    Arguments:
        name: The string which to tag these frames with.
        run_name: The string which is the filepath to the csv file holding the frames.
        mode: A string indicating whether this data was generated on Theta or Habana.
        utilz: A lambda function to filter the utilization metric's name, default none.
        exclude_frames: A list to target certain frames for exclusion, default empty.
    Returns: A list of the frames, loaded and filtered, each tagged with a name.
    """

    print(f"\nfilter_frames.name: {name}")
    print(f"filter_frames.run_name: {run_name}")
    print(f"filter_frames.mode: {mode}")

    # The different profilers might have different names that wasn't dealt with in data cleaning. This utilz is one fix.
    if utilz is None:
        utilz = lambda mode: "utilization.aip [%]" if mode.__eq__("hl-smi") else "utilization.gpu [%]"

    frames = []

    for frame in load_hl_csv(run_name, mode=mode, exclude_frames=exclude_frames).items():
        if frame[1].var()[utilz(mode)] > 1.0: # Filters most inactive cards, does not catch foreign runners on those cards.
            # Keep data for card utilization > 1.0.
            frames.append((f"{name} {frame[0]}", frame[1]))
    return frames


def average_run(frames: List[pd.DataFrame], metrics: List[str], units: List[str]) -> pd.DataFrame:
    """
    Averages multiple frames of different runs together into one average.
    Arguments:
        frames: A list of the dataframe smi data.
        metrics: A list of the metrics to track.
        units: A list of the metrics' units, sharing ordering.
    Returns: the combined dataframe.
    """
    dup = frames[0][1].copy()
    for metric in zip(metrics, units):
        metric = f"{metric[0]} [{metric[1]}]"
        for frame in frames[1:]:
            dup[metric] += frame[1][metric]
        dup[metric] /= len(frames)
    dup.dropna(inplace=True)
    return dup


def clean_curve(
    name: str,
    frame: pd.DataFrame,
    metric: str,
    unit: str,
    boundary=10,
    num_bases=2,
    inplace=True
    ) -> pd.DataFrame:
    """
    Helper function for the main function. Bundles the curve calculation and deleteion functions together,
    which is handy since they're usually called in sequence.
    This also catches and exits if it detects a bad frame.
    Arguments:
        name: String name of the frame.
        frame: The dataframe to clean up.
        metric: The string for which metric to target for cleaning.
        unit: The string for what unit that metric uses.
        boundary: The time difference to keep on both ends of the active period, default 10.
        num_bases: The number of baselines to look for in inactive time, default 2.
        inplace: A boolean for whether the operations should be inplace, default True.
    Returns: The new frame if not inplace, otherwise nothing.
    """

    dumpData = True
    if dumpData:
        print(f'name: {name}')

        csvOutPath = 'frame_dump.csv'
        frame.to_csv(csvOutPath, index=False)
        print(f'frame: {frame}')

        #columns = ['historic_close', 'future_close', 'historic_day', 'historic_time', 'future_day', 'future_time', 'symbol' ]
        #dfPrice = pd.read_csv(priceFilename, names=columns)

        print(f'metric: {metric}')
        print(f'unit: {unit}')
        print(f'boundary: {boundary}')
        print(f'num_bases: {num_bases}')
        print(f'inplace: {inplace}')
        input('Press <Enter> to continue...')



    if not inplace:
        frame = frame.copy()

    # Note the inner three are always inplace to the newly created frame.
    try:
        calculate_curve(frame, metric, unit, num_bases=num_bases)
        delete_indicated_start(frame, boundary=boundary)
        delete_indicated_end(frame, boundary=boundary)
    except IndexError as e:
        exit(f"\nCaught frame {name} in an error:\n{e}\nSuggested response: exclude the frame.\n")
    if not inplace:
        return frame


def delete_indicated_start(frame: pd.DataFrame, boundary=10, inplace=True) -> pd.DataFrame:
    """
    Deletes everything briefly before the indicator starts for the first time
    Arguments:
        frame: The dataframe to delete from.
        boundary: An integer for how long before the start to cut off.
        inplace: A boolean for whether it should be inplace.
    Returns: the new frame, if not inplace, else nothing.
    """
    start = frame[frame["indicator"] == 1]["time-diff"].iloc[0]
    mask = (frame["indicator"] == 0) & (frame["time-diff"] < start - boundary)
    if not inplace:
        frame = frame.copy()
    frame.drop(index=frame[mask].index, inplace=inplace)
    frame["time-diff"] = frame["time-diff"] - start
    if not inplace:
        return frame


def delete_indicated_end(frame: pd.DataFrame, boundary=10, inplace=True) -> pd.DataFrame:
    """
    Deletes everything briefly after the indicator shuts off for the last time
    Arguments:
        frame: The dataframe to delete from.
        boundary: An integer for how long after the ending to cut off.
        inplace: A boolean for whether it should be inplace.
    Returns: the new frame, if not inplace, else nothing.
    """
    end = frame[frame["indicator"] == 1]["time-diff"].iloc[-1]
    mask = (frame["indicator"] == 0) & (frame["time-diff"] > end + boundary)
    if not inplace:
        frame = frame.copy()
    frame.drop(index=frame[mask].index, inplace=inplace)
    if not inplace:
        return frame


def smoothed_curve(frame: pd.DataFrame, metric: str, unit: str, num_bases=1):
    """
    An old function for utilization. Unused, since utilization went unused.
    Strictly inplace in the current version.
    Arguments:
        frame: The dataframe which to smooth the curve from.
        metric: The string for which metric to target for cleaning.
        unit: The string for what unit that metric uses.
        num_bases: The number of baselines to look for in inactive time, default 2.
    Returns: nothing, as it's strictly inplace
    """
    time_name = "time-diff"
    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)
    find_ends(frame, metric, num_bases=num_bases)
    powers = []
    rows = frame.iterrows()
    idx, row = next(rows)
    prev_time = row[time_name]
    for idx, row in rows:
        dt = row[time_name] - prev_time
        prev_time = row[time_name]
        powers.append(dt * row[metric])
        frame.at[idx, "recent util"] = sum(powers[-10:-1]) / min(len(powers), 10)


def calculate_curve(frame: pd.DataFrame, metric: str, unit: str, num_bases=1, inplace=True):
    """
    Calculates the overall/cumulative metric curve across the whole run
    Arguments:
        frame: The dataframe to calculate on.
        metric: The string metric name to calculate.  E.g., 'power.draw'.
        unit: The string for the metric's unit.  E.g., 'W'.
        num_bases: The number of baselines to treat as inactive, default 1.
        inplace: Whether to modify the original frame or, default true.
    Returns: Adds rows 'total draw' and 'over baseline' to frame, returns it if not inplace.
    """

    print('\ncalulate_curve:')
    print(f'calculate_curve.metric: {metric}')
    print(f'calculate_curve.unit: {unit}')

    if not inplace:
        frame = frame.copy()

    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)

    # Grab the baseline for this function's calcs before calling find_ends.
    inactiveBaseline = np.float64(frame[metric].mode().squeeze())
    print(f'calculate_curve.inactiveBaseline: {inactiveBaseline}')

    find_ends(frame, metric, num_bases=num_bases)
    bases = find_bases(frame, metric, num_bases=num_bases)


    # Running calculation values and loop setup
    over_baseline = 0.0
    total = 0.0
    rows = frame.iterrows()
    idx, row = next(rows)
    prev_val = row[metric]
    prev_time = row["time-diff"]

    # Calculate the curve under the whole active run.
    for idx, row in rows:
        rowMetric = row[metric]

        if _within(rowMetric, bases):
            inactiveBaseline = rowMetric

        # Only use active rows. Does clip off half the very last entry but not a big error.
        if row["indicator"] == 1:
            dt = row["time-diff"] - prev_time
            over_baseline += dt * ((rowMetric + prev_val) / 2 - inactiveBaseline)
            total += dt * (rowMetric + prev_val) / 2

        # Add calculations to their points to graph later.
        frame.at[idx, "over baseline"] = over_baseline
        frame.at[idx, "total draw"] = total
        prev_time = row["time-diff"]
        prev_val = rowMetric

    if not inplace:
        return frame


def find_bases(frame: pd.DataFrame, metric: str, num_bases=1, bounds=1):
    """
    Gets the baselines of some metric for this frame. Useful since the data can be quite noisey.
    Arguments:
        frame: The dataframe to scrub.
        metric: The metric to scan by.
        bases: The number of baselines to treat as inactive, default 1.
        bounds: An integer to use for the sensetivity of an integer being in bounds, default 1.
    Returns: The baselines (most common values) for the specified metric.
    """
    # Get the first baseline
    bases = [np.float64(frame[metric].mode().squeeze())]

    # Get a second baseline that's distinct from the first.f
    # Sometimes this is necessary when the machine's default shifts mid-run.
    for i in range(1, num_bases):
        # Count each number's appearances, then use that as an index to file new baselines
        for val in frame[metric].value_counts().index:
            if not _within(val, bases, bounds=bounds):
                bases.append(val)
                break
    return bases


def find_ends(frame: pd.DataFrame, metric: str, num_bases=1, inplace=True, bounds=1):
    """
    Mostly helper function that finds the active stretch of csv metrics.
    Arguments:
        frame: The dataframe to scrub.
        metric: The metric to scan by.
        num_bases: The number of baselines to treat as inactive, default 1.
        inplace: A boolean whether to modify og frame or use a new one, default true.
        bounds: An integer to use for the sensetivity of an integer being in bounds, default 1.
    Returns: nothing, or a frame if not inplace.
    """
    bases = find_bases(frame, metric, num_bases=num_bases)
    if not inplace:
        frame = frame.copy()

    # Add an indicator row to the dataframe
    # 0 means outside of run, 1 means inside of run. Default is 1
    frame["indicator"] = 1
    run_start = False
    last_active = 0

    # Iterate through the rows to find where activity begins to designate a run start
    rows = frame.iterrows()
    for idx, row in rows:

        # Mark every inactive row with a 0
        if not run_start:
            if _within(row[metric], bases, bounds=bounds):
                frame.at[idx, "indicator"] = 0
            # Start the run, add last_active to track the end
            else:
                run_start = True
                last_active = idx
        # Every new known to be active value, update last_active index
        elif not _within(row[metric], bases, bounds=bounds):
            last_active = idx

    # Add a last iteration to mark post-run rows
    for i in range(last_active, idx+1):
        frame.at[i, "indicator"] = 0
    if not inplace:
        return frame


def _within(value: int, baselines: List[int], bounds=1) -> bool:
    """
    Helper method for find_ends and find_bases. Boolean for if a value is roughly around any baseline.
    Arguments:
        value: The value to check.
        baselines: The baselines to compare value to.
        bounds: The sensitivity of the baselines, default 1.
    Returns: True if the value is within the sensitivity of any baseline, false if not.
    """
    for baseline in baselines:
        if value <= baseline + bounds and value >= baseline - bounds:
            return True
    return False


# Old method, probably not to be used.
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

    # Earlier version of find_ends. Should replace
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


#def load_hl_csv(filename: str, mode="hl-smi", exclude_frames=[]) -> dict[pd.DataFrame]:
def load_hl_csv(filename: str, mode="hl-smi", exclude_frames=[]):
    """
    Loads a specific hl csv file, returning a dict of frames, one for each device polled.
    Arguments:
        filename: csv file to read from
        mode: which mode this csv is based on. One of ["hl-smi", "smi"].
        exclude_frames: frames to ignore for one reason or another. Most commonly
            because they have been identified to dirty graphs to generate after.
    Returns: a dict of DataFrames
    """

    print(f"\n\nload_hl_csv.filename: {filename}")
    print(f"load_hl_csv.mode: {mode}")
    print(f"load_hl_csv.exclude_frames: {exclude_frames}")

    # Load the data into a single larger dataframe.
    loc = os.path.dirname(os.path.abspath(__file__))

    filepath = f"poll-data/{mode}/post-procure/{filename}.csv"
    fullFilepath = os.path.join(loc, filepath)
    print(f"load_hl_csv.fullFilePath: {fullFilepath}")
    # /home/bwilson/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/poll-data/hl-smi/post-procure/64-max.csv

    # FileNotFoundError: [Errno 2] No such file or directory: '/home/bwilson/DL/github.com/BruceRayWilsonAtANL/habana-unet-power/unet_bench/performance/poll-data/hl-smi/post-procure/habana_init_test.csv'

    all_data = pd.read_csv(fullFilepath)

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
            # There is a runtime habana memory metric in htorch. That would probably produce better results.
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
