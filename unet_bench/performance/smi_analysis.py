import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
MAX_MEM = 32768

def smi_analysis(target, mode="hl-smi"):
    ## Val to change for excluding folks' runs

    habana_frames = []
    for frame in load_hl_csv(f"hl-smi-csvs/post-procure/normal-batch.csv", mode="hl-smi", exclude_frames=[]).items():
        ## Other val to change for excluding other folks' runs
        if frame[1].var()["utilization.aip [%]"] > 1.0: # not in []
            ## This one actually need signpost for where to stop by
            habana_frames.append(frame)

    theta_frames = []
    for frame in load_hl_csv("theta-csvs/post-procure/theta-smaller.csv", mode="theta", exclude_frames=[]).items():
        if frame[1].var()["utilization.gpu [%]"] > 1.0: # not in []
            ## This one actually need signpost for where to stop by
            theta_frames.append(("theta run", frame[1]))

    # id_frames = load_hl_csv(f"{mode}-csvs/post-procure/{target}.csv", mode=mode, exclude_frames=[])
    utilz = "utilization.aip" if mode.__eq__("hl-smi") else "utilization.gpu"
    time_name = "time-diff" if mode.__eq__("hl-smi") else "rough-time"

    metrics = [("power.draw", "[W]")]
    # metrics = [tuple(metric.split(' ')) for metric in next(iter(id_frames.values())) if len(metric.split(' ')) > 1]
   

    if ("power.draw", "[W]") in metrics:
        plt.figure(figsize=(10, 4))
        total_power = {}
        for frame in habana_frames:
            total_power[frame[0]] = calculate_curve(frame[1], "power.draw", "W")
            plt.plot(frame[1][time_name], frame[1]["over baseline"], label=f"{frame[0]} model")
            plt.plot(frame[1][time_name], frame[1]["total draw"], label=f"{frame[0]} total")
        for frame in theta_frames:
            total_power[frame[0]] = calculate_curve(frame[1], "power.draw", "W", mode="theta")
            plt.plot(frame[1]["rough-time"], frame[1]["over baseline"], label=f"{frame[0]} model")
            plt.plot(frame[1]["rough-time"], frame[1]["total draw"], label=f"{frame[0]} total")
        plt.legend(loc='upper right')
        plt.xlabel("seconds from start of profiler")
        plt.title("Model power draw and total power draw, batch 64")
        plt.ylabel("W", rotation="horizontal")
        plt.savefig(f"pngs/theta/128-batch-64-power.png")
        plt.show()
        print(total_power)

    # if (utilz, "[%]") in metrics:
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
    time_name = "time-diff" if mode.__eq__("hl-smi") else "rough-time"
    frame.rename(columns={f"{metric} [{unit}]": metric}, inplace=True)
    inactive = np.float64(frame[metric].mode().squeeze())
    frame["running total"] = 0.0
    find_ends(frame, metric)
    over_baseline = 0.0
    total = 0.0
    rows = frame.iterrows()
    idx, row = next(rows)
    prev_val = row[metric]
    prev_time = row[time_name]
    for idx, row in rows:
        if row["indicator"] == 1:
            dt = row[time_name] - prev_time
            over_baseline += dt * ((row[metric] + prev_val) / 2 - inactive)
            total += dt * (row[metric] + prev_val) / 2
        frame.at[idx, "over baseline"] = over_baseline
        frame.at[idx, "total draw"] = total
        prev_time = row[time_name]
        prev_val = row[metric]
    return total

def find_ends(frame: pd.DataFrame, metric: str):
    inactive = np.float64(frame[metric].mode().squeeze())
    counts = frame[metric].value_counts()
    for val in counts.index:
        if abs(val - inactive) > 1:
            scd_bl = val
            break
    frame["indicator"] = 1
    rows = frame.iterrows()
    run_start = False
    for idx, row in rows:
        if not run_start:
            if within(row[metric], [inactive, scd_bl]):
                frame.at[idx, "indicator"] = 0
            else:
                run_start = True
                last_unique = idx
        elif not within(row[metric], [inactive, scd_bl]):
            last_unique = idx
    for i in range(last_unique, idx+1):
        frame.at[i, "indicator"] = 0

def within(value, baselines):
    for baseline in baselines:
        if value <= baseline + 1 and value >= baseline - 1:
            return True
    return False

def utilization_vals(frame: pd.DataFrame, metric, time_name):
    baseline = frame[metric].mode().squeeze()
    totals = frame.groupby(metric).count()
    try:
        totals.drop(labels=[baseline, baseline-1], inplace=True)
    except KeyError:
        totals.drop(labels=baseline, inplace=True)
    
    sum = 0
    total = 0
    for index, row in totals.iterrows():
        total += row[time_name]
        sum += index * row[time_name]
    if total == 0:
        return 0.0, 0
    avg = sum / total
    max = index
    # returning non-baseline average, and max
    return sum / total, index

def load_hl_csv(target: str, mode="hl-smi", exclude_frames=[]) -> dict[pd.DataFrame]:
    all_data = pd.read_csv(target)
    if mode.__eq__("hl-smi"):
        id_frames = {id: pd.DataFrame() for id in all_data['device'].unique() if id not in exclude_frames}
    elif mode.__eq__("theta"):
        id_frames = {id: pd.DataFrame() for id in all_data['bus_id'].unique() if id not in exclude_frames}

    for id in id_frames.keys():
        if mode.__eq__("hl-smi"):
            id_frames[id] = all_data[all_data["device"] == id].copy()
        elif mode.__eq__("theta"):
            id_frames[id] = all_data[all_data["bus_id"] == id].copy()
        id_frames[id].drop(columns=["device", "bus_id"], inplace=True, errors='ignore')
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
