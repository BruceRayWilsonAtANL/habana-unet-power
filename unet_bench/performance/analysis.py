from logging import exception
import pandas as pd
import numpy as np
import json
from typing import Tuple, List
import matplotlib.pyplot as plt
from responses import target
from itertools import product
import sys

MAX_MEM = 32768

def main():
    if len(sys.argv) <= 1:
        exit("usage: analysis.py hl-smi OR analysis.py run")  
    elif sys.argv[1].__eq__("run"):
        run_analysis()
    elif sys.argv[1].__eq__("hl-smi"):
        hl_smi_analysis()
    else:
        exit("usage: analysis.py hl-smi OR analysis.py run")

def hl_smi_analysis():
    id_frames = load_hl_csv()

    metrics = [tuple(metric.split(' ')) for metric in next(iter(id_frames.values())) if len(metric.split(' ')) > 1]
    # example_frame = next(iter(id_frames.values()))
    # for name in example_frame.columns:
        # metrics.append(name)

    # fig = plt.figure()
    frames = []
    for id_frame in id_frames.items():
        frames.append(id_frame)
        # if id_frame[1].var()["utilization.aip [%]"] > 1.0:

    if ("utilization.aip", "[%]") in metrics: #TODO get average util that's above the baseline
        for frame in frames:
            print(utilization_vals(frame[1]))
            plt.plot(frame[1]["time-diff"], frame[1]["utilization.aip [%]"], label=frame[0])

        plt.legend()
        plt.xlabel("utilization.aip")
        plt.ylabel("%")
        plt.show()
        
    if ("power.draw", "[W]") in metrics:
        plt.close()
        for frame in frames:
            plt.plot(frame[1]["time-diff"], frame[1]["power.draw [W]"], label=frame[0])

        plt.legend()
        plt.xlabel("power.draw")
        plt.ylabel("[W]")
        plt.show()

    # if ('memory.used', '[%]') in metrics: #TODO needs spans of time to be useful doubtful even then
    #     for frame in frames:
    #         print(frame[1].groupby('memory.used [%]').count())

def utilization_vals(frame: pd.DataFrame):
    #TODO get length (in time) of non-baseline
    metric = "utilization.aip [%]"
    baseline = frame[metric].mode().squeeze()
    totals = frame.groupby(metric).count()
    try:
        totals.drop(labels=[baseline, baseline-1], inplace=True)
    except KeyError:
        totals.drop(labels=baseline, inplace=True)
    
    sum = 0
    total = 0
    for index, row in totals.iterrows():
        total += row["time-diff"]
        sum += index * row["time-diff"]
    if total == 0:
        return 0.0, 0
    avg = sum / total
    max = index
    # returning non-baseline average, and max
    return sum / total, index

def load_hl_csv() -> dict[pd.DataFrame]:
    target = "hl-smi-csvs/post-procure/dup-cache.csv"
    all_data = pd.read_csv(target)
    id_frames = {id: pd.DataFrame() for id in all_data['device'].unique()}

    for id in id_frames.keys():
        id_frames[id] = all_data[all_data["device"] == id].copy()
        id_frames[id].drop(columns="device", inplace=True)
        try:
            id_frames[id]["memory.used [MiB]"] = id_frames[id]["memory.used [MiB]"] / MAX_MEM
            id_frames[id].rename(columns={"memory.used [MiB]": "memory.used [%]"}, inplace=True)
        except KeyError:
            pass
        try:
            id_frames[id]["power.draw [W]"] = id_frames[id]["power.draw [W]"]
        except KeyError:
            pass

    return id_frames    

def run_analysis():
    plt.close("all")
    
    t_info = []
    t_origs = []
    e_info = []
    e_origs = []
    
    ## These are the things to change when targetting different files
    skeleton = "habana01_no_cache/256-{}-cache"
    formats = ["no"]
    run_numbers = ["1", "2", "3"]

    total_times = dict.fromkeys(formats)
    for format in formats:
        trains, evals, facts = load_runs(skeleton.format(format), run_numbers)
        t_info.append(trains)
        e_info.append(evals)
        total_times[format] = facts

    train_data, train_runs = train_analysis(t_info, formats, run_numbers)
    eval_data, eval_runs = eval_analysis(e_info, formats, run_numbers)
    outside_runs = outsides_analysis(total_times, formats, run_numbers)

    collector = {key: {} for key in train_runs.keys()}
    for run in collector.keys():
        for k, v in train_runs[run].items():
            collector[run][k] = v
        for k, v in eval_runs[run].items():
            collector[run][k] = v
        for k, v in outside_runs[run].items():
            collector[run][k] = v
    
    json_safe = cat_tuples(collector, "cache")
    print(json.dumps(json_safe, indent=2))
    
    # run_names = [f"{kind}_{num}_{format}" for kind in ["loss"] for num in range(1, 3+1) for format in formats]
    # train_data.plot(x="epoch", y=run_names)
    # loading_time.plot(kind='bar')
    # merged_info.plot(x="epoch")
    # plt.xticks(rotation='horizontal')
    # plt.show()

def cat_tuples(data: dict, affix: str) -> dict:
    dup = {}
    for key, value in data.items():
        dup_key = f"{affix} {key}"
        if type(key) is tuple:
            dup_key = f"{affix} {key[1]}: run {key[0]}"
        dup[dup_key] = value
    dup = dict(sorted(dup.items(), key=lambda v: v[0]))
    return dup


def train_analysis(train_data: pd.DataFrame, formats: List[str], run_names: List[str]) -> Tuple[pd.DataFrame, dict]:
    runs = {(run, format): dict() for run in run_names for format in formats}
    for data in train_data:
        data.drop(columns=["loss_x", "loss_y", "loss"], inplace=True)
    t_info = merge_cols(train_data, [f"time_{name}" for name in run_names], formats)
    merged_info = pd.DataFrame(t_info["epoch"])
    for format in formats:
        merged_info[f"time_{format}"] = (t_info[f"time_2"] + t_info[f"time_3"] + t_info[f"time_1"]) / 3
    
    loading_time = merged_info.max()
    for format in formats:
        runs[format] = {}
        runs[format]["first run"] = loading_time.get(f'time_{format}')
        for run in run_names:
            runs[(run, format)]["first train time"] = t_info.max().get(f"time_{run}")
    
    merged_info.drop(index=merged_info.idxmax()[f"time_{format}"], inplace=True)
    t_info.drop(index=t_info.idxmax()[f"time_2"], inplace=True)

    for format in formats:
        runs[format]["average train time"] = merged_info.mean().get(f'time_{format}')
        for run in run_names:
            runs[(run, format)]["average train time"] = t_info.mean().get(f"time_{run}")

    for run, format in product(run_names, formats):
        if len(formats) > 1:
            combo = f"time_{run}_{format}"
        else:
            combo = f"time_{run}"
        runs[(run, format)]["max train time"] = t_info.loc[t_info[combo].idxmax()][combo]
        runs[(run, format)]["min train time"] = t_info.loc[t_info[combo].idxmin()][combo]

    return t_info, runs

def merge_cols(data: List[pd.DataFrame], rows: List[str], names: List[str]) -> pd.DataFrame:
    if len(data) == 1:
        return data[0]
    tmp = data[0]
    for entry in data[1:]:
        tmp = tmp.merge(entry, on="epoch")

    for row in rows:
        tmp.rename(columns = {
            f"{row}_x": f"{row}_{names[0]}",
            f"{row}_y": f"{row}_{names[1]}",
            f"{row}": f"{row}_{names[2]}"
        }, inplace=True)
    
    return tmp

def eval_analysis(eval_data: pd.DataFrame, formats: dict, run_names: List[str]) -> Tuple[pd.DataFrame, dict]:
    runs = {(run, format): dict() for run in run_names for format in formats}
    for format in formats:
         runs[format] = dict()
    to_merges = [f"{kind}_{num}" for kind in ["time", "loss", "dsc"] for num in run_names]
    e_info = merge_cols(eval_data, to_merges, formats)
    merged_info = pd.DataFrame(e_info["epoch"])
    for format in formats:
        merged_info[f"loss_{format}"] = (e_info[f"loss_2"] + e_info[f"loss_3"] + e_info[f"loss_1"]) / 3
    
    dsc = "max eval dsc"
    for run, format in product(run_names, formats):
        if len(formats) > 1:
            name = f"dsc_{run}_{format}"
        else:
            name = f"dsc_{run}"
        runs[(run, format)][dsc] = e_info.loc[e_info[name].idxmax()][name]
        runs[(run, format)]["max dsc index"] = int(e_info[name].idxmax())
    
    for format in formats:
        avg = (runs[('2', format)][dsc] + runs[('3', format)][dsc] + runs[('1', format)][dsc]) / 3
        runs[format][dsc] = avg

    for run, format in product(run_names, formats):
        if len(formats) > 1:
            name = f"loss_{run}_{format}"
        else:
            name = f"loss_{run}"
        window = e_info.tail(10)
        runs[(run, format)]["ending loss movement"] = window.head()[name].mean() - window.tail()[name].mean()
        
    return e_info, runs

def outsides_analysis(times_data: dict, formats: List[str], run_names: List[str]) -> dict:
    runs = {(run, format): dict() for run in run_names for format in formats}
    for format in formats:
        for run in times_data[format].keys():
            runs[(run, format)] = times_data[format][run].copy()
    for format in formats:
        runs[format] = {}
        for run in times_data[format].keys():
            avg = 0.0
            for key in times_data[format][run].keys():
                avg += runs[(run, format)][key]
            runs[format][key] = avg / 3

    return runs

def load_runs(source: str, run_names: List[str]):
    trains = []
    evals = []
    facts = []
    for name in run_names:
        train, eval, fact = load_blocks(f"csvs/{source}-{name}.csv")
        trains.append(train)
        evals.append(eval)
        facts.append(fact)

    t_info = merge_cols(trains, ["time"], run_names)
    e_info = merge_cols(evals, ["time", "loss", "dsc"], run_names)
    return t_info, e_info, {str(i+1): facts[i] for i in range(len(facts))}

def load_blocks(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = pd.read_csv(filepath)
    t_info = data[data["type"] == "train"].drop(columns=['type', 'dsc'])
    e_info = data[data["type"] == "eval"].drop(columns=['type'])
    loader_time = data.at[data[data["type"] == "loaders_init"].index[0], "time"]
    total_train = data.at[data[data["type"] == "total_train_time"].index[0], "time"]
    total_eval = data.at[data[data["type"] == "total_eval_time"].index[0], "time"]
    return t_info, e_info, {"loader_init": loader_time, "total_train": total_train, "total_eval": total_eval}
    

if __name__ == '__main__':
    main()