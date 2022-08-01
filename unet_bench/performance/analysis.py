from logging import exception
import pandas as pd
import numpy as np
import json
from typing import Tuple, List
import matplotlib.pyplot as plt
from responses import target
from itertools import product
import sys
from smi_analysis import smi_analysis

def main():
    if len(sys.argv) <= 1:
        exit("usage: analysis.py run OR analysis.py <hl-smi/nvidia-smi> <target-csv>")  
    elif sys.argv[1].__eq__("run"):
        run_analysis()
    elif len(sys.argv) <= 2:
        exit("usage: analysis.py run OR analysis.py <hl-smi/nvidia-smi> <target-csv>")  
    elif sys.argv[1].__eq__("hl-smi"):
        smi_analysis(sys.argv[2], mode="hl-smi")
    elif sys.argv[1].__eq__("nvidia-smi"):
        smi_analysis(sys.argv[2], mode="theta")
    else:
        exit("usage: analysis.py run OR analysis.py <hl-smi/nvidia-smi> <target-csv>")

def run_analysis():
    plt.close("all")
    
    t_info = []
    t_origs = []
    e_info = []
    e_origs = []
    
    ## These are the things to change when targetting different files
    skeleton = "habana02_large_batch/128-{}-64"
    # skeleton = "theta/128-{}-theta"
    formats = ["batch"]
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
    
    json_safe = cat_tuples(collector, "64")
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
        data.drop(columns=["loss_x", "loss_y", "loss"], inplace=True, errors='ignore')
    if len(train_data) == 1:
        t_info = train_data[0]
    else:
        t_info = merge_cols(train_data, [f"time_{name}" for name in run_names], formats)
    merged_info = pd.DataFrame(t_info["epoch"])
    for format in formats:
        tmp = t_info["time_1"].copy()
        for name in run_names[1:]:
            tmp += t_info[f"time_{name}"]
        merged_info[f"time_{format}"] = tmp / len(run_names)
        # merged_info[f"time_{format}"] = (t_info[f"time_2"] + t_info[f"time_3"] + t_info[f"time_1"]) / 3
    
    loading_time = merged_info.max()
    for format in formats:
        runs[format] = {}
        runs[format]["first epoch"] = loading_time.get(f'time_{format}')
        for run in run_names:
            runs[(run, format)]["first epoch train time"] = t_info.max().get(f"time_{run}")
    
    merged_info.drop(index=merged_info.idxmax()[f"time_{format}"], inplace=True)
    t_info.drop(index=t_info.idxmax()[f"time_{run}"], inplace=True)

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
        data[0].rename(columns = {f"{row}": f"{row}_{names[0]}" for row in rows}, inplace=True)
        return data[0]
    tmp = data[0].copy()
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
    to_merges = [f"{kind}_{num}" for kind in ["time", "loss", "dsc"] for num in run_names]
    if len(eval_data) == 1:
        e_info = eval_data[0]
    else:
        e_info = merge_cols(eval_data, to_merges, formats)
    merged_info = pd.DataFrame(e_info["epoch"])
    for format in formats:
        tmp_frame = e_info["loss_1"]
        for i in range(1, len(run_names)):
            tmp_frame += e_info[f"loss_{i}"]
        merged_info[f"loss_{format}"] = tmp_frame / len(run_names)
    
    for format in formats:
        runs[format] = dict()

    dsc = "max eval dsc"
    for run, format in product(run_names, formats):
        if len(formats) > 1:
            name = f"dsc_{run}_{format}"
        else:
            name = f"dsc_{run}"
        runs[(run, format)][dsc] = e_info.loc[e_info[name].idxmax()][name]
        runs[(run, format)]["max dsc index"] = int(e_info[name].idxmax())
    
    for format in formats:
        tmp_frame = runs[(run_names[0], format)][dsc].copy()
        for run in run_names[1:]:
            tmp_frame += runs[(run, format)][dsc]
        avg = tmp_frame / len(run_names)
        runs[format][dsc] = avg

    for run, format in product(run_names, formats):
        if len(formats) > 1:
            name = f"loss_{run}_{format}"
        else:
            name = f"loss_{run}"
        
    return e_info, runs

def outsides_analysis(times_data: dict, formats: List[str], run_names: List[str]) -> dict:
    runs = {(run, format): dict() for run in run_names for format in formats}
    for format in formats:
        for run in times_data[format].keys():
            runs[(run, format)] = times_data[format][run].copy()

    for format in formats:
        runs[format] = {}
        for key in times_data[format][run].keys():
            avg = 0.0
            for name in run_names:
                avg += runs[(name, format)][key]
            runs[format][key] = avg / len(run_names)

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