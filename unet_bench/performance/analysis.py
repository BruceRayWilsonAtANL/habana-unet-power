import pandas as pd
import numpy as np
import json
from typing import Tuple, List
import matplotlib.pyplot as plt
from responses import target
from itertools import product

def main():
    plt.close("all")
    
    t_info = []
    t_origs = []
    e_info = []
    e_origs = []
    
    ## These are the things to change when targetting different files
    skeleton = "habana01_layers/256-{}-layer"
    formats = ["3", "4", "5"]
    total_times = dict.fromkeys(formats)
    for format in formats:
        trains, evals, facts = load_runs(skeleton.format(format))
        t_info.append(trains)
        e_info.append(evals)
        total_times[format] = facts

    train_data, train_runs = train_analysis(t_info, formats)
    eval_data, eval_runs = eval_analysis(e_info, formats)
    outside_runs = outsides_analysis(total_times, formats)

    collector = {key: {} for key in train_runs.keys()}
    for run in collector.keys():
        for k, v in train_runs[run].items():
            collector[run][k] = v
        for k, v in eval_runs[run].items():
            collector[run][k] = v
        for k, v in outside_runs[run].items():
            collector[run][k] = v
    
    json_safe = cat_tuples(collector, "layers")
    # print(json_safe)
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


def train_analysis(train_data: pd.DataFrame, formats: List[str]) -> Tuple[pd.DataFrame, dict]:
    runs = {(run, format): dict() for run in range(1,3+1) for format in formats}
    for data in train_data:
        data.drop(columns=["loss_x", "loss_y", "loss"], inplace=True)
    t_info = merge_cols(train_data, ["time_1", "time_2", "time_3"], formats)
    merged_info = pd.DataFrame(t_info["epoch"])
    for format in formats:
        merged_info[f"time_{format}"] = (t_info[f"time_1_{format}"] + t_info[f"time_2_{format}"] + t_info[f"time_3_{format}"]) / 3
    
    loading_time = merged_info.max()
    for format in formats:
        runs[format] = {}
        runs[format]["first run"] = loading_time.get(f'time_{format}')
        for run in range(1, 3+1):
            runs[(run, format)]["first train time"] = t_info.max().get(f"time_{run}_{format}")
    
    merged_info.drop(index=merged_info.idxmax()[f"time_{format}"], inplace=True)
    t_info.drop(index=t_info.idxmax()[f"time_1_{format}"], inplace=True)

    for format in formats:
        runs[format]["average train time"] = merged_info.mean().get(f'time_{format}')
        for run in range(1, 3+1):
            runs[(run, format)]["average train time"] = t_info.mean().get(f"time_{run}_{format}")

    for run, format in product(range(1, 3+1), formats):
        combo = f"time_{run}_{format}"
        runs[(run, format)]["max train time"] = t_info.loc[t_info[combo].idxmax()][combo]
        runs[(run, format)]["min train time"] = t_info.loc[t_info[combo].idxmin()][combo]

    return t_info, runs

def merge_cols(data: pd.DataFrame, rows: List[str], names: List[str]) -> pd.DataFrame:
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

def eval_analysis(eval_data: pd.DataFrame, formats: dict) -> Tuple[pd.DataFrame, dict]:
    runs = {(run, format): dict() for run in range(1,3+1) for format in formats}
    for format in formats:
         runs[format] = dict()
    to_merges = [f"{kind}_{num}" for kind in ["time", "loss", "dsc"] for num in range(1, 3+1)]
    e_info = merge_cols(eval_data, to_merges, formats)
    merged_info = pd.DataFrame(e_info["epoch"])
    for format in formats:
        merged_info[f"loss_{format}"] = (e_info[f"loss_1_{format}"] + e_info[f"loss_2_{format}"] + e_info[f"loss_3_{format}"]) / 3
    
    dsc = "max eval dsc"
    for run, format in product(range(1, 3+1), formats):
        name = f"dsc_{run}_{format}"
        runs[(run, format)][dsc] = e_info.loc[e_info[name].idxmax()][name]
        runs[(run, format)]["max dsc index"] = int(e_info[name].idxmax())
    
    for format in formats:
        avg = (runs[(1, format)][dsc] + runs[(2, format)][dsc] + runs[(3, format)][dsc]) / 3
        runs[format][dsc] = avg

    for run, format in product(range(1, 3+1), formats):
        name = f"loss_{run}_{format}"
        window = e_info.tail(10)
        runs[(run, format)]["ending loss movement"] = window.head()[name].mean() - window.tail()[name].mean()
        
    return e_info, runs

def outsides_analysis(times_data: dict, formats: List[str]) -> dict:
    runs = {(run, format): dict() for run in range(1,3+1) for format in formats}
    for format in formats:
        for run in times_data[format].keys():
            runs[(run, format)] = times_data[format][run].copy()

    for format in formats:
        runs[format] = {}
        for key in times_data[format][run].keys():
            avg = 0.0
            for run in range(1,3+1):
                avg += runs[(run, format)][key]
            runs[format][key] = avg / 3

    return runs

def load_runs(source: str):
    train1, eval1, facts1 = load_blocks(f"csvs/{source}-1.csv")
    train2, eval2, facts2 = load_blocks(f"csvs/{source}-2.csv")
    train3, eval3, facts3 = load_blocks(f"csvs/{source}-3.csv")
    t_info = merge_cols([train1, train2, train3], ["time"], ["1", "2", "3"])
    e_info = merge_cols([eval1, eval2, eval3], ["time", "loss", "dsc"], ["1", "2", "3"])
    return t_info, e_info, {1: facts1, 2: facts2, 3: facts3}

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