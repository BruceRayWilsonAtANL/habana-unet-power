import sys
import json
import math
import pandas as pd
from collections import OrderedDict

DATA_DICT ="analyzer_metadata"
COLS = ["op", "duration (usec)", "active (usec)", "mme util (%)", "input vector util", "exposed tpc (usec)", 
        "exposed dma (usec)", "MME Compute Utilization (%).", "MME Expected Compute (cycles)."]


def main():
    data_mode = sys.argv[1]
    if data_mode == "dirty":
        data = load_from_dirty()
        with open('cleaned.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif data_mode == "clean":
        data = load_from_clean()
    else:
        exit("usage: test_json.py <dirty/clean>")
    
    # print(json.dumps(get_row(data[1], 50), indent=2))
    operators = total_commands(data)
    # print(json.dumps(operators,indent=2))
    rows = sieve_command("", data[1])
    dists = get_rows_dists(rows)

    # print(json.dumps(dists,indent=2))

    pandas_test(dists)


def load_from_dirty():
    data = {}
    for i in range(1, 4+1):
        with open(f"default_profiling_hl0_{str(i).zfill(4)}.json") as json_file:
            data[i] = json.load(json_file)

    cured_data = {int(k) : {} for k in data.keys()}
    
    # Build a list element for every file
    for k in data.keys():
        cured_data[k] = {c: [] for c in COLS}
        # Separates the massive rows
        for row in data[k][DATA_DICT]:
            # Grabs each of the respective values from the column
            for col in COLS:
                # if row[col].__eq__(""):
                #     cured_data[k][col].append("empty")
                # else:
                cured_data[k][col].append(row[col])
    return cured_data

def load_from_clean():
    with open("cleaned.json") as json_file:
        data = json.load(json_file)
    reconstruct = {}
    for idx, log in data.items():
        reconstruct[int(idx)] = log
    return reconstruct

def total_commands(data):
    operators = {}
    for idx, log in data.items():
        operators[idx] = {}
        for i in range(len(log["op"])):
            oper = log["op"][i]
            if oper in operators[idx]:
                operators[idx][oper] += 1
            else:
                operators[idx][oper] = 1

        operators[idx] = {k: v for k, v in sorted(operators[idx].items(), key=lambda item: item[1], reverse=True)}
    return operators

def sieve_command(command, data):
    rows = []
    for i in range(len(data["op"])):
        if data["op"][i].__eq__(command):
            found_row = {col: data[col][i] for col in COLS}
            rows.append(found_row)
    return rows

def get_rows_dists(data):
    columns = {c : {} for c in COLS}
    for row in data:
        for col, value in row.items():
            try:
                value = math.ceil(value * 10) / 10.0
            except TypeError:
                pass

            if value in columns[col]:
                columns[col][value] += 1
            else:
                columns[col][value] = 1
    divisor = len(data)
    for col in COLS:
        # for entry, weight in columns[col].items():
            # columns[col][entry] = weight / divisor
        # try:
        #     columns[col] = {float(k): v for k, v in sorted(columns[col].items(), key=lambda item: item[1], reverse=True)}
        # except ValueError:
        columns[col] = {k: v for k, v in sorted(columns[col].items(), key=lambda item: item[1], reverse=True)}
    del columns["input vector util"]
    return columns

def get_row(data, idx):
    row_vals = {c : None for c in COLS}
    for col in COLS:
        row_vals[col] = data[col][idx]
    return row_vals

def pandas_test(dist):
    basic = pd.Series(dist)
    print(basic)
    # basic.plot()

    
    pass

if __name__ == "__main__":
    main()

#TODO look at keys for the columns again
    # print(f"{i}: {test[n]['input shapes']} -> {test[n]['output shapes']}")