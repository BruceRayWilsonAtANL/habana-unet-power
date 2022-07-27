import os
import shutil


new_dir = "../data/kaggle_duped"
for root, dirs, files in os.walk("../data/kaggle_3m"):
    for dir in dirs:
        for i in range(4):
            try:
                os.mkdir(os.path.join(new_dir, f"{dir}_{i}"))
            except FileExistsError:
                pass
        # os.mkdir(os.join)
    parent = os.path.split(root)[1]
    for i in range(4):
        target_dir = f"{parent}_{i}"
        if '.DS_Store' in files:
            continue
        for filename in files:
            shutil.copy(f"{root}/{filename}", f"{new_dir}/{target_dir}/{filename}")


# shutil