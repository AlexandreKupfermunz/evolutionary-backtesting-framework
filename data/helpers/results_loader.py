from pathlib import Path
import pandas as pd

## results/rolling/robust_fitness/ONE_MONTH/rep_1/walk_forward_results.csv
## results/expanding/robust_fitness/rep_1/walk_forward_results.csv

def extract_metadata(csv_path):

    path = Path(csv_path)

    metadata = {}

    is_rolling = False 

    for parent in path.parents:
        if parent.name == "rolling":
            is_rolling = True

    if is_rolling:
        metadata.update({ 
            "walk_forward_type": "rolling",
            "fitness_function": path.parent.parent.parent.name,
            "train_size_name": path.parent.parent.name,
            "repetition": path.parent.name
        })
    else:
        metadata.update({ 
            "walk_forward_type": "expanding",
            "fitness_function": path.parent.parent.name,
            "train_size_name": None,
            "repetition": path.parent.name
        })

    metadata.update({"csv_type": path.stem})

    return metadata

def load_single_csv(csv_path):

    metadata = extract_metadata(csv_path)

    df = pd.read_csv(csv_path)

    for key, value in metadata.items():
        df[key] = value

    return df
       
def load_all_csvs(results_folder, csv_filename):

    csv_dataframes = []

    for path in results_folder.rglob(csv_filename):
        csv_dataframes.append(load_single_csv(path))

    if len(csv_dataframes) == 0:
        return pd.DataFrame()

    df = pd.concat(csv_dataframes, ignore_index=True)

    return df