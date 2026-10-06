from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from pyprojroot import here

# Par défaut, here() cherche des marqueurs comme .git, pyproject.toml, etc.
root_dir = here()
output_dir = Path(root_dir) / "data"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "breast_cancer_wisconsin.csv"


def fetch_and_save_dataset(
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame]:

# Fetch dataset
    dataset = fetch_ucirepo(id=17) 
    
    X = dataset.data.features 
    y = dataset.data.targets 

    df = pd.concat([X, y], axis=1)
    # print(df.metadata) 
    # print(df.variables) 
    df = pd.concat([X, y], axis=1)

    df.to_csv(output_path, index=False)
    return df, X, y

df, X, y = fetch_and_save_dataset(output_path)