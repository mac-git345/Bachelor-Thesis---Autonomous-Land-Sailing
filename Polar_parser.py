import pandas as pd
import numpy as np

def parse(file_name, seperation):
    df = pd.read_csv(file_name, sep=seperation)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: "TWA"})
    df = df.set_index("TWA")

    # Nur die gewünschten TWA-Werte (0 bis 180 in 45er Schritten)
    twa_target = list(range(0, 181, 5))
    df = df[df.index.isin(twa_target)]

    # Nur die gewünschten TWS-Werte (0 bis 10 in 2er Schritten)
    tws_target = list(range(0, 11, 2))
    df = df[[col for col in df.columns if float(col) in tws_target]]

    # In numpy-Matrix umwandeln und von Knoten in m/s (1 kn = 0.514444 m/s)
    matrix_ms = df.to_numpy(dtype=float) * 0.514444

    return matrix_ms

if __name__ == "__main__":
    print(parse("A35.csv", ";"))