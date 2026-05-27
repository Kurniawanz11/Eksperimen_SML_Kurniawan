import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing():

    # Load dataset
    df = pd.read_csv("../dataset_raw/heart.csv")


    # Hapus duplikat
    df = df.drop_duplicates()


    # Pisah fitur dan target
    X = df.drop("target", axis=1)
    y = df["target"]


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Save hasil preprocessing
    pd.DataFrame(X_train_scaled).to_csv(
        "X_train.csv",
        index=False
    )

    pd.DataFrame(X_test_scaled).to_csv(
        "X_test.csv",
        index=False
    )

    y_train.to_csv(
        "y_train.csv",
        index=False
    )

    y_test.to_csv(
        "y_test.csv",
        index=False
    )


    print("Preprocessing selesai")


if __name__ == "__main__":
    preprocessing()