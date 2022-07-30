import sys
import requests
import numpy as np
import pandas as pd

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN" \
          "-SkillsNetwork/labs/Data%20files/auto.csv"
filename = "dataset_auto.csv"


def download_dataset(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("File has been downloaded")
    else:
        print("File has not been downloaded", file=sys.stderr)
        print("Status code:", response.status_code, file=sys.stderr)


def get_prepared_data():
    download_dataset(url, filename)

    df = pd.read_csv(filename, header=None)
    headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
               "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight",
               "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio",
               "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
    df.columns = headers

    # clean data
    df.replace('?', np.nan, inplace=True)
    # drop rows without price
    df.dropna(subset=["price"], axis=0, inplace=True)
    # replace rows without number of doors by frequency
    df["num-of-doors"].replace(np.nan, df["num-of-doors"].value_counts().idxmax(), inplace=True)
    # replace other rows with missing data by mean
    df["peak-rpm"].replace(np.nan, df["peak-rpm"].astype("float").mean(axis=0), inplace=True)
    df["horsepower"].replace(np.nan, df["horsepower"].astype("float").mean(axis=0), inplace=True)
    df["bore"].replace(np.nan, df["bore"].astype("float").mean(axis=0), inplace=True)
    df["stroke"].replace(np.nan, df["stroke"].astype("float").mean(axis=0), inplace=True)
    df["normalized-losses"].replace(np.nan, df["normalized-losses"].astype("float").mean(axis=0), inplace=True)

    # print(df.dtypes)
    # Convert data from strings to numbers
    df[["bore", "stroke", "price", "peak-rpm"]] = df[["bore", "stroke", "price", "peak-rpm"]].astype("float")
    df[["normalized-losses", "horsepower"]] = df[["normalized-losses", "horsepower"]].astype("int")

    # Convert data from american to european format
    df["city-L/100km"] = 235 / df["city-mpg"]
    df["highway-L/100km"] = 235 / df["highway-mpg"]
    df.drop(columns=["city-mpg", "highway-mpg"], axis=1, inplace=True)
    print("Data have been prepared")
    df.to_csv("dataset_auto_clean.csv")  # save prepared data
    return df
