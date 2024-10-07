
import os
import argparse
import pandas as pd
import logging
import mlflow


def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--clean_data", type=str, help="path to save cleaned data")
    args = parser.parse_args()

    mlflow.start_run()

    logging.info("Input data: %s", args.data)
    print("Input data:", args.data)

    df = pd.read_csv(args.data)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", df.shape[1])

    missing_values = df.isnull().sum()
    missing_values_dict = missing_values[missing_values > 0].to_dict()
    for feature, count in missing_values_dict.items():
        mlflow.log_metric(f"missing_values_{feature}", count)


    duplicate_values = df.duplicated().sum()
    mlflow.log_metric("duplicate_values", duplicate_values)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    output_path = os.path.join(args.clean_data, "cleaned_data.csv")
    df.to_csv(output_path, index=False)

    logging.info("Cleaned data saved to: %s", output_path)


    mlflow.end_run()


if __name__ == "__main__":
    main()
