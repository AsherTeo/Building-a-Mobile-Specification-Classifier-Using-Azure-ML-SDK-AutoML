
import os
import argparse
import pandas as pd
import logging
import mlflow
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.feature_selection import SelectKBest, chi2, f_classif


def select_first_file(path):
    files = os.listdir(path)
    if not files:
        logging.error("No files found in the specified directory.")
        raise FileNotFoundError("No files found in the specified directory.")
    return os.path.join(path, files[0])


def main():
    """Main function of the script."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--display_dir", type=str, help="directory to save plots and results")
    parser.add_argument("--output_dir", type=str, help="directory to save plots and results")
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()

    input_file = select_first_file(args.data)
    logging.info(f"Loading data from {input_file}")
    
    df = pd.read_csv(input_file)

    numerical = ['battery_power', 'clock_speed', 'fc',  'int_memory',  'mobile_wt',  
                 'pc', 'px_height','px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

    cat = ['blue','dual_sim','four_g','m_dep','n_cores', 'three_g','touch_screen', 'wifi']

    # Before Skewness
    skewness_val = {col: skew(df[col]) for col in numerical}
    skewness_before_df = pd.DataFrame(list(skewness_val.items()), columns=['Feature', 'Skewness']).sort_values(by='Skewness', ascending=False)

    skewness_before_csv = os.path.join(args.display_dir, "skew_before_df.csv")
    skewness_before_df.to_csv(skewness_before_csv, index=False)
    logging.info(f"Skewness Before saved to {skewness_before_csv}")

    mlflow.log_artifact(skewness_before_csv)

    # After Skewness
    for col in ['px_height', 'fc', 'sc_w', 'clock_speed']:
        df[col] += 1  
        transformed_data, _ = boxcox(df[col])
        df[col] = transformed_data

    skewness_val = {col: skew(df[col]) for col in numerical}
    skewness_after_df = pd.DataFrame(list(skewness_val.items()), columns=['Feature', 'Skewness']).sort_values(by='Skewness', ascending=False)

    skewness_after_csv = os.path.join(args.display_dir, "skew_after_df.csv")
    skewness_after_df.to_csv(skewness_after_csv, index=False)
    logging.info(f"Skewness After saved to {skewness_after_csv}")

    mlflow.log_artifact(skewness_after_csv)

    # ANOVA
    selector = SelectKBest(f_classif, k=12)
    X_anova = selector.fit_transform(df[numerical], df['price_range'])

    anova_scores = selector.scores_
    anova_p_values = selector.pvalues_

    anova_df = pd.DataFrame({'Feature': numerical, 'P-Value': anova_p_values, 'Scores': anova_scores})
    anova_df.sort_values(by='P-Value', ascending=True, inplace=True)

    plt.figure(figsize=(12, 8))
    plt.bar(anova_df['Feature'], anova_df['P-Value'], color='tab:red', alpha=0.6)
    plt.xticks(rotation=45)
    plt.xlabel('Features')
    plt.ylabel('P-Value')
    plt.title('P-Values of Features from ANOVA')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='Significance Level (0.05)')
    plt.legend()

    anova_chart = os.path.join(args.display_dir, "anova_p_values.png")
    plt.savefig(anova_chart)
    plt.close()

    mlflow.log_artifact(anova_chart)

    # Save ANOVA DataFrame as a CSV
    anova_csv = os.path.join(args.display_dir, "anova_df.csv")
    anova_df.to_csv(anova_csv, index=False)
    logging.info(f"ANOVA DataFrame saved to {anova_csv}")

    mlflow.log_artifact(anova_csv)

    # Chi Square
    chi_scores = chi2(df[cat].astype(int), df['price_range'])
    cat_df = pd.DataFrame({'Feature': cat, 'P-Value': chi_scores[1],})
    cat_df.sort_values(by='P-Value', ascending=True, inplace=True)

    plt.figure(figsize=(15, 8))
    plt.bar(cat_df['Feature'], cat_df['P-Value'], color='tab:blue', alpha=0.6)
    plt.xlabel('Features')
    plt.ylabel('P-Value')
    plt.title('P-Values of Features from Chi-Squared Test')
    plt.axhline(y=0.05, color='gray', linestyle='--', label='Significance Level (0.05)')
    plt.xticks(rotation=45)

    chi_square_chart = os.path.join(args.display_dir, "chi_square_p_values.png")
    plt.savefig(chi_square_chart)
    plt.close()

    mlflow.log_artifact(chi_square_chart)

    # Save Chi-Square DataFrame as a CSV
    chi_square_csv = os.path.join(args.display_dir, "chi_square_df.csv")
    cat_df.to_csv(chi_square_csv, index=False)
    logging.info(f"Chi-Square DataFrame saved to {chi_square_csv}")

    mlflow.log_artifact(chi_square_csv)

    # Feature Selection
    num_features = list(anova_df[anova_df['P-Value'] <= 0.05]['Feature'])
    cat_features = list(cat_df[cat_df['P-Value'] <= 0.05]['Feature'])

    #df[cat_features] = df[cat_features].astype(str)

    output_data = pd.concat([df[num_features].add_suffix('_num'), 
                          df[cat_features].add_suffix('_cat'), 
                          df['price_range']], 
                         axis=1)

    output_data_csv = os.path.join(args.output_dir, "output_data.csv")
    output_data.to_csv(output_data_csv, index=False)
    
    logging.info(f"Output data saved to {output_data_csv}")
    mlflow.log_artifact(output_data_csv)

 

    mlflow.end_run()   


if __name__ == "__main__":
    main()
