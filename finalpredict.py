"""
Iiro Tauriainen

Prediction script for Google Trends keywords

What IT DOES:
- loads a trained Logistic Regression model and LabelEncoder
- reads trending keywords from a CSV file
- fetches interest_over_time data for each keyword from Google Trends (past 7 days hourly)
- calculates features for each keyword based on the interest_over_time data
- predict trend duration group (long/not_long) for each keyword with probabilities

HOW TO USE:
1. Make sure you have the required libraries installed:
   - pandas
   - numpy
   - joblib
   - pytrends
2. Go to Google Trends: https://trends.google.com/trending?geo=US&hours=24
   - choose region: United States
   - choose timeframe: past 24h
   - choose all categories
   - click "Export" -> download as CSV
3. Save the CSV file in the same folder as this script and make sure the filename matches the TRENDING_CSV variable in the script.
4. Make sure you have the trained Logistic Regression model and LabelEncoder saved as .joblib files in this folder.
5. Run the script: `python finalpredict.py`

NOTE:
- Make sure you first run the lr.py script to train the model and save it as a .joblib file.
    - lr_pipeline.joblib
    - lr_label_encoder.joblib

"""

import pandas as pd # data handling
import numpy as np # numerical operations
import joblib # model saving / loading
import sys # system operations, errors / stop
from pytrends.request import TrendReq # Google Trends API

TRENDING_CSV = "trending_US_1d_1.csv" # csv dataset with trending keywords
TIMEFRAME = "now 7-d" # timeframe for Google Trends

PIPELINE_PATH = "lr_pipeline.joblib"
ENCODER_PATH = "lr_label_encoder.joblib" 

FEATURE_COLUMNS = ['past48h_max', 'past48h_mean', 'past48h_std', 'slope_last_6h', 'trend_peak_hour'] #calculated using interest_over_time data

# This function calculates features for one word ^
def calculate_features(series):

    series = series[-48:] # Last 48h of data because model was trained on that

    # if data is shorter than 48h, pad with zeros
    if len(series) < 48:
        series = np.pad(series, (48 - len(series)), constant_values=(0,))

     
    arr = series.values # convert to numpy array
    past48h_max = arr.max() # max value last 48h
    past48h_mean = arr.mean() # mean value last 48h
    past48h_std = arr.std() # how much the values vary in last 48h e.g. 0.0 = no variation, 1.0 = some variation
    slope = np.polyfit(range(6), arr[-6:], 1)[0] # last 6h slope (trend direction)
    peak_hour = -1 * (48 - np.argmax(arr)) # hour of the peak value in last 48h (negative because we count from the end)

    return [past48h_max, past48h_mean, past48h_std, slope, peak_hour]


# Download model and LabelEncoder
print("\nDownload model and LabelEncoder...")
try:
    pipeline = joblib.load(PIPELINE_PATH) # Load the trained model pipeline

    label_encoder = joblib.load(ENCODER_PATH) # Load the LabelEncoder used to encode the target variable (long/not_long)
    print("downloaded successfully!")

except Exception as e:

    print(f"ERROR {e}")
    sys.exit(1)


# Reads keywords from csv file
print(f"\nReading keywords from {TRENDING_CSV}")

try:
    trending_df = pd.read_csv(TRENDING_CSV)

    print(trending_df.head())

    if 'Title' in trending_df.columns:
        all_keywords = trending_df['Title'].dropna().unique().tolist() # 'Title' is the column with trending keywords

    elif trending_df.columns[0]:
        all_keywords = trending_df.iloc[:, 0].dropna().unique().tolist() # plan B if 'Title' column is not present, use the first column

    else:
        raise ValueError("CSV doesnt contain expected columns")
except Exception as e:
    print(f"error while trying to read csv file {e}")
    sys.exit(1)

all_keywords = list(set(all_keywords)) # Remove duplicates and convert to list
print(f"\n{len(all_keywords)} words read from CSV.")

if not all_keywords:
    print("Csv file is empty or does not contain any keywords.")
    sys.exit(1)


#Getting interest_over_time data for each keyword
print("\nGetting interest_over_time data for each keyword from Google Trends ...") # Last 7 days
pytrends = TrendReq(hl='en-US', tz=360) # Make pytrends session with US timezone

results = {} # collect interest data for each keyword
batch_size = 5 # Number of keywords to fetch at once

# go through all keyword in 5 keyword batches
for i in range(0, len(all_keywords), batch_size):
    batch = all_keywords[i:i + batch_size]

    try:
        pytrends.build_payload(batch, timeframe=TIMEFRAME) # set the query and timeframe for Google Trends
        data = pytrends.interest_over_time() # get interest over time data

        if data.empty: # No data returned for this batch
            continue
        # go through each word in the batch and save the data in results
        for word in batch:
            if word in data:
                results[word] = data[word]
    # if error, skip the batch and continue with the next
    except Exception as e:
        print(f"SKIP THIS {batch} (ERROR: {e})")

print(f"\n---Interest_over_time GOT for {len(results)} words.")

# if no data in results then exit
if not results:
    print("DID NOT GET ANY interest_over_time data for the keywords.")
    sys.exit(1)


# prrint interest_over_time for the first word (last 48h)
first_word = list(results.keys())[0]
print(f"\nFirst word: '{first_word}' interest_over_time for last 48h: ")
print(results[first_word][-48:])



# Count features for each word
print("\nCounting features...")
feature_rows = []
words_ok = [] # List to collect words that have valid interest_over_time data
skipped_zero_trends = 0 # counter how many words we skipped because they had no interest_over_time data in last 48h

# loop through each word (time last 48h) 
for word, series in results.items():
    last_48h = series[-48:] 
    if last_48h.sum() == 0: # if sum of last 48h is zero, skip this word
        skipped_zero_trends += 1
        continue
    if len(series) < 10: # if series is too short, skip this word
        continue

    feats = calculate_features(series) # Calculate features for the last 48h of interest_over_time data
    feature_rows.append(feats) # Append the features to the list
    words_ok.append(word) # Collect the word for which features were calculated


if not feature_rows:
    print("No Features calculated. All words had zero interest_over_time in last 48h.")
    sys.exit(1)

# Create a DataFrame with the calculated features
X_new = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)

print(f"\nFeature lines created for {len(words_ok)} words.")
print(f"{skipped_zero_trends} words skipped beacause interest over time value was zero in last 48h.")


# predicting trend duration group with new data
print("\nPredicting trend duration...")


preds = pipeline.predict(X_new) # predicting target with the feature data 
all_probas = pipeline.predict_proba(X_new) # probabilities for each class (long/not_long)

# Get the indices of the classes in the label encoder
# what is the index of 'long' and 'not_long' in the label encoder
class_order = list(label_encoder.classes_)
long_index = class_order.index('long')
not_long_index = class_order.index('not_long')


#-----------------------------
# printing results
#-----------------------------

long_results = []
not_long_results = []

# Changing the output format to be more readable
# Loop through each word, its prediction and probabilities
for word, pred, prob_pair in zip(words_ok, preds, all_probas):
    pred_label = label_encoder.inverse_transform([pred])[0]
    prob_long = prob_pair[long_index]
    prob_not_long = prob_pair[not_long_index]
    line = f"- {word}: {pred_label} (P_long={prob_long:.2f}, P_not_long={prob_not_long:.2f})"
    if pred_label == 'long':
        long_results.append(line)
    else:
        not_long_results.append(line)


print("\nPREDICTIONS\n")

print("\nPossible LONG trends: ")
for line in long_results:
    print(line)

print("\nNot long trends: ")
for line in not_long_results:
    print(line)

print("\nREADY")
