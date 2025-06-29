
"""
iiro Tauriainen

lr.py logistic regression model for predicting trend_duration_group (long/not_long) class.

WHAT THIS SCRIP DOES:
- loads data from CSV
- selects features that are not leaking information about the target variable
- encodes the target variable (trend_duration_group) to numeric values
- splits the data into training and test sets
- applies SMOTE to balance the training set (minority in this case is 'long' trend)
- creates a pipeline with StandardScaler and Logistic Regression
- fits the model to the training set
- evaluates the mode on the test set
    - classification report (precision, recall, F1-score
    - confusion matrix heatmap
    - ROC curve and AUC score
- saves the model and label encoder as .joblib files for later use

ABOUT DATA:
- Dataset was collected in two stages from google trends:
    - First, trending keywords were collected from Google Trends (Trending now 24h)
    - then interest_over_time was collecter for each keyword for past 48h to compute features (max, mean, std, slope, peak hour)
    
    - Second, 24h later the same keywords were collected again to see how their trend continued
    - based on the future behavior, target label "long or not long" was assigned to indicate trend duration.

HOW TO USE THIS SCRIPT:
1. Make sure you have the required libraries installed:
    - pandas
    - numpy
    - joblib
    - imbalanced-learn
    - scikit-learn
    - matplotlib
    - seaborn
2. Save the dataset as "data_2label.csv" in the same folder as this script.
    - Should contain target column "trend_duration_group" with values "long" and "not_long" and features from interest_over_time data.
3. Run the script: `python lr.py`

OUTPUT:
- Classification report with precision, recall, F1-score for each class
- Confusion matrix heatmap
- ROC curve and AUC score
- Saved model and label encoder as .joblib files for later use.

"""

import pandas as pd # data handling
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization / heatmaps
import joblib # model saving / loading

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# download data
print("\n--- Downloading data ---\n")
data = pd.read_csv("data_2label.csv") # Read the dataset to pandas dataframe
print(data.head())  # Display the first few rows of the dataset

# dropping features that are not useful
features_to_drop = [
    'keyword', # not feature, just text
    'trend_duration_category', # correlates with target
    'trend_duration_group' # target variable, not feature
]

X = data.drop(columns=features_to_drop) # features
y = data['trend_duration_group'] # target variable

print("\n--- Features used for training ---\n")
print(list(X.columns))

# because target is string value, encode it to numeric
# 'not_long' -> 0, 'long' -> 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# printing class distribution so we can see if we have class imbalance
print("\n--- Class distribution before train/test split ---\n")
print(pd.Series(y_encoded).value_counts())

# 20% to test set, 80% to train set
# stratify ensures that the class distribution is similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\n--- Class distribution after train/test split ---\n")
print(pd.Series(y_train).value_counts(), "(train)")
print(pd.Series(y_test).value_counts(), "(test)")


print("\n--- Using Smote to balance the training set (long is minority class) ---\n")
smote = SMOTE(random_state=42) # smote instance with same random state
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train) # fit and resample the training set

print("\n--- Class distribution after SMOTE ---\n")
print(pd.Series(y_train_sm).value_counts())


# Pipeline: StandardScaler+Logistic Regression
print("\n--- Creating Logistic Regression pipeline with StandardScaler ---\n")
logreg_pipeline = Pipeline([ # scale features and then fit logistic regression
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=500))
])

logreg_pipeline.fit(X_train_sm, y_train_sm) # fit the pipeline to the training set


y_pred = logreg_pipeline.predict(X_test) # predict the test set, which one is true 1 or 0
y_proba = logreg_pipeline.predict_proba(X_test)[:, 1] # predict probabilities, how likely it is to be 1 = long trend



# print results:
# Precision = täsmällisyys (Kun malli ennusti 1, kuinka usein se oli oikeassa)
# Recall = Herkkyys (kuinka hyvin malli löysi kaikki 1?)
# F1-score = F1 score combines precision and recall into a single value
print("\n--- Classification Report for logistic regression ---\n")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))


# Confusion matrix
# row = actual
# column = predicted
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Purples',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()



# ROC Curve and AUC
# roc shows how well the model can distinguish between classes
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# how good model is at distinguishing between classes
print(f"\n--- AUC result: {auc_score:.3f}") # AUC = Area Under the Curve,


# Save the model and label encoder
# logreg is ready a pipeline, so we can save it directly
# encoder is needed to decode the predictions back to original labels
joblib.dump(logreg_pipeline, "lr_pipeline.joblib")
joblib.dump(label_encoder, "lr_label_encoder.joblib")

print("\n--- model and label encoder saved ---")
print(" - lr_pipeline.joblib")
print(" - lr_label_encoder.joblib")
