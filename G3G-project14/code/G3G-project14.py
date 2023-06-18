import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.graphics.gofplots as smplts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from scipy.stats import shapiro
from sklearn.svm import SVR
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# Load data
data_maths = pd.read_csv('../data/Maths.csv')
data_portuguese = pd.read_csv('../data/Portuguese.csv')

# How many rows and columns?
# print(f"In the maths dataset there are {data_maths.shape[0]} rows and {data_maths.shape[1]} columns.\n")
# print(f"In the dataset there are {data_portuguese.shape[0]} rows and {data_portuguese.shape[1]} columns.\n")

# Same features for both data frame
# Add the feature 'subject' = maths or portuguese
data_maths['subject'] = 'maths'
data_portuguese['subject'] = 'portuguese'

# Merge data frame to get more datapoints
frames = [data_maths, data_portuguese]
data_whole = pd.concat(frames)


# Add the feature 'Talc' = total alcohol
# data_whole.insert(data_whole.columns.get_loc('Walc') + 1, 'Talc', data_whole['Walc'] + data_whole['Dalc'])
# data_whole['Talc'] = data_whole['Talc'].astype(int)

# Remove column with alcohol
# data_whole = data_whole.drop('Dalc', axis=1)
# data_whole = data_whole.drop('Walc', axis=1)



# Data Overview
# Datapoints and features
print(f"Number of rows: {data_whole.shape[0]} and columns: {data_whole.shape[1]}")

overview = data_whole.head()
print(f'\nFirst five rows of the data: {overview}')
overview.to_csv('../output/alcohol/data_overview.csv', index=False)

# Checking the data types       dtypes: int64(17), object(18)
print("\nInformation about the data:")
print(data_whole.info())

# Adjustment of the data types      dtypes: category(22), int64(5), object(8)
cat_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
               'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
               'failures', 'famrel', 'freetime', 'goout', 'health', 'Dalc', 'Walc', 'subject']

data_whole[cat_columns] = data_whole[cat_columns].astype('category')
print("\nInformation about the data:")
print(data_whole.info())


# Missing data -> no missing data
print("\nCount of missing values:")
print(data_whole.isnull().sum())


# Replace possible missing data with NaN
# Avoid to delete entire datapoint if at least one performance is available
data_whole['G1'] = data_whole['G1'].replace(0, np.nan)
data_whole['G2'] = data_whole['G2'].replace(0, np.nan)
data_whole['G3'] = data_whole['G3'].replace(0, np.nan)


# Target variable: mean of grade G1, G2, G3 = mean_grade
data_whole['mean_grade'] = data_whole[['G1', 'G2', 'G3']].mean(axis=1, skipna=True).round().astype(int)
data_whole.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
print(f'\nNumber of rows: {data_whole.shape[0]} and columns: {data_whole.shape[1]}')

# Check for duplicates
print(f"\nDuplicates: {sum(data_whole.duplicated(subset=None, keep='first'))}.")


# Statistics of numerical columns
descr = data_whole.describe()
print('\nSummary statistics of numerical columns:')
print(descr)
descr.to_csv(f'../output/alcohol/Descriptive Statistics before cleaning.csv')


# Dealing the outliers (drop and visualisation)
num_cols = len(data_whole.select_dtypes(include='int64').columns)
num_rows = 2
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
fig.tight_layout(pad=3.0)
outlier_counts = {}

for i, feature in enumerate(data_whole.select_dtypes(include='int64').columns):
    row = i // num_cols
    col = i % num_cols

    # Count the total number of data points
    total_data_points = len(data_whole[feature])
    print(f'\nTotal number of data points: {total_data_points}')

    # Create the boxplot with outliers
    data_whole.boxplot(column=feature, ax=axes[0][col])
    axes[0][col].set_title(feature + ' Boxplot (with outliers)')

    # Calculate the upper whisker value
    q3 = data_whole[feature].quantile(0.75)
    q1 = data_whole[feature].quantile(0.25)
    iqr = q3 - q1
    upper_whisker = q3 + 1.5 * iqr
    print(f'Upper whisker of {feature}: {upper_whisker}')

    # Count the number of outliers outside the upper whisker
    outliers = data_whole[data_whole[feature] > upper_whisker][feature]
    outlier_count = len(outliers)
    outlier_counts[feature] = outlier_count

    # Remove outliers above the upper whisker
    data_whole = data_whole[data_whole[feature] <= upper_whisker]

    # Create the boxplot without outliers
    data_whole.boxplot(column=feature, ax=axes[1][col])
    axes[1][col].set_title(feature + ' Boxplot (without outliers)')

    # Add text to the plot indicating the upper whisker value for both plots
    axes[0][col].text(0.95, 0.95, f"Upper whisker: {upper_whisker:.2f}", transform=axes[0][col].transAxes,
                      verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    axes[1][col].text(0.95, 0.95, f"Upper whisker: {upper_whisker:.2f}", transform=axes[1][col].transAxes,
                      verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Print the outlier and total datapoints counts
    print(f"Number of outliers in {feature}: {outlier_count}")

fig.suptitle("Boxplot with and without outliers")
fig.tight_layout()
fig.savefig('../output/alcohol/Boxplot outliers vs no outliers.pdf')
plt.show()


# Datapoints and features
print(f'\nNumber of rows: {data_whole.shape[0]} and columns: {data_whole.shape[1]}')

# Descriptive statistics after data-cleaning
descr_after = data_whole.describe()
descr_after.to_csv(f'../output/alcohol/Descriptive Statistics after cleaning.csv')


# Data Distributions: imbalance ratio
imbalance_ratios = {}
for column in data_whole.columns:
    value_counts = data_whole[column].value_counts()
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count
    imbalance_ratios[column] = imbalance_ratio

# Print the imbalance ratios
print('')
for column, ratio in imbalance_ratios.items():
    print(f'{column.capitalize()} imbalance ratio: {ratio}')


# Distribution histograms
num_cols = len(data_whole.select_dtypes(include='int64').columns)
num_rows = 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
for i, feature in enumerate(data_whole.select_dtypes(include=['int64']).columns):
    ax = axes[i]
    sns.histplot(data=data_whole, x=feature, ax=ax, binwidth=1, color='#969696')
    ax.set_title(feature + ' histogram')

fig.tight_layout()
fig.savefig("../output/alcohol/Distribution histograms.pdf")
plt.show()


# Test for normality
print('')
for column in data_whole.select_dtypes(include='int64').columns:
    stat, p = shapiro(data_whole[column])
    alpha = 0.05
    significance_level = alpha / ((data_whole.dtypes == 'int64') | (data_whole.dtypes == 'float64')).sum()

    if p > significance_level:
        print(f"{column} is normally distributed (p={p:.4f})")
    else:
        print(f"{column} is NOT normally distributed (p={p:.4f})")

    # Plot QQ plot
    smplts.qqplot(data_whole[column], line='s')
    plt.title(f"QQ plot of {column}")
    plt.show()

print("\n")


# Identify numerical variables for statistical performance and standardization
numerical_columns = data_whole.select_dtypes(include='int64').columns.tolist()
numerical_columns.remove('mean_grade')


# Linearity Test
# Perform statistical tests for numerical variable
for var in numerical_columns:
    # Compute Pearson correlation coefficient
    pearson_corr, p_value = stats.pearsonr(data_whole[var], data_whole['mean_grade'])

    # Perform F-test for linearity
    f_value, f_p_value = stats.f_oneway(data_whole[var], data_whole['mean_grade'])

    print(f"Variable: {var}")
    print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"F-value: {f_value:.4f}")
    print(f"F-test p-value: {f_p_value:.4f}")
    print()


# Apply one-hot Encoding
encoded_data = pd.get_dummies(data_whole, columns=cat_columns)


# Convert binary categorical variables to numeric values
binary_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
encoded_data[binary_columns] = encoded_data[binary_columns].replace({'yes': 1, 'no': 0})


print(f'\nNumber of rows and columns: {encoded_data.shape}')
print(f'\nFirst five rows of the encoded data: {encoded_data.head()}')


# Split the data into training and testing sets: 80% training and 20% testing
X = encoded_data.drop('mean_grade', axis=1)
y = encoded_data['mean_grade']

# Column to stratify
stratify_features = ['failures']
stratify_features = pd.get_dummies(data_whole[stratify_features])

# Split the data into training and testing sets: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=stratify_features)


print(f'\nNumber of feature: {X.shape[1]}')


# Perform standardization on numerical features
def standardize(X_train, X_test, numerical_columns):
    sc = StandardScaler()
    X_train[numerical_columns] = sc.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = sc.transform(X_test[numerical_columns])
    return X_train, X_test

X_train, X_test = standardize(X_train, X_test, numerical_columns)



# Feature Selection with Univariate feature selection (UVFS) and Cross-Validation
best_k = None
best_score = float('-inf')
for k in range(1, X.shape[1] + 1):
    UVFS_Selector = SelectKBest(score_func=f_regression, k=k)
    X_train_UVFS = UVFS_Selector.fit_transform(X_train, y_train)
    X_test_UVFS = UVFS_Selector.transform(X_test)

    # Perform 5-fold cross-validation
    model = LinearRegression()
    scores = cross_val_score(model, X_train_UVFS, y_train, cv=5)

    avg_score = scores.mean()

    if avg_score > best_score:
        best_score = avg_score
        best_k = k
        selected_features_UVFS = X.columns[UVFS_Selector.get_support()]

print('\nUnivariate feature selection (UVFS)')
print(f'Best k: {best_k}')
print(f"Selected Feature: {', '.join(selected_features_UVFS)}")

# Sort in decreasing order
sorted_indices_UVFS = UVFS_Selector.scores_.argsort()[::-1]
sorted_features_UVFS = X.columns[sorted_indices_UVFS]
sorted_scores_UVFS = UVFS_Selector.scores_[sorted_indices_UVFS]



# Feature Selection with L1 Regularization (Lasso) and Cross-Validation
lasso = Lasso(random_state=10)

# Define the range of alpha values to be tested
param_grid = {'alpha': np.arange(start=0, stop=1, step=0.01)}

# Perform grid search with cross-validation
grid_search = GridSearchCV(lasso, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best alpha value and the corresponding model
best_alpha = grid_search.best_params_['alpha']
best_model = grid_search.best_estimator_

# Use the best model for feature selection
sel_ = SelectFromModel(best_model)
sel_.fit(X_train, y_train)

# Get the selected features
X_train_L1 = sel_.transform(X_train)
X_test_L1 = sel_.transform(X_test)
selected_features_L1 = X.columns[sel_.get_support()]

print('\nL1 feature selection')
print(f'Best alpha: {best_alpha}')
print(f"Feature selected by L1: {', '.join(selected_features_L1)}")

# Sort in decreasing order
sorted_indices_L1 = np.argsort(-np.abs(sel_.estimator_.coef_))
sorted_features_L1 = X.columns[sorted_indices_L1]
sorted_scores_L1 = np.abs(sel_.estimator_.coef_[sorted_indices_L1])



# Plot Feature selection in decreasing order of importance

# UVFS plot
plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_scores_UVFS)), sorted_scores_UVFS, width=0.9, color='#bdbdbd')
plt.xticks(range(len(sorted_features_UVFS)), sorted_features_UVFS, rotation=90, fontsize=6)
plt.xlabel('Feature')
plt.ylabel('Feature Importance (F-value)')
plt.yticks(fontsize=8)
plt.title('Selected Features Based on Univariate Analysis')
plt.tight_layout()
plt.xlim(-0.5, len(sorted_scores_UVFS) - 0.5)
plt.savefig(f"../output/alcohol/Selected Features UVFS.pdf")
plt.show()

# L1 plot
plt.figure(figsize=(12, 8))
plt.bar(range(len(sorted_scores_L1)), sorted_scores_L1, width=0.9, color='#bdbdbd')
plt.xticks(range(len(sorted_features_L1)), sorted_features_L1, rotation=90, fontsize=6)
plt.xlabel('Feature')
plt.ylabel('L1 Regularization Coefficient Magnitude')
plt.yticks(fontsize=8)
plt.title('Selected Features Based on L1 Analysis')
plt.tight_layout()
plt.xlim(-0.5, len(sorted_scores_L1) - 0.5)
plt.savefig(f"../output/alcohol/Selected Features L1.pdf")
plt.show()


# Create a parameter grid for the max_depth and criterion hyperparameters
# and save these in a dictionary
decision_tree_params = {
    'criterion': ['absolute_error', 'squared_error', 'poisson', 'friedman_mse'],
    'splitter': ['best', 'random'],
    'max_depth': list(range(0, 10)) + [None]
}
random_forest_params = {
    'n_estimators': list(range(10, 20)),
    'criterion': ['absolute_error', 'squared_error', 'poisson', 'friedman_mse'],
    'max_depth': list(range(1, 5))
}
gradient_boosting_params = {
    'n_estimators': list(range(50, 100, 10)),
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
    'max_depth': list(range(1, 5)) + [None]
}
svr_params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'epsilon': [i/10 for i in range(1, 5)]
}



# Decision Tree Regressor -> tend to overfit on training set
def decision_tree(X_selected_features, y_train, feature_selection, params):
    DT = DecisionTreeRegressor(random_state=42)
    DT_GS = GridSearchCV(DT, params, cv=5)
    DT_GS.fit(X_selected_features, y_train)
    best_params_DT = DT_GS.best_params_
    best_model_DT = DT_GS.best_estimator_

    print(f'\nDecision Tree Regressor ({feature_selection})')
    print('Best Criterion:', best_model_DT.get_params()['criterion'])
    print('Best Splitter:', best_model_DT.get_params()['splitter'])
    print('Best max_depth:', best_model_DT.get_params()['max_depth'])
    return best_params_DT


# Random Forest Regressor
def random_forest(X_selected_features, y_train, feature_selection, params):
    RF = RandomForestRegressor(n_estimators=100, random_state=42)
    RF_GS = GridSearchCV(RF, params, cv=5)
    RF_GS.fit(X_selected_features, y_train)
    best_params_RF = RF_GS.best_params_
    best_model_RF = RF_GS.best_estimator_

    print(f'\nRandom Forest Regressor ({feature_selection})')
    print('n_estimators:', best_model_RF.get_params()['n_estimators'])
    print('Best Criterion:', best_model_RF.get_params()['criterion'])
    print('Best max_depth:', best_model_RF.get_params()['max_depth'])
    return best_params_RF


# Gradient Boosting Regressor
def gradient_boosting(X_selected_features, y_train, feature_selection, params):
    gb = GradientBoostingRegressor(random_state=42)
    gb_GS = GridSearchCV(gb, params, cv=5)
    gb_GS.fit(X_selected_features, y_train)
    best_params_gb = gb_GS.best_params_
    best_model_gb = gb_GS.best_estimator_

    print(f'\nGradient Boosting Regressor ({feature_selection})')
    print('Best n_estimators:', best_model_gb.get_params()['n_estimators'])
    print('Best learning_rate:', best_model_gb.get_params()['learning_rate'])
    print('Best max_depth:', best_model_gb.get_params()['max_depth'])
    return best_params_gb


# Support Vector Machines Regression
def support_vector_regression(X_selected_features, y_train, feature_selection, params):
    np.random.seed(42)
    svr = SVR()
    svr_GS = GridSearchCV(svr, params, cv=5)
    svr_GS.fit(X_selected_features, y_train)
    best_params_svr = svr_GS.best_params_
    best_model_svr = svr_GS.best_estimator_
    print(f'\nSupport Vector Machines ({feature_selection})')
    print('kernel:', best_model_svr.get_params()['kernel'])
    print('epsilon:', best_model_svr.get_params()['epsilon'])
    return best_params_svr


# Run the 4 models with both feature selections
best_params_DT_L1 = decision_tree(X_train_L1, y_train, "L1", decision_tree_params)
best_params_DT_UVFS = decision_tree(X_train_UVFS, y_train, "UVFS", decision_tree_params)

best_params_RF_L1 = random_forest(X_train_L1, y_train, "L1", random_forest_params)
best_params_RF_UVFS = random_forest(X_train_UVFS, y_train, "UVFS", random_forest_params)

best_params_gb_L1 = gradient_boosting(X_train_L1, y_train, "L1", gradient_boosting_params)
best_params_gb_UVFS = gradient_boosting(X_train_UVFS, y_train, "UVFS", gradient_boosting_params)

best_params_svr_L1 = support_vector_regression(X_train_L1, y_train, "L1", svr_params)
best_params_svr_UVFS = support_vector_regression(X_train_UVFS, y_train, "UVFS", svr_params)



def get_scores(model, X_train, y_train, X_test, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # evaluation
    r2_test = r2_score(y_test, y_pred_test).round(3)
    mse_test = mean_squared_error(y_test, y_pred_test).round(3)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False).round(3)
    mae_test = mean_absolute_error(y_test, y_pred_test).round(3)

    r2_train = r2_score(y_train, y_pred_train).round(3)
    mse_train = mean_squared_error(y_train, y_pred_train).round(3)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False).round(3)
    mae_train = mean_absolute_error(y_train, y_pred_train).round(3)

    return [r2_test, mse_test, rmse_test, mae_test, r2_train, mse_train, rmse_train, mae_train]


# Cross-validation with train and evaluation split to get the performance of the models
def cross_val_performance(X_train, y_train, feature_selection, best_params_DT,
                          best_params_RF, best_params_gb, best_params_svr):
    # Perform a 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    for train_index, validation_index in kf.split(X_train):

        X_train_fold, X_validation_fold = X_train[train_index], X_train[validation_index]
        y_train_fold, y_validation_fold = y_train.iloc[train_index], y_train.iloc[validation_index]

        # Decision Tree Regressor
        DT = DecisionTreeRegressor(**best_params_DT)
        DT.fit(X_train_fold, y_train_fold)

        # Random Forest Regressor
        RF = RandomForestRegressor(**best_params_RF)
        RF.fit(X_train_fold, y_train_fold)

        # Gradient Boosting Regressor
        gb = GradientBoostingRegressor(**best_params_gb)
        gb.fit(X_train_fold, y_train_fold)

        # Support Vector Machines
        svr = SVR(**best_params_svr)
        svr.fit(X_train_fold, y_train_fold)

        # Calculate metrics
        models = [('DT', feature_selection), ('RF', feature_selection),
                  ('gb', feature_selection), ('svr', feature_selection)]
        for model_type, feature_type in models:
            validation_scores = get_scores(eval(f"{model_type}"), eval(f"X_train_fold"), y_train_fold,
                                           eval(f"X_validation_fold"), y_validation_fold)[:4]

            train_scores = get_scores(eval(f"{model_type}"), eval(f"X_train_fold"), y_train_fold,
                                      eval(f"X_validation_fold"), y_validation_fold)[-4:]

            df_performance.loc[len(df_performance), :] = [fold, model_type, feature_selection, 'Train'] + train_scores
            df_performance.loc[len(df_performance), :] = [fold, model_type, feature_selection, 'Validation'] + validation_scores

        # increase counter for folds
        fold += 1

    return df_performance


# Data frame to save the evaluation metrics
df_performance = pd.DataFrame(columns=['Fold', 'Model', 'Feature selection',
                                       'Train/Validation', 'R2', 'MSE', 'RMSE', 'MAE'])


# UVFS cross-validation
df_performance_UVFS = cross_val_performance(X_train_UVFS, y_train, 'UVFS', best_params_DT_UVFS,
                                            best_params_RF_UVFS, best_params_gb_UVFS, best_params_svr_UVFS)

# L1 cross-validation
df_performance_L1 = cross_val_performance(X_train_L1, y_train, 'L1', best_params_DT_L1,
                                          best_params_RF_L1, best_params_gb_L1, best_params_svr_L1)


# Get table with all performance
frames = [df_performance_UVFS, df_performance_L1]
CV_performance = pd.concat(frames)
CV_performance.drop("Fold", axis=1, inplace=True)


# Calculate the mean performance over the 5 folds
mean_performance = CV_performance.groupby(['Model', 'Feature selection', 'Train/Validation']).mean()

# Calculate the std on the 5-folds
std = CV_performance.groupby(['Model', 'Feature selection', 'Train/Validation']).std()

# Table with the mean on 5-folds ± std
mean_std_performance = mean_performance.round(3).astype(str) + ' ± ' + std.round(3).astype(str)
mean_std_performance.columns = [col + ' mean±std' for col in mean_std_performance.columns]
mean_std_performance.to_csv(f'../output/alcohol/CV_Mean_Performance.csv')
print(mean_std_performance)


# Search the model with the best performance
# Select only the validation split
mean_performance = mean_performance.reset_index()
mean_performance_validation = mean_performance[mean_performance['Train/Validation'] == 'Validation']

# Validation performance in decreasing order based on R2 score
mean_performance_validation = mean_performance_validation.sort_values(by='R2', ascending=False)
print(mean_performance_validation)

# Get the best model, feature selection, and test score
best_model_validation = mean_performance_validation.iloc[0]['Model']
best_feature_selection = mean_performance_validation.iloc[0]['Feature selection']
best_validation_score = mean_performance_validation.iloc[0]['R2']

# Print the best model, feature selection, and test score
print("\nBest Validation Performance:")
print(f"Model: {best_model_validation}")
print(f"Feature Selection: {best_feature_selection}")
print(f"Validation Score (R2): {best_validation_score}")


# Create the best model based on the best model name and best parameters
best_model = None
if best_model_validation == 'DT':
    best_model = DecisionTreeRegressor(**best_params_DT_UVFS if best_feature_selection == 'UVFS' else best_params_DT_L1)
elif best_model_validation == 'RF':
    best_model = RandomForestRegressor(**best_params_RF_UVFS if best_feature_selection == 'UVFS' else best_params_RF_L1)
elif best_model_validation == 'gb':
    best_model = GradientBoostingRegressor(**best_params_gb_UVFS if best_feature_selection == 'UVFS' else best_params_gb_L1)
elif best_model_validation == 'svr':
    best_model = SVR(**best_params_svr_UVFS if best_feature_selection == 'UVFS' else best_params_svr_L1)


# Calculate the test performance metrics on the best model

# Data frame to save the evaluation metrics
df_top_performance = pd.DataFrame(columns=['Train/Validation', 'R2', 'MSE', 'RMSE', 'MAE'])

# Select X_train and X_test of the best feature selection
X_train_best = eval(f"X_train_{best_feature_selection}")
X_test_best = eval(f"X_test_{best_feature_selection}")

# Train the best model on the training set
best_model.fit(X_train_best, y_train)

# Calculate the performance metrics on the test set
best_test_scores = get_scores(best_model, X_train_best, y_train, X_test_best, y_test)[:4]
best_train_scores = get_scores(best_model, X_train_best, y_train, X_test_best, y_test)[-4:]
df_top_performance.loc[len(df_top_performance), :] = ['Train'] + best_train_scores
df_top_performance.loc[len(df_top_performance), :] = ['Test'] + best_test_scores

df_top_performance.to_csv(f'../output/alcohol/Best_performance_{best_model_validation}_{best_feature_selection}.csv', index=False)

print("\nBest Test Performance:")
print(f"Model: {best_model_validation}")
print(f"Feature Selection: {best_feature_selection}")
print(f"Test Score (R2): {df_top_performance.iloc[1]['R2']}")


# Correlation between alcohol and School Performance
def alcohol_boxplot(data, a, b, day):
    color = ['#bdbdbd', '#737373']
    sns.boxplot(data=data, x=a, y=b, hue="sex", palette=color)
    sns.color_palette("YlOrBr", as_cmap=True)
    plt.title(f'{day} alcohol consumption vs. mean grade')
    plt.savefig(f'../output/Mean Grade vs alcohol boxplot_{b}.pdf')
    plt.close()


alcohol_boxplot(data_whole, 'mean_grade', 'Walc', 'Weekend')
alcohol_boxplot(data_whole, 'mean_grade', 'Dalc', 'Work day')
