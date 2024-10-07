import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
import os
from joblib import dump
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def add_new_data(file_paths, new_file):
    """
    Adds a new Excel file to the existing file paths and returns a combined dataframe.
    
    Args:
        file_paths (tuple): Existing file paths.
        new_file (str): Path to the new Excel file to add.
        
    Returns:
        pd.DataFrame: Combined dataframe of all Excel files.
    """
    file_paths = file_paths + (new_file,)
    dfs = [pd.read_excel(file) for file in file_paths]
    combined_df = pd.concat(dfs)
    return combined_df



# # Define the existing file paths
# file_paths = (
#     "raw data/Market Prices.xls", "raw data/Market Prices 2.xls", 
#     "raw data/Market Prices 3.xls", "raw data/Market Prices 4.xls", 
#     "raw data/Market Prices 5.xls", "Raw Data/Market Prices 6.xls", 
#     "raw data/Market Prices 7.xls", "Raw Data/Market Prices 8.xls"
# )

# # Add a new Excel file
# new_file = "raw data/Market Prices 9.xls"
# df = add_new_data(file_paths, new_file)

# # Preprocess the data and save it to clean_data2.csv
# clean_df = preprocess_data(df)



def preprocess_data(df, clean_csv="clean_data2.csv"):
    """
    Preprocesses the data by dropping irrelevant columns, handling NaN values, 
    removing outliers, and saving the cleaned data to a CSV file.
    
    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        clean_csv (str): Path to save the clean CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Drop irrelevant columns
    df.drop(['Commodity', 'Grade', 'Sex'], axis=1, inplace=True)

    # Replace specific missing values with NaN
    df.replace(['-', ' - ', '- ', ' -'], np.nan, inplace=True)

    # Clean price columns
    price_columns = ["Wholesale", "Retail"]
    for col in price_columns:
        df[col] = df[col].str.lower().str.replace("/kg", "").str.strip()
        df[col] = df[col].str.lower().str.replace("s", "").str.strip().astype(float)

    # Impute missing values using KNN
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_columns = ["Supply Volume", "Retail", "Wholesale"]
    df[knn_columns] = knn_imputer.fit_transform(df[knn_columns])

    # Drop any remaining NaN values
    df.dropna(inplace=True)

    # Sort values by certain columns
    df.sort_values(by=['County', 'Market', 'Classification', 'Date'], inplace=True)

    # Remove markets with less than 10 records
    threshold = 10
    market_counts = df["Market"].value_counts()
    markets_to_keep = market_counts[market_counts >= threshold].index
    df = df[df['Market'].isin(markets_to_keep)]

    # Remove outliers
    num_columns = ["Retail", "Wholesale", "Supply Volume"]
    outliers = np.zeros(df.shape[0], dtype=bool)
    for col in num_columns:
        z_scores = stats.zscore(df[col])
        outliers = outliers | (np.abs(z_scores) > 3)
    df = df[~outliers]

    # Remove duplicates
    df = df.drop_duplicates()

    # Save the cleaned data to CSV
    df.to_csv(clean_csv, index=False)
    
    return df

#    feature engineering functions 

def feature_engineering(input_csv="clean_data2.csv", output_csv="modeling_data_2.csv"):
    # Load the cleaned data
    data = pd.read_csv(input_csv)

    # Convert 'Date' to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract time features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Quarter'] = data['Date'].dt.quarter

    # Cyclic features for Month, Day, DayOfWeek, Quarter, and Year
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
    data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    data['Quarter_sin'] = np.sin(2 * np.pi * data['Quarter'] / 4)
    data['Quarter_cos'] = np.cos(2 * np.pi * data['Quarter'] / 4)

    # Normalize Year using its min and max to create sin and cos
    year_min = data['Year'].min()
    year_max = data['Year'].max()
    data['Year_sin'] = np.sin(2 * np.pi * (data['Year'] - year_min) / (year_max - year_min))
    data['Year_cos'] = np.cos(2 * np.pi * (data['Year'] - year_min) / (year_max - year_min))

    # Add lag features for Wholesale, Retail, and Supply Volume
    for lag in [7]:
        data[f'Wholesale_lag_{lag}'] = data.groupby(['County', 'Market', 'Classification'])['Wholesale'].shift(lag)
        data[f'Retail_lag_{lag}'] = data.groupby(['County', 'Market', 'Classification'])['Retail'].shift(lag)
        data[f'Supply_Volume_lag_{lag}'] = data.groupby(['County', 'Market', 'Classification'])['Supply Volume'].shift(lag)

    # Sort data by Market, Classification, and Date
    data.sort_values(by=['Market', 'Classification', 'Date'], inplace=True)

    # Rolling mean and std features for 7-day windows
    rolling_windows = {'7d': 7}
    for window_name, window_size in rolling_windows.items():
        data[f'Wholesale_rolling_mean_{window_name}'] = data.groupby(['Market', 'Classification'])['Wholesale'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        data[f'Retail_rolling_mean_{window_name}'] = data.groupby(['Market', 'Classification'])['Retail'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        data[f'Supply_Volume_rolling_mean_{window_name}'] = data.groupby(['Market', 'Classification'])['Supply Volume'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        
        data[f'Wholesale_rolling_std_{window_name}'] = data.groupby(['Market', 'Classification'])['Wholesale'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        data[f'Retail_rolling_std_{window_name}'] = data.groupby(['Market', 'Classification'])['Retail'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        data[f'Supply_Volume_rolling_std_{window_name}'] = data.groupby(['Market', 'Classification'])['Supply Volume'].transform(lambda x: x.rolling(window=window_size, min_periods=1).std())

    # Fill missing values with forward and backward fill
    data = data.bfill().ffill()

    # Binary encoding for categorical columns
    columns_to_encode = ['County', 'Market', 'Classification']
    binary_encoder = ce.BinaryEncoder(cols=columns_to_encode, return_df=True)
    data_encoded = binary_encoder.fit_transform(data)

    # Define columns to normalize
    columns_to_normalize = [
        'Wholesale', 'Retail', 'Supply Volume',
        'Wholesale_rolling_mean_7d', 'Retail_rolling_mean_7d',
        'Supply_Volume_rolling_mean_7d', 'Wholesale_rolling_std_7d',
        'Retail_rolling_std_7d', 'Supply_Volume_rolling_std_7d',
        'Wholesale_lag_7', 'Retail_lag_7', 'Supply_Volume_lag_7',
        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter',
        'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Quarter_sin', 'Quarter_cos',
        'Year_sin', 'Year_cos'
    ]

    # Scaling the selected columns
    scalers = {}
    for column in columns_to_normalize:
        scaler = MinMaxScaler()
        data_encoded[column] = scaler.fit_transform(data_encoded[[column]])
        scalers[column] = scaler

    # Save the scalers for future use
    scaler_file_path = 'models/scalers.pkl'
    os.makedirs(os.path.dirname(scaler_file_path), exist_ok=True)
    dump(scalers, scaler_file_path)

    # Filter the most relevant features based on correlation threshold
    correlation_matrix = data_encoded.corr()
    correlation_with_target = correlation_matrix[['Retail', 'Wholesale']]
    
    # Set a threshold for correlation
    threshold = 0.1
    filtered_columns = correlation_with_target[(correlation_with_target['Retail'].abs() > threshold) | (correlation_with_target['Wholesale'].abs() > threshold)].index.tolist()

    # Exclude target columns from the filtered columns list
    filtered_columns = [col for col in filtered_columns if col not in ['Retail', 'Wholesale']]

    # Select final columns for modeling
    final_modeling_data = data_encoded[filtered_columns].copy()
    final_modeling_data = final_modeling_data.join(data_encoded[['Wholesale', 'Retail']])

    # Save the final modeling data to a CSV file
    final_modeling_data.to_csv(output_csv, index=False)

    return final_modeling_data



# Create the Models directory if it doesn't exist
if not os.path.exists('Models'):
    os.makedirs('Models')

def train_and_evaluate_lstm_wholesale(data_path, save_model_path='Models/wholesale_lstm_model.h5'):
    # Load the data
    data = pd.read_csv(data_path)

    # Drop date if it's in the dataset
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])

    # Drop rows with missing values
    data = data.dropna()

    # Define the target variable for Wholesale
    target = data['Wholesale']

    # Define features excluding the target variable
    features = data.drop(columns=['Wholesale', 'Retail'])

    # Splitting the data into train, validation, and test sets
    train_size = int(len(features) * 0.7)
    validation_size = int(len(features) * 0.15)
    test_size = len(features) - train_size - validation_size

    train_features, test_features = features[:train_size], features[train_size:]
    train_target, test_target = target[:train_size], target[train_size:]

    validation_features, test_features = test_features[:validation_size], test_features[validation_size:]
    validation_target, test_target = test_target[:validation_size], test_target[validation_size:]

    # Reshape features to be compatible with LSTM input
    train_X = train_features.values.reshape((train_features.shape[0], 1, train_features.shape[1]))
    validation_X = validation_features.values.reshape((validation_features.shape[0], 1, validation_features.shape[1]))
    test_X = test_features.values.reshape((test_features.shape[0], 1, test_features.shape[1]))

    # Create LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(units=50, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Instantiate and summarize the model
    model = create_model()
    model.summary()

    # Train the model
    history = model.fit(train_X, train_target, 
                        epochs=20, 
                        batch_size=64, 
                        validation_data=(validation_X, validation_target), 
                        verbose=2)

    # Make predictions on the test set
    test_predictions = model.predict(test_X).flatten()

    # Calculate metrics
    mse = mean_squared_error(test_target, test_predictions)
    rmse = np.sqrt(mse)
    pmse = mean_absolute_percentage_error(test_target, test_predictions)

    # Plot the actual vs predicted values
    actual = np.array(test_target).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(test_predictions, label='Predicted Values', color='red', linestyle='--', linewidth=2)
    plt.title('Comparison of Actual and Predicted Wholesale Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Wholesale Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save the model
    model.save(save_model_path)

    return model, mse, pmse, rmse, plt


def train_and_evaluate_lstm_retail(data_path, save_model_path='Models/retail_lstm_model.h5'):
    # Load the data
    data = pd.read_csv(data_path)

    # Drop date if it's in the dataset
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])

    # Drop rows with missing values
    data = data.dropna()

    # Define the target variable for Retail
    target = data['Retail']

    # Define features excluding the target variable
    features = data.drop(columns=['Wholesale', 'Retail'])

    # Splitting the data into train, validation, and test sets
    train_size = int(len(features) * 0.7)
    validation_size = int(len(features) * 0.15)
    test_size = len(features) - train_size - validation_size

    train_features, test_features = features[:train_size], features[train_size:]
    train_target, test_target = target[:train_size], target[train_size:]

    validation_features, test_features = test_features[:validation_size], test_features[validation_size:]
    validation_target, test_target = test_target[:validation_size], test_target[validation_size:]

    # Reshape features to be compatible with LSTM input
    train_X = train_features.values.reshape((train_features.shape[0], 1, train_features.shape[1]))
    validation_X = validation_features.values.reshape((validation_features.shape[0], 1, validation_features.shape[1]))
    test_X = test_features.values.reshape((test_features.shape[0], 1, test_features.shape[1]))

    # Create LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(units=50, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Instantiate and summarize the model
    model = create_model()
    model.summary()

    # Train the model
    history = model.fit(train_X, train_target, 
                        epochs=20, 
                        batch_size=64, 
                        validation_data=(validation_X, validation_target), 
                        verbose=2)

    # Make predictions on the test set
    test_predictions = model.predict(test_X).flatten()

    # Calculate metrics
    mse = mean_squared_error(test_target, test_predictions)
    rmse = np.sqrt(mse)
    pmse = mean_absolute_percentage_error(test_target, test_predictions)

    # Plot the actual vs predicted values
    actual = np.array(test_target).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(test_predictions, label='Predicted Values', color='red', linestyle='--', linewidth=2)
    plt.title('Comparison of Actual and Predicted Retail Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Retail Prices')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save the model
    model.save(save_model_path)

    return model, mse, pmse, rmse, plt


# # For Wholesale model
# wholesale_model, wholesale_mse, wholesale_pmse, wholesale_rmse, wholesale_plot = train_and_evaluate_lstm_wholesale('path_to_your_data.csv')

# # For Retail model
# retail_model, retail_mse, retail_pmse, retail_rmse, retail_plot = train_and_evaluate_lstm_retail('path_to_your_data.csv')























