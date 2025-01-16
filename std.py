# Question No 1
import requests
import pandas as pd

url = 'https://raw.githubusercontent.com/Blaq5/CSV-Testing/refs/heads/main/Sales_Transactions_Dataset.csv'

response = requests.get(url)
open('Sales_Transactions_Dataset.csv', 'wb').write(response.content)

data = pd.read_csv('Sales_Transactions_Dataset.csv', nrows=200)
data.to_csv('For_Prediction.csv', index=False)



# Question No 2
import requests
import pandas as pd
url = 'https://raw.githubusercontent.com/Blaq5/CSV-Testing/refs/heads/main/sandstone_and_shale_Data1.csv'

try:

        response = requests.get(url)
        response.raise_for_status()  
        
        with open('Sales_Transactions_Dataset.csv', 'wb') as file:
            file.write(response.content)
        print("CSV file downloaded successfully.")
except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the CSV file: {e}")



        
        
        
        
        
        
        
        
        
        

# Question No 3
import pandas as pd

# Load the dataset
dataset_path = 'For_Prediction.csv'  # Replace with your dataset path
data = pd.read_csv(dataset_path)

# Get the number of rows and columns
num_rows, num_columns = data.shape

# Get the unique classes in the last column
unique_classes = data.iloc[:, -1].unique()
num_classes = len(unique_classes)

# Print the results
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')
print(f'Number of unique classes in the last column: {num_classes}')











# Question No 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Department\Sales_Transactions_Dataset.csv')


x =data[['Quantity', 'TotalSalesValue']]
y = data[['TotalSalesValue']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(y_pred)