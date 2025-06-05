import pandas as pd
import os

class FashionDataAPI:
    def __init__(self):
        # Define the path to the CSV file
        csv_path = os.path.join('Data', 'store_zara.csv')
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_path)
        print("Data loaded successfully!")
        print(self.data.head())
        self.drop_columns()
    
    def drop_columns(self):
        columns_to_drop = ["brand", "sku", "currency", "scraped_at", "image_downloads"]
        self.data.drop(columns=columns_to_drop, inplace=True)
        print("Columns dropped successfully!")
        

    def query_data(self, query):
        # Placeholder for query logic
        # You can implement specific query methods here
        return self.data.query(query)

# Instantiate the API
fashion_api = FashionDataAPI()

# Example usage of the API
# result = fashion_api.query_data("your_query_here")
# print(result)

# You can add additional processing or database creation logic here
