import pandas as pd
import os

class FashionDataAPI:
    def __init__(self):
        # Define the path to the CSV file
        csv_path = os.path.join('Data', 'store_zara.csv')
        # Load the CSV file into a pandas DataFrame
        self.catalog_data = pd.read_csv(csv_path)
        columns_to_drop = ["brand", "sku", "currency", "scraped_at", "image_downloads"]
        self.catalog_data.drop(columns=columns_to_drop, inplace=True)
        print(self.catalog_data.head())

        # Load the Common_Outfit_Styles.csv file into a pandas DataFrame
        style_csv_path = os.path.join('Data', 'Common_Outfit_Styles.csv')
        self.style_data = pd.read_csv(style_csv_path)
        print(self.style_data.head())
    
    def query_data(self, query):
        # Placeholder for query logic
        # You can implement specific query methods here
        return self.catalog_data.query(query)

# Instantiate the API
fashion_api = FashionDataAPI()

# Example usage of the API
# result = fashion_api.query_data("your_query_here")
# print(result)

# You can add additional processing or database creation logic here
