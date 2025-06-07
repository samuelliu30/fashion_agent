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

        # Load the Common_Outfit_Styles.csv file into a pandas DataFrame
        style_csv_path = os.path.join('Data', 'Common_Outfit_Styles.csv')
        self.style_data = pd.read_csv(style_csv_path)
    
    def query_data(self, query):
        # Placeholder for query logic
        # You can implement specific query methods here
        return self.catalog_data.query(query)
    
    def get_categories(self):
        # For our store_zara data, we have the following categories:
        # ['jackets' 'puffers' 'pants' 'jeans' 'sweaters' 'cardigans' 'hoodies'
        # 'sweatshirts' 't-shirts' 'overshirts' 'linen' 'shorts' 'suits' 'blazers'
        # 'tracksuits' 'coats' 'shoes' 'bags' 'dresses' 'skirts' 'tops' 'bodysuits'
        # 'knitwear']

        # The method will be called by the agent to get the categories
        # The agent will match the style with the categories
        return self.catalog_data['terms'].unique()


# Instantiate the API
fashion_api = FashionDataAPI()

# Example usage of the API
# result = fashion_api.query_data("your_query_here")
# print(result)

# You can add additional processing or database creation logic here
