
import pandas as pd
pd.set_option('display.max_colwidth', None)

 # Load data and set labels
grocery_data = pd.read_csv('data/groceries_dataset.csv')
#print(grocery_data.head(5))
shopping_df =  grocery_data[(grocery_data["label"]=="add") | (grocery_data["label"]=="remove")]
shopping_df.drop(['label'], inplace=True, axis=1)
shopping_df['label']=1
print(shopping_df.info())

# Save shopping data for testing
shopping_df_for_test = shopping_df.sample(frac=0.01, random_state=42)  # Adjust frac as needed
shopping_df.drop(shopping_df_for_test.index,inplace=True)
shopping_df_for_test.drop(['label'], inplace=True, axis=1)
print(shopping_df.info())
print(shopping_df_for_test.info())

print(shopping_df.head(5))
non_shopping_df = grocery_data[(grocery_data["label"]=="ignore")]
non_shopping_df.drop(['label'], inplace=True, axis=1)
non_shopping_df['label']=0
print(non_shopping_df.info())

# Save non_shopping data for testing
non_shopping_df_for_test = non_shopping_df.sample(frac=0.1, random_state=42)  # Adjust frac as needed
non_shopping_df.drop(non_shopping_df_for_test.index,inplace=True)
non_shopping_df_for_test.drop(['label'], inplace=True, axis=1)
print(non_shopping_df.info())
print(non_shopping_df_for_test.info())

shopping_sentences_labled = pd.concat([shopping_df, non_shopping_df], axis=0).reset_index(drop=True)
print(shopping_sentences_labled.sample(5))
shopping_sentences_labled.to_csv("shopping_sentences_labled.csv",index=False)

shopping_sentences_test = pd.concat([shopping_df_for_test, non_shopping_df_for_test], axis=0).reset_index(drop=True)
print(shopping_sentences_test.sample(20))
shopping_sentences_test.to_csv("shopping_sentences_test.csv",index=False)
