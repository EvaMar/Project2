import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load the files input by the user."""
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge message and categories datasets together

    df = messages.merge(categories, how='left', on=('id'))
    
    return df

def clean_data(df):
    """Process the data to get it ready for the model."""
    
    #id is not a measure so the colde will make sure it is not a number
    df['id'] = df['id'].astype(str) 
    
    # the categories column is just one column where every category is separated by ;
    # each category needs to be split into a separate column in order to work in a ML algorithm
    
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # this row will contais the names of the columns, the below code will remove all -1, -0 etc.
    
    category_colnames = row.apply(lambda x: x[0:-2])
    print(category_colnames)
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Now that all the columns contatins the message type name, each column should cointains 1 or 0
    #This structire will allow machine learning algorithm to recognise if a message is a request or an offer, or medical help, etc.
    
    
    for column in categories:
        #set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 
    
    #Now the code will replace the new ML 0/1 category column with the old category column in the df dataset that is no longer needed
    
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    category_names = df.columns.tolist()
    category_names.remove('id')
    category_names.remove('message')
    category_names.remove('original')
    category_names.remove('genre')
    
    #we can only accept 1 or 0 value
    for i in range(len(category_names)):
        print('Binary for ',category_names[i])
        print('Before Binary method')
        print(df[category_names[i]].value_counts())
        df = df[(df[category_names[i]] == 1) | (df[category_names[i]] == 0)]
        print('After binary')
        print(df[category_names[i]].value_counts())
    
    # The code now needs to delete duplicates
    
    
    # check number of duplicates
    # the id number should be unique. the follwoing code will check if it is the case:
    
    #getting recurrences of the same id
    num_dup = df.groupby(['id']).size().reset_index(name='counts').sort_values('counts',ascending = False)
    #excluding ids that don't have duplicates
    num_dup = num_dup.loc[num_dup['counts'] > 1]
    
    #print the results
    print('number of unique ids with duplicates:',num_dup['id'].nunique(), 'Number of duplicates:', num_dup['counts'].sum() - num_dup['id'].nunique())
    
    # drop duplicates
    
    # recording the number of rows the dataset has before removing duplicates
    num_rows_bef_dup = df.shape[0]
    
    #removing duplicates
    df = df.drop_duplicates()
    
    #number of rwos after removing duplicates
    num_rows_aft_dup = df.shape[0]
    
    print('Before duplicates removal the dataset has: ',num_rows_bef_dup, 'rows. After duplicates removal the datasets has: ',num_rows_aft_dup)
    print('The difference of the above number, ',num_rows_bef_dup - num_rows_aft_dup, ', should be the same as the number of duplicates IDs previously found: ',num_dup['counts'].sum() - num_dup['id'].nunique()) 
    
    if num_rows_bef_dup - num_rows_aft_dup != num_dup['counts'].sum() - num_dup['id'].nunique():
        print('Some unique ids display different combinations of categories, causing duplicates')
    else:
        print('duplicates eliminated')
    
    # if unique IDs displays different combination of categories, this needs to be solved
    # The code will need to select one combination of categories. In the attempt to not lose information, the most conservative approach will be used:
    # In case of different combinations of categories per same id, the one with more 1s will be selected
    
    #getting the sum of the 1s for each id in temporary dataframe
    df['Categories_sum'] = df.sum(axis=1, numeric_only=True)
    
    
    #getting the max number of 1s that a single Id can display
    
    df_max = df.groupby('id')['Categories_sum'].max().reset_index(name='Categories_max')
    
    #df_max will be merged with df. Any combination of categories that does not correspond to the maximum combination will be dropped
    
    df = df.merge(df_max, how='left', on=('id'))
    df = df.loc[df['Categories_sum'] == df['Categories_max']]
    
    print('After removing duplicate ids, based on the most consevative approach, the dataset shows number of rows: ', df.shape[0], 'with unique ids: ', df['id'].nunique())
    
    #there still may be duplicates of the same id with the same amount od 1s (althought different). in this case, the code will only keep based on index
    df = df.drop_duplicates(subset=['id'])
    
    print('Final cleanse. Number of rows: ', df.shape[0], 'unique ids:', df['id'].nunique())

    return df


def save_data(df, database_filename):
    """Save the data in a specific input location as an sql database."""
    
    engine = create_engine(database_filename)
    df.to_sql('MessagesML', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              '"sqlite:///Disaster.db"')


if __name__ == '__main__':
    main()