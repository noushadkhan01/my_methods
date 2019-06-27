class MyDummyVariable:
  '''it is a class to get dummy variable from a dataframe in this method we are using OneHotEncoder 
    and this method automatically distinguish numeric and categorical columns'''
  #initialize OneHotEncoder for future use when we transform data
  ohe_encoders = {}
  def __init__(self, drop_first = True):
    self.drop_first = drop_first
 
  #fit_transform
  def fit_transform(self, features):
    '''features must be an dataframe
    it requires an argument features which is a dataframe containing numeric and categorical columns'''
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import numpy as np
    import pandas as pd
    categorical_data = features.select_dtypes(include = 'object')
    categorical_ohe = None
    for i in categorical_data.columns:
      ohe = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
      encoded_column = ohe.fit_transform(categorical_data[[i]].values)
      if self.drop_first:
        encoded_column = encoded_column[:, 1:]
      MyDummyVariable.ohe_encoders[i] = ohe
      if categorical_ohe:
        categorical_ohe = np.concatenate([categorical_ohe, encoded_columns], axis = 1)
      else:
        categorical_ohe = encoded_column
        
      return self.combined_dataset(features, categorical_ohe)
  
  
  #combine data
  def combined_dataset(self, features, categorical_ohe):
    import numpy as np
    #combine categorical_ohe and numeric columns
    numeric_data = features.select_dtypes(exclude = 'object').values
    
    #OneHotEncoded Data 
    ohe_data = np.concatenate([categorical_ohe, numeric_data], axis = 1)
    return ohe_data
  
  #transform data
  def transform(self, features):
    '''features must be an dataframe
    it requires an argument features which is a dataframe containing numeric and categorical columns'''
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    import numpy as np
    import pandas as pd
    categorical_data = features.select_dtypes(include = 'object')
    categorical_ohe = None
    for i in categorical_data.columns:
      encoded_column = MyDummyVariable.ohe_encoders[i].transform(categorical_data[[i]].values)
      if self.drop_first:
        encoded_column = encoded_column[:, 1:]
      
      if categorical_ohe:
        categorical_ohe = np.concatenate([categorical_ohe, encoded_columns], axis = 1)
      else:
        categorical_ohe = encoded_column
        
      return self.combined_dataset(features, categorical_ohe)
