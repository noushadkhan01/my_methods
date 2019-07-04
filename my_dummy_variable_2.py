class MyDummyVariable:
  '''it is a class to get dummy variable from a dataframe in this method we are using OneHotEncoder 
    and this method automatically distinguish numeric and categorical columns'''
  #initialize OneHotEncoder for future use when we transform data
  ohe_encoders = {}
  def __init__(self, drop_first = True, categorical_features = None, all_categorical = False):
    self.drop_first = drop_first
    self.categorical_features = categorical_features
    self.all_categorical = False
 
  #fit_transform
  def __repr__(self):
    return f'MyDummyVariable(drop_first = {self.drop_first}, categorical_features = {self.categorical_features})'

  #separate numerica and categorical data
  def separate_data(self, features):
    import numpy as np
    import pandas as pd
    if self.categorical_features:
      if type(self.categorical_features) == list or type(self.categorical_features) == tuple:
        categorical_data = features.iloc[: ,self.categorical_features]
        if not self.all_categorical:
          numeric_data = features.iloc[: , [i for i in np.arange(features.shape[1]) if i not in self.categorical_features]]
        else:
          numeric_features = pd.DataFrame()
      else:
        raise TypeError(f'Type of Categorical_fetures {self.categorical_features} must be list or tuple')
    else:
      categorical_data = features.select_dtypes(include = 'object')
      numeric_data = features.select_dtypes(exclude = 'object')
    return categorical_data, numeric_data
    
    
  def fit_transform(self, features):
    '''features must be an dataframe
    it requires an argument features which is a dataframe containing numeric and categorical columns'''
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    import pandas as pd
    categorical_data, numeric_data = self.separate_data(features)
    categorical_ohe = None
    for i in categorical_data.columns:
      #print(f'doing for feature {i}')
      ohe = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
      encoded_column = ohe.fit_transform(categorical_data[[i]].values)
      #print(f'encoded column shape for feature {i} is {encoded_column.shape}')
      if self.drop_first:
        encoded_column = encoded_column[:, 1:]
      MyDummyVariable.ohe_encoders[i] = ohe
      if type(categorical_ohe) == 'np.int64':
        categorical_ohe = np.concatenate([categorical_ohe, encoded_columns], axis = 1)
      else:
        categorical_ohe = encoded_column
       
    return self.combined_dataset(features, categorical_ohe, numeric_data)
  
  
  #combine data
  def combined_dataset(self, features, categorical_ohe, numeric_data):
    import numpy as np
    #combine categorical_ohe and numeric columns
    ohe_data = np.concatenate([categorical_ohe, numeric_data], axis = 1)
    return ohe_data
  
  #transform data
  def transform(self, features):
    '''features must be an dataframe
    it requires an argument features which is a dataframe containing numeric and categorical columns'''
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np
    import pandas as pd
    categorical_data, numeric_data = self.separate_data(features)
    categorical_ohe = None
    for i in categorical_data.columns:
      encoded_column = MyDummyVariable.ohe_encoders[i].transform(categorical_data[[i]].values)
      if self.drop_first:
        encoded_column = encoded_column[:, 1:]
      
      if type(categorical_ohe) == 'np.int64':
        categorical_ohe = np.concatenate([categorical_ohe, encoded_columns], axis = 1)
      else:
        categorical_ohe = encoded_column
        
    return self.combined_dataset(features, categorical_ohe, numeric_data)
