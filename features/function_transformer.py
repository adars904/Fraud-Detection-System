def check_skew(x_train):
    #====== ===FUNCTION TRANSFORMER ===================================================  
    #=======first we find the numerical column  name in train dataset            
    numerical_columntrain = x_train.select_dtypes(include=['int64','float64']).columns
    numerical_columntrain = list(numerical_columntrain)
    # now we will find the skew of all the column    
    #  if skew is positve then this column is right skewed     
    # if skew  is negative  then  this column is left skewed  
    # if skew is 0 then it is normal   skewed 

    for col in numerical_columntrain:
        skew_values = x_train[col].skew()
        #print(col,skew_values)
    # Observation:
    # Many numerical features are highly skewed.

    # Decision:
    # Skewness transformation is not applied because this project uses
    # tree-based models (Random Forest/XGBoost/LightGBM/CatBoost),
    # which are generally robust to skewed feature distributions.
    # Therefore, FunctionTransformer is not included in the preprocessing pipeline.