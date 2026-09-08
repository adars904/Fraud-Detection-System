def drop_high_missing_cols(x_train, x_test):
    # ==============================================================================
    #                    DROP HIGH-MISSING COLUMNS (>99% missing)
    # ==============================================================================
    # Imputing columns with ~99% missing values is unlikely to produce reliable
    # values and may introduce noise - dropped before further preprocessing.
    # before droppint the column we check the missing invvolment in the fraud and non fraud transaction 
    high_missing_cols = ['id_24','id_25','id_07','id_08','id_21','id_26','id_22','dist2','D7','id_18']

    #for col in high_missing_cols:
        #print(col)
       #print(df3.groupby(df3[col].isnull())['isFraud'].mean())
       # print(df3[col].isnull().mean()*100)
       # print()    
    # Columns with approximately 99% missing values are removed before handling
    # missing data because they contain very few non-missing observations.
    # Imputing such columns is unlikely to produce reliable values and may
    # introduce unnecessary noise into the model. Therefore, these features
    # are dropped as part of the initial data preprocessing step. 

    removed_cols=["id_24","id_25","id_07","id_08","id_21","id_26","id_22"]
    for col in removed_cols:
        x_train.drop(columns=[col],inplace=True)
        x_test.drop(columns=[col],inplace=True)  

    return x_train, x_test