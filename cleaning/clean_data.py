import pandas as pd
import numpy as np


def clean_data(df3):
    #         ==============================================================================
    #                               DATA CLEANING
    #         ============================================================================== 
    print(df3["isFraud"].unique())  # there is only 0 and 1 in isFraud column so we don't need to clean this colu# ==============================================================================
    #                      DATA CLEANING - CATEGORICAL FEATURES
    # ==============================================================================
    categorical_features=df3.select_dtypes(include=['object']).columns
    #print("Categorical Features:", categorical_features)       
    #print all type of element in categorical features    
    l=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1',
           'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15',
           'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33',
           'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
           'DeviceInfo']   
    #for i in l:
    #print(df3[i].unique())   
    #print(df3["P_emaildomain"].value_counts().get("gmail", 0))
    #print(df3["P_emaildomain"].value_counts().get("gmail.com", 0))

    #print(df3["R_emaildomain"].value_counts().get("gmail", 0))
    #print(df3["R_emaildomain"].value_counts().get("gmail.com", 0))     
    # Only 6 rows contain 'gmail' while other contain 'gmail.com'.
    # Treat 'gmail' as an inconsistent representation and standardize it to 'gmail.com'.
    df3['P_emaildomain'] = df3['P_emaildomain'].replace('gmail', 'gmail.com')  # change  gmail to gmail.com  
    df3['R_emaildomain'] = df3['R_emaildomain'].replace('gmail', 'gmail.com') # chane the gmail to gmail.com
    #print((df3["id_33"] == "0x0").sum())  # so we replace this by np.nan because it is not useful for our model and it is not a valid value in id_33
    df3['id_33'] =df3['id_33'].replace('0x0',np.nan)  # change the 0x0 to Nan in id-33             
    #print(df3['card6'].value_counts(normalize=True))        
    # debit              0.746963
     #credit             0.252961
    #debit or credit    0.000051
    #charge card        0.000025
    #Name: proportion, dtype: float64 that why we can drop the charge card and debit or credit because they are less count also we  can check the mean of fraud in debit or credit and charge card
    #print(df3.groupby('card6')['isFraud'].mean())   # debit or credit and charge card having zero  mean of fraud so we can drop them because they are not useful for our model   
    #card6
    #charge card        0.000000
    #credit             0.066785
    #debit              0.024263
    #debit or credit    0.000000
    #Name: isFraud, dtype: float64  so the code is below to drop the charge card and debit or credit from card6 with name other 
    df3['card6']=df3['card6'].replace(['charge card', 'debit or credit'], 'other')  
    # clean id_30 (OS info) - standardize OS family, remove version numbers, fix invalid values
    #print((df3['id_30']=="func").sum())  # check the value counts of id_30 column  
    def clean_id30(val):
        if pd.isna(val):
            return val                    # keep existing nulls as nulls
        val = val.lower()
        if 'func' in val:
            return np.nan                 # 'func' is invalid/garbage data, treat as missing   
                               #         'func' appears only 10 times and is not a valid operating system.
                                                  # Treat it as missing data.
        if 'windows' in val:
            return 'windows'              # group all Windows versions (Windows 10, Windows 7, etc.) into one category
        if 'mac' in val:
            return 'mac'                  # group all Mac OS X versions into one category
        if 'ios' in val:
            return 'ios'                  # group all iOS versions into one category
        if 'android' in val:
            return 'android'              # group all Android versions into one category
        if 'linux' in val:
            return 'linux'                # group Linux into one category
        return 'other'                    # anything unrecognized (e.g. blank/rare OS names) -> other

    df3['id_30'] = df3['id_30'].apply(clean_id30)   # apply OS cleanup to id_30 column  
    # ==============================================================================
    #                       DATA CLEANING - NUMERICAL FEATURES
    # ==============================================================================  
    #print all numerical columns in the datset 
    num_cols = df3.select_dtypes(include=["int64", "float64"]).columns
    #print(f"Total numerical columns: {len(num_cols)}")
    # so total no of numerical column is 403  
    #print(num_cols)       
    #Now we can check that which numerical coulmn having negative value  
    negative_counts = {}
    for col in num_cols:
        if (df3[col] < 0).any():
            negative_counts[col] = (df3[col] < 0).sum()
    negative_counts  
    #print(f"Columns with negative values: {len(negative_counts)}")  
    # so number of columns having negative values is 16 

    negative_cols = []

    for col in num_cols:
        if (df3[col] < 0).any():       #here we print all the column having negative value 
            negative_cols.append(col)
    #print("column with negative value are ",negative_cols)  
    negative_cols=['D4', 'D6', 'D11', 'D12', 'D14', 'D15', 'id_01', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_14'] 
    #print(df3[negative_cols].describe())  #from this we decided that not chnage the negative value to postive 
                      #NOW missing value ananlysis of numerical columnss 
    missing_num = df3[num_cols].isnull().sum().sort_values(ascending=False)

    missing_num = missing_num[missing_num > 0]

    missing_df = pd.DataFrame({
        "Missing Count": missing_num,
        "Missing %": (missing_num / len(df3) * 100).round(2)
    })
    missing_df 
    #print(missing_df)
    missing_percent = (df3[num_cols].isnull().mean() * 100)

    #print(">= 90% :", (missing_percent >= 90).sum())
    #print("70-90% :", ((missing_percent >= 70) & (missing_percent < 90)).sum())
    #print("40-70% :", ((missing_percent >= 40) & (missing_percent < 70)).sum())
    #print("10-40% :", ((missing_percent >= 10) & (missing_percent < 40)).sum())
    #print("< 10%  :", (missing_percent < 10).sum())

    missing_percent = (df3[num_cols].isnull().mean() * 100)
    high_missing_cols = missing_percent[missing_percent >= 90].sort_values(ascending=False)
    #print(high_missing_cols)    
        #The column having more than 90% missing value is  id_24    99.196159
    #id_25    99.130965
    #id_07    99.127070
    #id_08    99.127070
    #id_21    99.126393
    #id_26    99.125715
    #id_22    99.124699
    #dist2    93.628374
    #D7       93.409930
    #id_18    92.360721   
      #so the conclusinon is Columns with >99% missing values are candidates for removal. Final removal will be performed during preprocessing after evaluating their usefulness.
    #now for check zero value in numerical column
    zero_counts = (df3[num_cols] == 0).sum()
    zero_counts = zero_counts[zero_counts > 0].sort_values(ascending=False)
    #print(zero_counts) 
     # ============================================================
    # NUMERICAL DATA CLEANING SUMMARY
    # ============================================================

    # 1. Identified all numerical columns in the dataset.
    # 2. Verified the data types of all numerical features.
    # 3. Checked all numerical columns for negative values.
    # 4. Negative values were found only in some D and id_* features.
    #    These values were retained because they are valid for these features.
    # 5. Analyzed missing values in all numerical columns.
    # 6. Found 10 numerical columns with more than 90% missing values.
    #    These columns were identified as candidates for removal, but no columns
    #    were dropped during data cleaning. The final decision will be made
    #    during preprocessing.
    # 7. Observed that many numerical features contain zero values.
    #    Zero values were retained because they represent valid information
    #    and are not necessarily missing values.
    # 8. Analyzed the distribution of important numerical features.
    #    - TransactionAmt is right-skewed (Mean > Median).
    #    - TransactionDT is approximately symmetric (Mean ≈ Median).
    # 9. Outliers were not removed because tree-based models are robust to
    #    outliers, and extreme values may represent fraudulent transactions.
    # 10. Numerical data cleaning is complete.
    #     Missing value imputation, feature selection, encoding, and scaling
    #     will be performed during the preprocessing stage.

    return df3