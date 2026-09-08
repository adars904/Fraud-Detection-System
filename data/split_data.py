from sklearn.model_selection import train_test_split


def train_test_split_data(df3):
                # ============================================================    
                #  TRAIN  TEST  SPLIT 
                #=============================================================  
    #define the feature and target values
    x=df3.drop(columns=["isFraud"])
    y=df3["isFraud"] 
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    return x_train, x_test, y_train, y_test


def train_val_split_data(x_train, y_train):
    # ==============================================================================
    # STEP: TRAIN / VALIDATION SPLIT (FOR THRESHOLD TUNING - NOT FOR GridSearchCV)
    # ==============================================================================
    # WHAT:
    # We split x_train/y_train further into a smaller training set
    # (x_train_model/y_train_model) used for GridSearchCV, and a validation set
    # (x_val/y_val) reserved for threshold selection below.
    #
    # WHY:
    # Choosing the probability threshold that turns predicted probabilities
    # into hard 0/1 fraud predictions is itself a modeling decision. If that
    # threshold were chosen by looking at x_test, x_test would no longer be a
    # clean, untouched hold-out set for the final evaluation - it would have
    # already influenced a decision about the model. To keep x_test completely
    # untouched until FINAL evaluation, we carve a validation set out of the
    # training data specifically for threshold tuning:
    #
    #   x_train -> x_train_model (GridSearchCV) + x_val (threshold tuning)
    #   x_test  -> used ONLY for the final evaluation, later in this script
    #
    # stratify=y_train is used for the same reason stratify=y was used in the
    # original train/test split: this is an imbalanced fraud dataset, and a
    # non-stratified split could leave x_val with very few fraud examples,
    # making the threshold chosen from it unreliable.
    x_train_model, x_val, y_train_model, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    return x_train_model, x_val, y_train_model, y_val