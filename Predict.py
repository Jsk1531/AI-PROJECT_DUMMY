from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from IPython.display import display 
import pandas as pd  
def train_model(df,X,y):              
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("definig the features and Spliting data done ")

    model=LinearRegression()
    print("model tranig started")
    model.fit(X_train, y_train)
    print("Tranig is done ")
    print("Model is Ready to Predict")
    y_pred = model.predict(X_test)
    comparison = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": y_pred,
        "Error (Actual - Predicted)": y_test.to_numpy() - y_pred
    })
    print(comparison.head(10))
    return df,model,y_test,y_pred,comparison
