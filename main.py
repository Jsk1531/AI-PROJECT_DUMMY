from Data_preprocessing import load_data, clean_data, explore
from Model_Train import add_features
from Predict import train_model
from Error import evaluate_model                            
def main():
    df = load_data()
    df = clean_data(df)
    df, X, y = add_features(df)
    explore(df)
    df, model,y_test,y_pred,comparison = train_model(df, X, y)
    mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)                 
if __name__ == "__main__":
    main()