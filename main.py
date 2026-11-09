from Data_preprocessing import load_data, clean_data, explore
from Model_Train import add_features
from Predict import train_model
def main():
    df = load_data()
    df = clean_data(df)
    df, X, y = add_features(df)
    explore(df)
    df, model, y_pred, comparison = train_model(df, X, y)

if __name__ == "__main__":
    main()