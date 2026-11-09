from Data_preprocessing import load_data, clean_data, explore
from Model_Train import add_features

def main():
    df = load_data()
    df = clean_data(df)
    df, X, y = add_features(df)   
    explore(df)

if __name__ == "__main__":
    main()