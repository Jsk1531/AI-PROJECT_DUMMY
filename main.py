from Data_preprocessing import load_data, clean_data, explore

def main():
    df = load_data()
    df = clean_data(df)
    explore(df)

if __name__ == "__main__":
    main()