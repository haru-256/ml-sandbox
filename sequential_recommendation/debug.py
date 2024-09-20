from data.dataset import fetch_dataset, fetch_metadata, preprocess_dataset

if __name__ == "__main__":
    dataset_dict = fetch_dataset()
    metadata = fetch_metadata()
    dataset = preprocess_dataset(dataset_dict, metadata)
