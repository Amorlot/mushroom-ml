from src.data_loader import DataLoader

def test_data_loader():
    dl = DataLoader(
        dataset_id=73,
        target_col='poisonous',
        drop_cols=['veil-type'],
        drop_missing_thresh=0.5
    )
    dl.load()
    dl.info()
    dl.report_missing()

    assert dl.df is not None
    assert dl.X is not None
    assert dl.y is not None
    assert len(dl.df) == 8124
    assert 'poisonous' not in dl.X.columns
    assert 'veil-type' not in dl.df.columns

    print("\nTutti i test passati!")

if __name__ == "__main__":
    test_data_loader()
