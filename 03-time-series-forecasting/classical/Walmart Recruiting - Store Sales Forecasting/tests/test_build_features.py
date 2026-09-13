import pandas as pd

from src.features.build_features import add_lags, build_training_frame, time_split


def _toy_frame():
    dates = pd.date_range("2012-01-01", periods=6, freq="W")
    return pd.DataFrame({
        "Date": dates,
        "Store": [1] * 6,
        "Dept": [1] * 6,
        "Weekly_Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


def test_add_lags_shifts_within_series():
    df = add_lags(_toy_frame())
    assert df["lag_1"].isna().tolist() == [True, False, False, False, False, False]
    assert df["lag_1"].dropna().tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_add_lags_does_not_leak_across_series():
    df = _toy_frame()
    other = df.copy()
    other["Store"] = 2
    other["Weekly_Sales"] = [1000.0] * 6
    combined = pd.concat([df, other], ignore_index=True)

    result = add_lags(combined)
    store1_lag1 = result[result["Store"] == 1]["lag_1"].dropna().tolist()
    assert 1000.0 not in store1_lag1


def test_build_training_frame_splits_by_target_na():
    df = _toy_frame()
    df["Size"] = 100
    df["month"] = df["Date"].dt.month
    df["IsHoliday"] = 0
    df["MarkDown_Total"] = 0
    df["MarkDown_Count"] = 0

    train = df.iloc[:4].copy()
    test = df.iloc[4:].drop(columns=["Weekly_Sales"]).copy()

    train_with_lags, test_with_lags = build_training_frame(train, test)
    assert train_with_lags["Weekly_Sales"].notna().all()
    assert test_with_lags["Weekly_Sales"].isna().all()
    assert len(train_with_lags) + len(test_with_lags) == len(df)


def test_time_split_respects_week_boundary():
    dates = pd.date_range("2012-01-01", periods=20, freq="W")
    df = pd.DataFrame({
        "Date": dates,
        "Store": 1, "Dept": 1, "Size": 100, "month": dates.month,
        "IsHoliday": 0, "MarkDown_Total": 0, "MarkDown_Count": 0,
        "lag_1": range(20), "lag_4": range(20), "lag_52": range(20),
        "roll_mean_4": range(20), "roll_mean_52": range(20),
        "Weekly_Sales": range(20),
    })
    train_split, valid_split = time_split(df, weeks=8)
    assert valid_split["Date"].min() > df["Date"].max() - pd.Timedelta(weeks=8)
    assert len(train_split) + len(valid_split) == len(df)
