import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_wmae, evaluate, wmae_by_group


def test_calculate_wmae_no_holidays_equals_mae():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    is_holiday = np.array([False, False, False])
    assert calculate_wmae(y_true, y_pred, is_holiday) == pytest.approx(10.0)


def test_calculate_wmae_weights_holidays_5x():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([110.0, 110.0])
    is_holiday = np.array([True, False])
    # errores absolutos = 10 en ambos, pesos = [5, 1] -> (5*10 + 1*10) / (5+1) = 10
    assert calculate_wmae(y_true, y_pred, is_holiday) == pytest.approx(10.0)

    y_true = np.array([100.0, 100.0])
    y_pred = np.array([120.0, 110.0])
    is_holiday = np.array([True, False])
    # errores = [20, 10], pesos = [5,1] -> (5*20 + 1*10)/6 = 18.33
    assert calculate_wmae(y_true, y_pred, is_holiday) == pytest.approx(110 / 6)


def test_evaluate_returns_all_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([100.0, 200.0, 300.0, 400.0])
    is_holiday = np.array([False, True, False, True])
    report = evaluate(y_true, y_pred, is_holiday)
    assert report.wmae == pytest.approx(0.0)
    assert report.mae == pytest.approx(0.0)
    assert report.rmse == pytest.approx(0.0)
    assert report.r2 == pytest.approx(1.0)


def test_wmae_by_group_ranks_worst_first():
    df = pd.DataFrame({
        "y_true": [100, 100, 100, 100],
        "y_pred": [100, 150, 100, 100],
        "IsHoliday": [False, False, False, False],
        "Dept": [1, 2, 3, 4],
    })
    result = wmae_by_group(df, "y_true", "y_pred", "IsHoliday", "Dept", top_n=2)
    assert list(result.index[:1]) == [2]
