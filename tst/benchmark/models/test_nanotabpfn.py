from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_nanotabpfn():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.nanotabpfn.nanotabpfn_model import NanoTabPFNModel

        FitHelper.verify_model(model_cls=NanoTabPFNModel, model_hyperparameters={})
    except (ImportError, FileNotFoundError) as err:
        pytest.skip(
            f"Skipping test — ensure latable is installed and checkpoints are available:\n"
            f"{err}"
        )


def test_nanotabpfn_preprocess_category_columns():
    """_preprocess must convert category dtype columns to float32 int codes."""
    from tabarena.benchmark.models.ag.nanotabpfn.nanotabpfn_model import NanoTabPFNModel

    model = NanoTabPFNModel.__new__(NanoTabPFNModel)

    df = pd.DataFrame(
        {
            "num_col": [1.0, 2.0, 3.0],
            "cat_col": pd.Categorical(["a", "b", "a"]),
        }
    )
    assert df["cat_col"].dtype.name == "category", "cat_col must be category dtype before preprocessing"

    result = model._preprocess(df, is_train=False)
    assert result["cat_col"].dtype.name != "category", "category column should be converted"

    arr = result.to_numpy().astype(np.float32)
    assert not np.any(np.isnan(arr)), "no NaNs after converting category columns"
