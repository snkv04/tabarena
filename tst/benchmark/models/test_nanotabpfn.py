from __future__ import annotations

import pytest


def test_nanotabpfn():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.nanotabpfn.nanotabpfn_model import NanoTabPFNModel

        FitHelper.verify_model(model_cls=NanoTabPFNModel, model_hyperparameters={})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have latable installed and checkpoints available:\n"
            f"{err}"
        )
