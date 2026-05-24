from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
import os

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class NanoTabPFNModel(AbstractTorchModel):
    """NanoTabPFN: a small in-context learning foundation model for tabular data.

    Trained with SIGReg regularisation on a synthetic prior; at inference time
    it prepends training rows as context and predicts test rows directly —
    no gradient updates on user data.

    Codebase: ~/Desktop/latable (local research repo)
    """

    ag_key = "NANOTABPFN"
    ag_name = "NanoTabPFN"
    ag_priority = 65

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        import torch

        logger.warning(f"NanoTabPFN fitting with num_cpus={num_cpus}, num_gpus={num_gpus}")
        available_num_gpus = ResourceManager.get_gpu_count_torch(cuda_only=True)
        if num_gpus > available_num_gpus:
            raise AssertionError(
                f"Fit specified to use {num_gpus} GPU, but only {available_num_gpus} "
                "CUDA GPUs are available. Please activate CUDA or switch to CPU usage.",
            )
        device = "cuda" if num_gpus != 0 else "cpu"

        hps = self._get_model_params()
        lam = hps.pop("lam", 0.0)
        ckpt_seed = hps.pop("ckpt_seed", 42)
        ckpt_step = hps.pop("ckpt_step", 2500)
        latable_dir = str(Path(hps.pop("latable_dir", "~/Desktop/latable")).expanduser())
        logger.log(15, f"NanoTabPFN checkpoint params: lam={lam:g}, ckpt_seed={ckpt_seed}, ckpt_step={ckpt_step}, latable_dir={latable_dir}")

        # Load latable files directly via importlib to avoid triggering
        # src/__init__.py, which eagerly imports stable_pretraining (not
        # available outside the latable venv).
        import importlib.util
        import types

        def _load_latable_module(rel_path: str, full_name: str):
            spec = importlib.util.spec_from_file_location(
                full_name, str(Path(latable_dir) / rel_path)
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = mod
            spec.loader.exec_module(mod)
            return mod

        # Register stub packages so cross-imports between latable files
        # resolve without triggering src/__init__.py.
        for pkg_name, sub_dir in [
            ("src", "src"),
            ("src.encoders", "src/encoders"),
            ("src.supervised", "src/supervised"),
        ]:
            if pkg_name not in sys.modules:
                stub = types.ModuleType(pkg_name)
                stub.__path__ = [str(Path(latable_dir) / sub_dir)]
                stub.__package__ = pkg_name
                sys.modules[pkg_name] = stub

        # Loads the definitions of these classes
        _enc_mod = _load_latable_module("src/encoders/nanotabpfn.py", "src.encoders.nanotabpfn")
        _NanoTabPFNBackbone = _enc_mod.NanoTabPFNModel
        _clf_mod = _load_latable_module("src/supervised/nanotabpfn_classifier.py", "src.supervised.nanotabpfn_classifier")
        NanoTabPFNClassifier = _clf_mod.NanoTabPFNClassifier

        # Hardcoded directory for checkpoints
        ckpt_path = (
            Path(f"/cs/data/people/{os.getenv('USER')}/latable_cache/supervised_lambda_grid/classification/with_projector")
            / f"cache_lc{lam:g}_lr{lam:g}"
            / f"seed{ckpt_seed}_ckpts"
            / f"step{ckpt_step}.pt"
        )
        logger.log(15, f"NanoTabPFN loading checkpoint: {ckpt_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"NanoTabPFN checkpoint not found: {ckpt_path}\n"
                "Check that latable_dir is correct and that the checkpoint exists."
            )

        # Builds a new backbone and classifier
        backbone = _NanoTabPFNBackbone(
            embedding_size=96,
            num_attention_heads=4,
            mlp_hidden_size=192,
            num_layers=3,
            num_outputs=2,
            sigreg_layer=0,
            use_projector=False,
        )
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        backbone.eval()
        classifier = NanoTabPFNClassifier(model=backbone, device=device)

        # Fits classifier to train rows
        X = self.preprocess(X, y=y, is_train=True)
        classifier.fit(
            X.to_numpy().astype(np.float32),
            y.to_numpy().astype(np.int64),
        )
        self.model = classifier

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, is_train=is_train, **kwargs)
        cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
        if cat_cols:
            X = X.copy()
            for col in cat_cols:
                X[col] = X[col].cat.codes.astype(np.float32)
        return X

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:  # noqa: ARG002
        proba = self.model.predict_proba(X.to_numpy().astype(np.float32))
        return proba[:, 1]  # AutoGluon expects 1D positive-class probabilities for binary

    def _set_default_params(self):
        default_params = {
            "lam": 0.0,
            "ckpt_seed": 42,
            "ckpt_step": 2500,
            "latable_dir": "~/Desktop/latable",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary"]

    def get_device(self) -> str:
        return str(self.model.device)

    def _set_device(self, device: str):
        self.model.model.to(device)
        self.model.device = device

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {
            "num_cpus": 1,
            "num_gpus": 0.5 if is_gpu_available else 0,
        }

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        default_ag_args_ensemble.update({"fold_fitting_strategy": "sequential_local"})
        return default_ag_args_ensemble

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": False}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
