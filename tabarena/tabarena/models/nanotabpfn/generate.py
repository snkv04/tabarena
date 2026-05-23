from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.ag.nanotabpfn.nanotabpfn_model import NanoTabPFNModel
from tabarena.utils.config_utils import ConfigGenerator

gen_nanotabpfn = ConfigGenerator(
    model_cls=NanoTabPFNModel,
    search_space={
        "lam": Categorical(0.0, 0.01, 0.1),
    },
    manual_configs=[
        {"lam": 0.0},
        {"lam": 0.01},
        {"lam": 0.1},
    ],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_nanotabpfn.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
