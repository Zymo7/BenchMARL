#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .lstm import Lstm, LstmConfig
from .hebbian import Hebbian, HebbianConfig
from .full_hebbian import FullHebbianModel, FullHebbianConfig
from .han import HanConfig, HanModel
from .mlp import Mlp, MlpConfig
from .static_mlp import StaticMlpConfig, StaticMlpModel

classes = [
    "Hebbian",
    "HebbianConfig",
    "FullHebbianModel",
    "FullHebbianConfig",
    "HanModel",
    "HanConfig",
    "Mlp",
    "MlpConfig",
    "StaticMlpModel",
    "StaticMlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
]

model_config_registry = {
    "hebbian": HebbianConfig,
    "full_hebbian": FullHebbianConfig,
    "han": HanConfig,
    "mlp": MlpConfig,
    "static_mlp": StaticMlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
}
