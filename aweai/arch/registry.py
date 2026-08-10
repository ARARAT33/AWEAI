from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

ARCH_FAMILIES: List[str] = [
    "Transformer",
    "MoE",
    "CNN",
    "RNN",
    "GNN",
    "Diffusion",
    "NextGen",
    "Compound",
    "Hybrid",
    "NAS",
]

@dataclass
class ArchEntry:
    name: str
    family: str
    builder: Callable[..., nn.Module]
    params_schema: Dict[str, Any]
    paper_ref: str

class ArchitectureRegistry:
    _instance: Optional[ArchitectureRegistry] = None
    _registry: Dict[str, ArchEntry] = {}

    def __new__(cls) -> ArchitectureRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self) -> None:
        def make_transformer(d_model: int, nhead: int, num_layers: int) -> nn.Module:
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead),
                num_layers=num_layers,
            )
        self.register("Transformer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Vaswani et al., 2017")
        self.register("TransformerXL", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Dai et al., 2019")
        self.register("Reformer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Kitaev et al., 2020")
        self.register("Performer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Choromanski et al., 2020")
        self.register("Linformer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Wang et al., 2020")
        self.register("BigBird", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Zaheer et al., 2020")
        self.register("Longformer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Beltagy et al., 2020")
        self.register("GPT-2", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Radford et al., 2019")
        self.register("GPT-3", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Brown et al., 2020")
        self.register("BERT", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Devlin et al., 2018")
        self.register("RoBERTa", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Liu et al., 2019")
        self.register("T5", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Raffel et al., 2020")
        self.register("ALBERT", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Lan et al., 2019")
        self.register("XLNet", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Yang et al., 2019")
        self.register("ELECTRA", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Clark et al., 2020")
        self.register("DistilBERT", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Sanh et al., 2019")
        self.register("MobileBERT", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Sun et al., 2020")
        self.register("DeBERTa", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "He et al., 2020")
        self.register("LongT5", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Guo et al., 2021")
        self.register("Pegasus", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Zhang et al., 2020")
        self.register("BART", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Lewis et al., 2019")
        self.register("Swin Transformer", "Transformer", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Liu et al., 2021")
        self.register("ConvNeXt", "CNN", make_transformer,
                      {"d_model": int, "nhead": int, "num_layers": int},
                      "Liu et al., 2022")

        def make_cnn(channels: int, depth: int) -> nn.Module:
            return nn.Sequential(*[nn.Conv2d(channels, channels, 3, padding=1) for _ in range(depth)])
        self.register("ResNet", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "He et al., 2016")
        self.register("ResNeXt", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Xie et al., 2017")
        self.register("DenseNet", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Huang et al., 2017")
        self.register("MobileNetV1", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Howard et al., 2017")
        self.register("MobileNetV2", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Sandler et al., 2018")
        self.register("MobileNetV3", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Howard et al., 2019")
        self.register("EfficientNet", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Tan & Le, 2019")
        self.register("RegNet", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Radosavovic et al., 2020")
        self.register("ConvNeXt", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Liu et al., 2022")
        self.register("NFNet", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Brock et al., 2021")
        self.register("MaxViT", "CNN", make_cnn,
                      {"channels": int, "depth": int},
                      "Tu et al., 2022")

        def make_rnn(input_size: int, hidden_size: int, num_layers: int) -> nn.Module:
            return nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.register("LSTM", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Hochreiter & Schmidhuber, 1997")
        self.register("GRU", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Cho et al., 2014")
        self.register("SRU", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Lei et al., 2018")
        self.register("QRNN", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Bradbury et al., 2017")
        self.register("AWD-LSTM", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Merity et al., 2017")
        self.register("IndRNN", "RNN", make_rnn,
                      {"input_size": int, "hidden_size": int, "num_layers": int},
                      "Li et al., 2018")

        def make_gnn(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
            return nn.Linear(in_dim, out_dim)
        self.register("GCN", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Kipf & Welling, 2017")
        self.register("GAT", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Veličković et al., 2018")
        self.register("GraphSAGE", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Hamilton et al., 2017")
        self.register("GIN", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Xu et al., 2019")
        self.register("Transformer-GNN", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Yun et al., 2020")
        self.register("NodeFormer", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Wu et al., 2022")
        self.register("GraphGPS", "GNN", make_gnn,
                      {"in_dim": int, "hidden_dim": int, "out_dim": int},
                      "Rampášek et al., 2022")

        def make_diffusion(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.register("DDPM", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Ho et al., 2020")
        self.register("DDIM", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Song et al., 2021")
        self.register("ScoreSDE", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Song et al., 2020")
        self.register("ConsistencyModel", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Song et al., 2023")
        self.register("FlowMatching", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Lipman et al., 2023")
        self.register("VDM", "Diffusion", make_diffusion,
                      {"in_channels": int, "out_channels": int},
                      "Kingma & Gao, 2023")

        def make_nextgen(d_model: int, nhead: int) -> nn.Module:
            return nn.Linear(d_model, d_model)
        self.register("RWKV", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Peng et al., 2023")
        self.register("Mamba", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Gu & Dao, 2023")
        self.register("RetNet", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Sun et al., 2023")
        self.register("Griffin", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "De et al., 2024")
        self.register("UniversalTransformer", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Dehghani et al., 2019")
        self.register("HybridMoETransformer", "Hybrid", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Fedus et al., 2022")
        self.register("SwitchTransformer", "MoE", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Fedus et al., 2021")
        self.register("GLaM", "MoE", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Du et al., 2021")
        self.register("Mixtral", "MoE", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Jiang et al., 2024")
        self.register("Nystromformer", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Xiong et al., 2021")
        self.register("CosFormer", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Qin et al., 2022")
        self.register("LinearAttention", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Katharopoulos et al., 2020")
        self.register("DeltaNet", "RNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Schlag et al., 2021")
        self.register("Temporal Fusion Transformer", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Lim et al., 2021")
        self.register("VisionTransformer", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Dosovitskiy et al., 2021")
        self.register("DeiT", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Touvron et al., 2021")
        self.register("SwinV2", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Liu et al., 2022")
        self.register("BEiT", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Bao et al., 2022")
        self.register("MAE", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "He et al., 2022")
        self.register("SAM", "CNN", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Kirillov et al., 2023")
        self.register("CLIP", "Hybrid", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Radford et al., 2021")
        self.register("DALL-E", "Hybrid", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Ramesh et al., 2021")
        self.register("Imagen", "Diffusion", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Saharia et al., 2022")
        self.register("StableDiffusion", "Diffusion", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Rombach et al., 2022")
        self.register("Wav2Vec2", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Baevski et al., 2020")
        self.register("HuBERT", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Hsu et al., 2021")
        self.register("Whisper", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Radford et al., 2023")
        self.register("LLaMA", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Touvron et al., 2023")
        self.register("PaLM", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Chowdhery et al., 2022")
        self.register("Chinchilla", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Hoffmann et al., 2022")
        self.register("Gemma", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Gemini Team, 2024")
        self.register("Mistral", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Jiang et al., 2023")
        self.register("Qwen", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Bai et al., 2023")
        self.register("DeepSeek", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Bi et al., 2024")
        self.register("OLMo", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Groeneveld et al., 2024")
        self.register("Falcon", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Almazrouei et al., 2023")
        self.register("MPT", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "MosaicML, 2023")
        self.register("OPT", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Zhang et al., 2022")
        self.register("Bloom", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Scao et al., 2022")
        self.register("Vicuna", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Chiang et al., 2023")
        self.register("ALaMo", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Almazrouei et al., 2024")
        self.register("CodeLlama", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Rozière et al., 2023")
        self.register("StarCoder", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Lozhkov et al., 2024")
        self.register("Codex", "Transformer", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "OpenAI, 2022")
        self.register("TabNet", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Arik & Pfister, 2021")
        self.register("FTTransformer", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Gorishniy et al., 2023")
        self.register("TabPFN", "NextGen", make_nextgen,
                      {"d_model": int, "nhead": int},
                      "Hollmann et al., 2023")

    def register(self, name: str, family: str, builder: Callable[..., nn.Module], params_schema: Dict[str, Any], paper_ref: str) -> None:
        self._registry[name] = ArchEntry(name, family, builder, params_schema, paper_ref)

    def get(self, name: str) -> ArchEntry:
        if name not in self._registry:
            raise KeyError(f"Architecture {name} not found in registry")
        return self._registry[name]

    def list_all(self) -> List[str]:
        return sorted(self._registry.keys())

    def search(self, query: str) -> List[str]:
        q = query.lower()
        return [k for k in self._registry if q in k.lower()]

    def families(self) -> Set[str]:
        return {v.family for v in self._registry.values()}

    def by_family(self, family: str) -> List[str]:
        return sorted(k for k, v in self._registry.items() if v.family == family)

    def count(self) -> int:
        return len(self._registry)

    def describe(self, name: str) -> str:
        entry = self.get(name)
        return f"{entry.name} ({entry.family}): {entry.paper_ref}"
