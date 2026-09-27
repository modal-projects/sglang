from types import SimpleNamespace

from sglang.srt.configs.hybrid_arch import hybrid_kda_config
from sglang.srt.mem_cache import kv_cache_configurator


def test_glm5_next_hybrid_kda_config_is_available_to_kv_configurator():
    text_config = object()
    hf_config = SimpleNamespace(
        model_type="glm5_next",
        architectures=None,
        get_text_config=lambda: text_config,
    )
    model_config = SimpleNamespace(hf_config=hf_config, is_draft_model=False)

    assert hybrid_kda_config(model_config) is text_config
    assert kv_cache_configurator.hybrid_kda_config(model_config) is text_config
