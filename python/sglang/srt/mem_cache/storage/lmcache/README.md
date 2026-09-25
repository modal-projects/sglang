# LMCache Connector for SGLang

This document describes how to use LMCache as KV Cache Management Backend for SGLang engine.
For more details about LMCache, please refer to: https://lmcache.ai

## Multimodal prefix reuse

The local radix cache compares full multimodal content identities. The LMCache
SGLang connector accepts token IDs without per-span content identities, so external
loads and stores stop at the last complete page before the first media token.
Text after that token also depends on the media and stays in the local cache.
Text-only requests and exact local multimodal matches retain their usual reuse.

Multimodal EAGLE bigram requests use local caching only: the external connector's
token-to-KV counts do not represent the shared bigram boundary. No configuration
change is required.

## Install LMCache

### Method 1: with pip

```bash
pip install lmcache
```

### Method 2: from source

Clone LMCache project:

```bash
git clone https://github.com/LMCache/LMCache
```

Install:

```bash
cd LMCache
pip install -e . --no-build-isolation
```


## Use LMCache

LMCache supports two transport modes. **MP (multi-process, default)** issues a single blocking retrieve over ZMQ to a standalone daemon that owns the KV store and survives SGLang restarts. **IP (in-process)** uses an embedded layerwise connector — the cache lives and dies with the SGLang process. Mode selection is currently a code-level setting in `LMCRadixCache.__init__` (`self._mode`); only MP is reachable by default.

### MP mode (default): multi-process daemon

Uses `LMCacheMPConnector`. Daemon host/port come from the LMCache YAML config (`mp_host`, `mp_port`).

Terminal 1 — start the LMCache daemon:

```bash
lmcache server \
  --host 127.0.0.1 --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

Use the bundled `example_config_mp.yaml` (or any YAML setting `mp_host` / `mp_port`):

Terminal 2 — start SGLang:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_mp.yaml
```

For full LMCache config options see https://docs.lmcache.ai/api_reference/configurations.html.

### IP mode: in-process

Uses `LMCacheLayerwiseConnector`. KV transfer happens per layer inside the SGLang process; the cache lives and dies with the server. To enable, edit `LMCRadixCache.__init__` and set `self._mode = LMCacheMode.IP`.

The LMCache config still controls chunk_size and storage; `mp_host` / `mp_port` are ignored on this path. Use the bundled `example_config_ip.yaml`:

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --enable-lmcache \
  --lmcache-config-file example_config_ip.yaml
```
