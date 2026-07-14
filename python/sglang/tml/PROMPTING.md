# Inkling Prompting Guide

**NOTE:** The spec described in this document may change!

Inkling prompts are token streams made from typed message blocks. A normal block has this shape:

```text
<|message_ROLE|><|content_KIND|>payload<|end_message|>
```

Some tool-related blocks include a plain-text name between the message token and the content token:

```text
<|message_tool|>tool_name<|content_text|>payload<|end_message|>
```

A compact end-to-end prompt and response can look like this:

```text
<|message_system|><|content_text|>You are a concise assistant.<|end_message|>
<|message_system|><|content_text|>Thinking effort level: 0.8<|end_message|>
<|message_system|>tool_declare<|content_xml|>[{"description":"Get weather information","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>
<|message_user|><|content_text|>What is suitable to wear in this place today?<|end_message|>
<|message_user|><|content_image|><image input span><|end_message|>
<|message_user|><|content_audio_input|><audio input span><|audio_end|><|end_message|>
<|content_thinking|>I need the location and current weather.<|end_message|>
<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
<|message_tool|>get_weather<|content_text|>{"temperature":58,"condition":"fog"}<|end_message|>
<|content_text|>Wear a light jacket; it is cool and foggy.<|end_message|>
<|content_model_end_sampling|>
```

## Tokenizer

The Inkling tokenizer is the `o200k_base` tiktoken tokenizer with additional special tokens for message and content structure. Payload text is encoded with the base `o200k_base` tokenizer. The examples below show special tokens as literal text for readability.

## Special Tokens

The standard Inkling chat tokens in this tokenizer overlay are:

| Token | ID  | Meaning |
|-------|----:|---------|
| `<\|endoftext\|>` | 199999 | Global end-of-text token. |
| `<\|message_user\|>` | 200000 | Starts a user-authored message block. |
| `<\|message_model\|>` | 200001 | Starts a model-authored message block. |
| `<\|message_system\|>` | 200002 | Starts a system-authored message block. |
| `<\|message_tool\|>` | 200003 | Starts a tool-result message block. |
| `<\|content_text\|>` | 200004 | Starts visible text content. |
| `<\|content_image\|>` | 200005 | Starts image input content. |
| `<\|content_model_end_sampling\|>` | 200006 | Marks model end-of-sampling. This token is standalone and is not followed by <code><\|end_message\|></code>. |
| `<\|content_thinking\|>` | 200008 | Starts model reasoning content. |
| `<\|end_message\|>` | 200010 | Ends the current typed message block. |
| `<\|content_audio_input\|>` | 200020 | Starts audio input content. |
| `<\|content_tool_error\|>` | 200022 | Starts model-emitted tool-error content. |
| `<\|content_xml\|>` | 200024 | Starts structured-content payloads such as tool declarations. |
| `<\|audio_end\|>` | 200043 | Marks the end of an audio input span when present. |
| `<\|content_invoke_tool_json\|>` | 200049 | Starts a structured model tool call encoded as JSON. |
| `<\|content_invoke_tool_text\|>` | 200057 | Starts a raw text tool invocation. |

## Message Blocks

System, user, and model text messages use `content_text`:

```text
<|message_system|><|content_text|>You are a concise assistant.<|end_message|>
<|message_user|><|content_text|>What is 2 + 2?<|end_message|>
<|message_model|><|content_text|>4<|end_message|>
```

## Chat Template Shape

A text-only conversation is a concatenation of message blocks:

```text
<|message_system|><|content_text|>You are a helpful assistant.<|end_message|>
<|message_user|><|content_text|>Write a haiku about rain.<|end_message|>
```

Multiple content parts for the same conversational turn are represented as multiple Inkling blocks with the same message role and the same order:

```text
<|message_user|><|content_text|>Describe this image:<|end_message|>
<|message_user|><|content_image|><image input span><|end_message|>
<|message_user|><|content_text|>Focus on the objects in the center.<|end_message|>
```

## Multimodal Input Blocks

Inkling represents multimodal inputs as typed content blocks in the token stream plus separate modality features supplied to the model.

### Image Inputs

Each image is represented as an image content block. In the logical model input stream, the image content special token is followed by the image's ordered feature-token sequence:

```text
<|message_user|><|content_image|><image><|end_message|>
```

`<image>` denotes the preprocessed image features associated with that content block. It is not ordinary text. The Huggingface image processor converts the original image into one feature row per image token, and the Inkling vision tower embeds those feature rows for the language model.

```text
image bytes -> InklingImageProcessor -> per-token image features -> Inkling vision tower
```

Use `InklingImageProcessor` for image preprocessing:

```python
from PIL import Image

from sglang.tml.multimodal import InklingImageProcessor

# Use the checkpoint value. Current Inkling multimodal checkpoints store this at
# config.vision_config.patch_size.
image_processor = InklingImageProcessor(patch_size=config.vision_config.patch_size)

image = Image.open("example.png").convert("RGB")
features = image_processor.preprocess([image], return_tensors="pt")

vision_patches = features["vision_patches_bthwc"]
num_tokens = features["num_tokens"]
num_patches = features["num_patches"]
```

The returned `BatchFeature` contains:

```text
vision_patches_bthwc: torch.Tensor
  Shape: [sum(num_patches), 2, patch_size, patch_size, 3]
  Dtype: bfloat16

num_patches: list[int]
  Number of patch feature rows produced for each input image.

num_tokens: list[int]
  Number of Inkling image tokens produced for each input image.
```

For Inkling models, `num_tokens[i] == num_patches[i]`: each image patch produces one image token. The image processor output is the per-token image feature input for the Inkling vision tower. During inference, the model applies its hMLP image encoder independently to each image token feature and uses the resulting embeddings at the image token positions in the model input stream.

### Audio Inputs

Each audio clip is represented as an audio content block. In the logical model input stream, the audio content special token is followed by the audio's ordered feature-token sequence and then the audio-end special token:

```text
<|message_user|><|content_audio_input|><audio><|audio_end|><|end_message|>
```

`<audio>` denotes the preprocessed audio features associated with that content block. The Huggingface audio feature extractor converts the original audio into one dMel feature row per audio token, and the Inkling audio embedding table looks up those feature rows for the language model.

```text
audio bytes -> InklingAudioFeatureExtractor -> per-token dMel features -> Inkling audio embedding
```

Use `InklingAudioFeatureExtractor` for audio preprocessing:

```python
from pathlib import Path

from sglang.tml.multimodal import InklingAudioFeatureExtractor

# Use the checkpoint values. Current Inkling multimodal checkpoints store these at
# config.audio_config.
audio_feature_extractor = InklingAudioFeatureExtractor(
    params={
        "n_mels": config.audio_config.n_mel_bins,
        "num_dmel_bins": config.audio_config.mel_vocab_size,
        "dmel_min_value": config.audio_config.dmel_min_value,
        "dmel_max_value": config.audio_config.dmel_max_value,
    }
)

wav_bytes = Path("example.wav").read_bytes()
features = audio_feature_extractor([wav_bytes])

dmel_bins = features["dmel_bins"]
num_audio_tokens = features["num_audio_tokens"]
```

The returned `BatchFeature` contains:

```text
dmel_bins: list[torch.Tensor]
  One tensor per audio clip.
  Shape per tensor: [num_audio_tokens, n_mel_bins]
  Dtype: float32, with values representing discrete dMel bin ids.

num_audio_tokens: list[int]
  Number of Inkling audio tokens produced for each audio clip.
```

The extractor decodes audio, downmixes to mono, resamples to the configured sample rate, computes mel features, quantizes them into discrete dMel bins, and returns one dMel row per audio token. The expected token duration is 50 ms, so a clip contributes 20 audio tokens per second.

The audio extractor output is the per-token audio feature. During inference, the model embeds each audio token's dMel bin ids with a learned audio embedding table, combines the mel-bin embeddings for that token, and uses the resulting embedding at the audio token position in the model input stream.

## Reasoning

Reasoning content is represented with `content_thinking`. It is separate from visible text content:

```text
<|content_thinking|>Calculate 2 + 2.<|end_message|>
<|content_text|>4<|end_message|>
```

Reasoning can also precede a tool call:

```text
<|content_thinking|>I need current weather data.<|end_message|>
<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
```

### Reasoning Effort

Inkling supports a scalar reasoning-effort prompt control. It is represented as a normal system text message using the prefix `Thinking effort level:` followed by the requested effort value:

```text
<|message_system|><|content_text|>Thinking effort level: 0.8<|end_message|>
```

The effort value is a number between `0.0` and `1.0`, where larger values request more reasoning. Values are commonly rendered with up to two decimal places.

## Model Response Format

A Inkling model response is a sequence of typed content blocks. The most common response content kinds are text, reasoning, structured tool calls, tool errors, and end-of-sampling.

### Text Content

```text
<|content_text|>The answer is 4.<|end_message|>
```

The visible assistant content is the payload between `<|content_text|>` and `<|end_message|>`. Multiple text blocks represent multiple text segments in sequence.

### Reasoning Content

Reasoning content appears in `content_thinking` blocks, as described in the reasoning section.

### Structured Tool Calls

The structured tool-call response format is:

```text
<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
```

The payload is a JSON object with string field `name` and object field `args`. Multiple tool-call blocks represent multiple tool calls.

### Tool Error Content

Tool-error content uses `content_tool_error`:

```text
<|content_tool_error|>Error text here<|end_message|>
```

This content kind represents tool-error text emitted by the model.

### End Sampling

End-of-sampling is represented by a standalone content token:

```text
<|content_model_end_sampling|>
```

It is not followed by `<|end_message|>`.

## Tool Calling

Tool-related Inkling uses names as plain text between the message token and the content token.

### Tool Declarations

Available tools are represented as a system message named `tool_declare` with a JSON payload under `content_xml`:

```text
<|message_system|>tool_declare<|content_xml|>[{"description":"Get weather information","name":"get_weather","parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>
```

The payload is a JSON array of tool specs. Each tool spec contains:

```json
{
  "description": "Get weather information",
  "name": "get_weather",
  "parameters": {
    "properties": {
      "city": {
        "type": "string"
      }
    },
    "required": ["city"],
    "type": "object"
  },
  "type": "function"
}
```

### Tool Calls

Structured tool calls use `content_invoke_tool_json`:

```text
<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
```

The same content block is wrapped in `message_model` when it appears inside conversation history:

```text
<|message_model|><|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"SF"}}<|end_message|>
```

The JSON object has this shape:

```json
{
  "name": "tool_name",
  "args": {
    "arg_name": "arg_value"
  }
}
```

`name` is the tool name. `args` is a JSON object containing the tool arguments.

### Tool Results

A tool result is represented as a `message_tool` block. The tool name is plain text immediately after `<|message_tool|>`:

```text
<|message_tool|>get_weather<|content_text|>{"temperature":72}<|end_message|>
```

The token text carries the tool name and result content. It does not carry a tool call ID.

### Raw Tool Text

Inkling also has a raw text tool invocation form:

```text
<|content_invoke_tool_text|>search for weather in SF<|end_message|>
```

This form is distinct from structured JSON tool calling because it has no `name`/`args` object.