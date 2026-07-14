#!/usr/bin/env bash

INKLING_MODEL_PATH="${INKLING_MODEL_PATH:?missing}"
INKLING_TP_SIZE="${INKLING_TP_SIZE:-4}"

SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
python -m sglang.launch_server \
  --model-path "${INKLING_MODEL_PATH}" \
  --served-model-name "inkling-model" \
  --tensor-parallel-size "${INKLING_TP_SIZE}" \
  --mamba-scheduler-strategy "extra_buffer" \
  --disable-custom-all-reduce \
  --attention-backend fa4 \
  --enable-multimodal \
  --reasoning-parser inkling \
  --tool-call-parser inkling

# curl http://127.0.0.1:30000/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "model": "inkling-model",
#     "messages": [
#       {"role": "user", "content": "Write one sentence explaining what tensor parallelism is."}
#     ],
#     "max_tokens": 256,
#     "temperature": 0.2
#   }'

# curl http://127.0.0.1:30000/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "model": "inkling-model",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           {
#             "type": "text",
#             "text": "Describe this image. Identify the animal if you can."
#           },
#           {
#             "type": "image_url",
#             "image_url": {
#               "url": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
#             }
#           }
#         ]
#       }
#     ],
#     "max_tokens": 256,
#     "temperature": 0.2
#   }'
