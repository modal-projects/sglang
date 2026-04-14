Config:

  - HF=H200, SGLang=H200
  - attention_backend=flashinfer
  - deterministic inference enabled
  - disable_cuda_graph=1
  - mem_fraction_static=0.7
  - memory_mb=131072

  base_* = HF base vs SGLang base
  
  delta_* = adapter effect mismatch: (HF_merged - HF_base) vs (SG_merged - SG_base)
  
  merged_* = HF merged vs SGLang merged
  
  completion_mismatch_count = free-run greedy mismatches from completion_mismatches

  Completion-Side Teacher-Forced
  | Subset | base mean | base max | delta mean | delta max | merged mean | merged max | free-run mismatches |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | all | 0.01778 | 0.14329 | 0.06889 | 0.74345 | 0.06264 | 0.70602 | 3 |
  | routed_experts | 0.01442 | 0.10822 | 0.04391 | 1.08796 | 0.04032 | 1.01565 | 1 |
  | attention | 0.02052 | 0.17337 | 0.03500 | 0.32846 | 0.02779 | 0.22256 | 2 |
  | linear_attn | 0.02042 | 0.19477 | 0.03983 | 0.47213 | 0.03048 | 0.37163 | 3 |
  | self_attn | 0.01850 | 0.21733 | 0.03131 | 0.32070 | 0.02368 | 0.34071 | 2 |
  | shared_expert | 0.01533 | 0.15767 | 0.02111 | 0.19920 | 0.01763 | 0.12688 | 1 |
  | lm_head | 0.01442 | 0.10822 | 0.01196 | 0.15304 | 0.01911 | 0.16732 | 2 |
  | routed_w1 | 0.01442 | 0.10822 | 0.03000 | 0.75210 | 0.02691 | 0.67978 | 2 |
  | routed_w2 | 0.01442 | 0.10822 | 0.01752 | 0.15437 | 0.01597 | 0.16167 | 2 |
  | routed_w3 | 0.01442 | 0.10822 | 0.02965 | 0.62159 | 0.02769 | 0.54927 | 2 |

  Prefill Teacher-Forced
  | Subset | base mean | base max | delta mean | delta max | merged mean | merged max |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | all | 0.05413 | 0.38314 | 0.12894 | 0.54822 | 0.10842 | 0.60853 |
  | routed_experts | 0.05413 | 0.38314 | 0.10364 | 0.43384 | 0.11043 | 0.54448 |
  | attention | 0.05413 | 0.38314 | 0.09207 | 0.76179 | 0.06358 | 0.37865 |
  | linear_attn | 0.05100 | 0.38314 | 0.08713 | 0.55132 | 0.06523 | 0.49697 |
  | self_attn | 0.05413 | 0.38314 | 0.08405 | 0.34869 | 0.06122 | 0.26206 |
  | shared_expert | 0.05413 | 0.38314 | 0.06994 | 0.28438 | 0.05256 | 0.31816 |
  | lm_head | 0.05413 | 0.38314 | 0.02260 | 0.15940 | 0.05809 | 0.34629 |
  | routed_w1 | 0.05413 | 0.38314 | 0.06321 | 0.30738 | 0.07531 | 0.40334 |
  | routed_w2 | 0.05413 | 0.38314 | 0.05803 | 0.31734 | 0.06766 | 0.36055 |
  | routed_w3 | 0.05413 | 0.38314 | 0.07613 | 0.30193 | 0.08462 | 0.46025 |

  Readout

  - The loader-specific regression is fixed. The Modal unit harness is green again: scripts/modal_validate_lora_merge_loader.py, app ap-
    MRYnzHsLMSbuCwrgmlHbma.
  - The broad Qwen result did not improve from that fix. The headline all numbers are still delta mean/max = 0.06889 / 0.74345 on completion and 0.12894 /
    0.54822 on prefill.
  - The remaining gap is not just means. There are large outliers:
      - routed_experts completion delta max = 1.08796
      - attention prefill delta max = 0.76179
      - routed_w1 and routed_w3 still have large completion spikes
  - routed_experts is still the biggest single family by mean, and also the spikiest by completion max.
  - attention is still the biggest non-routed family; inside it, linear_attn is a bit worse on completion mean/max, while self_attn is a bit lower on mean but
    still material.
  - shared_expert is nontrivial.
  - lm_head is small and not the main problem.
  - Free-run mismatch count does not track teacher-forced severity tightly. routed_experts only has 1 free-run mismatch here, but still the worst completion-
    side delta max.
