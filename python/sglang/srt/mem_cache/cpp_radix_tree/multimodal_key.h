#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"

namespace radix_tree_v2 {

using mm_span_input_t = std::vector<std::tuple<std::size_t, std::size_t, std::string, std::uint64_t>>;

struct MultimodalSpan {
  std::size_t start;
  std::size_t end;
  std::array<token_t, 8> digest;
  std::uint64_t offset;
};

using mm_spans_t = std::vector<MultimodalSpan>;
using mm_span_slice = std::span<const MultimodalSpan>;

inline std::array<token_t, 8> parse_mm_identity(const std::string& identity) {
  if (identity.size() != 71 || !identity.starts_with("sha256:"))
    throw std::invalid_argument("Multimodal identity must be a canonical SHA-256 digest");
  std::array<token_t, 8> digest;
  for (std::size_t i = 0; i < digest.size(); ++i) {
    std::uint32_t word = 0;
    for (std::size_t j = 0; j < 8; ++j) {
      const auto c = identity[7 + i * 8 + j];
      if (!(('0' <= c && c <= '9') || ('a' <= c && c <= 'f')))
        throw std::invalid_argument("Multimodal identity must be a canonical SHA-256 digest");
      word = (word << 4) | (c <= '9' ? c - '0' : c - 'a' + 10);
    }
    digest[i] = std::bit_cast<token_t>(word);
  }
  return digest;
}

inline mm_spans_t parse_mm_spans(const mm_span_input_t& input, std::size_t key_size) {
  mm_spans_t spans;
  spans.reserve(input.size());
  for (const auto& [start, end, identity, offset] : input) {
    if (start >= end || end > key_size || (!spans.empty() && start < spans.back().end))
      throw std::invalid_argument("Multimodal spans must be ordered, nonempty, and within the key");
    if (end - start - 1 > std::numeric_limits<std::uint64_t>::max() - offset)
      throw std::invalid_argument("Multimodal offsets must fit in uint64");
    auto digest = parse_mm_identity(identity);
    if (!spans.empty()) {
      auto& prior = spans.back();
      const auto prior_length = prior.end - prior.start;
      if (prior.end == start && prior.digest == digest &&
          prior_length <= std::numeric_limits<std::uint64_t>::max() - prior.offset &&
          prior.offset + prior_length == offset) {
        prior.end = end;
        continue;
      }
    }
    spans.push_back({start, end, digest, offset});
  }
  return spans;
}

inline mm_spans_t slice_mm_spans(mm_span_slice spans, std::size_t start, std::size_t end) {
  mm_spans_t result;
  for (const auto& span : spans) {
    if (span.end <= start) continue;
    if (span.start >= end) break;
    const auto first = std::max(start, span.start);
    result.push_back({first - start, std::min(end, span.end) - start, span.digest, span.offset + first - span.start});
  }
  return result;
}

inline void append_u64(token_vec_t& output, std::uint64_t value) {
  output.push_back(std::bit_cast<token_t>(static_cast<std::uint32_t>(value)));
  output.push_back(std::bit_cast<token_t>(static_cast<std::uint32_t>(value >> 32)));
}

inline void make_page_key(token_vec_t& output, token_slice page, mm_span_slice spans, std::size_t start = 0) {
  output.assign(page.begin(), page.end());
  for (const auto& span : spans) {
    if (span.end <= start) continue;
    if (span.start >= start + page.size()) break;
    const auto first = std::max(start, span.start);
    const auto end = std::min(start + page.size(), span.end);
    std::fill(output.begin() + first - start, output.begin() + end - start, 0);
    // Text keys contain exactly one page. Media keys append the complete span
    // identity, so equality cannot confuse their normalized routing tokens with text.
    append_u64(output, first - start);
    append_u64(output, end - start);
    output.insert(output.end(), span.digest.begin(), span.digest.end());
    append_u64(output, span.offset + first - span.start);
  }
}

inline std::pair<const MultimodalSpan*, std::size_t>
mm_segment(mm_span_slice spans, std::size_t& index, std::size_t position, std::size_t end) {
  while (index < spans.size() && spans[index].end <= position)
    ++index;
  if (index == spans.size()) return {nullptr, end};
  const auto& span = spans[index];
  if (position < span.start) return {nullptr, std::min(end, span.start)};
  return {&span, std::min(end, span.end)};
}

inline std::size_t matching_token_count(
    token_slice a,
    mm_span_slice a_spans,
    std::size_t a_start,
    token_slice b,
    mm_span_slice b_spans,
    std::size_t offset) {
  if (a_spans.empty() && b_spans.empty()) {
    const auto left = a.subspan(offset);
    const auto [it_a, it_b] = std::ranges::mismatch(left, b.subspan(offset));
    return it_a - left.begin();
  }
  const auto length = std::min(a.size(), b.size());
  std::size_t position = offset, a_index = 0, b_index = 0;
  while (position < length) {
    const auto [left, left_end] = mm_segment(a_spans, a_index, a_start + position, a_start + length);
    const auto [right, right_end] = mm_segment(b_spans, b_index, position, length);
    const auto end = std::min(left_end - a_start, right_end);
    if ((left == nullptr) != (right == nullptr)) break;
    if (left != nullptr) {
      if (left->digest != right->digest ||
          left->offset + a_start + position - left->start != right->offset + position - right->start)
        break;
    } else {
      const auto left_tokens = a.subspan(position, end - position);
      const auto [it_a, it_b] = std::ranges::mismatch(left_tokens, b.subspan(position, end - position));
      const auto matched = static_cast<std::size_t>(it_a - left_tokens.begin());
      if (matched != left_tokens.size()) return position + matched - offset;
    }
    position = end;
  }
  return position - offset;
}

}  // namespace radix_tree_v2
