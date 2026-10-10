// SPDX-License-Identifier: Apache-2.0
// Pieces of Voxt's Swift model ports that other native models reproduce: the
// Slaney mel filter bank at any FFT size and Swift's whitespace trimming.
#pragma once

#include <string>
#include <vector>

namespace swift_port {

// Slaney-scale, Slaney-normalized filters [fft_size / 2 + 1, mel_bin_count]
// for 16 kHz audio, built in float32 in the order the Swift port builds them.
std::vector<float> SlaneyMelFilterBank(int fft_size, int mel_bin_count);

// Swift's trimmingCharacters(in: .whitespaces) on valid UTF-8, or with
// newlines, (in: .whitespacesAndNewlines).
std::string TrimWhitespace(const std::string &text, bool newlines);

} // namespace swift_port
