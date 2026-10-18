// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

#include "music3_model.h"

int main(int argc, char **argv) {
  try {
    std::filesystem::path directory, requests, output;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " needs a value");
      } else if (flag == "--model-path") {
        directory = argv[++i];
      } else if (flag == "--requests") {
        requests = argv[++i];
      } else if (flag == "--output-directory") {
        output = argv[++i];
      } else {
        throw std::invalid_argument("unknown argument " + flag);
      }
    }
    if (directory.empty() || requests.empty() || output.empty()) {
      throw std::invalid_argument("usage: music3_synthesize --model-path DIR "
                                  "--requests JSONL --output-directory DIR");
    } else {
    }
    mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
    music3::Model model(directory);
    std::ifstream input(requests);
    if (!input) {
      throw std::runtime_error("cannot read " + requests.string());
    } else {
    }
    std::filesystem::create_directories(output);
    const std::atomic<bool> cancelled{false};
    std::string line;
    while (std::getline(input, line)) {
      auto body = nlohmann::json::parse(line);
      const std::string identifier = body.at("id");
      if (!std::regex_match(identifier, std::regex("[A-Za-z0-9_-]{1,128}"))) {
        throw std::invalid_argument(
            "id must contain only letters, digits, hyphens and underscores");
      } else {
      }
      const auto request = music3::ParseRequest(
          body, model.tokenizer(), "minimax_music3", model.context_tokens());
      const auto started = std::chrono::steady_clock::now();
      const auto result = model.Synthesize(request, cancelled);
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      const auto path = output / (identifier + ".wav");
      const auto bytes = music3::EncodeAudio(result.samples, false);
      std::ofstream audio(path, std::ios::binary);
      audio.write(bytes.data(), bytes.size());
      audio.close();
      if (!audio) {
        throw std::runtime_error("cannot write " + path.string());
      } else {
      }
      std::cout << nlohmann::json({{"id", identifier},
                                   {"file", path.string()},
                                   {"generated_frames", result.frames},
                                   {"sample_count", result.samples.size() / 2},
                                   {"sample_rate", music3::kSampleRate},
                                   {"elapsed_s", elapsed}})
                       .dump()
                << std::endl;
      mlx::core::clear_cache();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "music3_synthesize: " << error.what() << "\n";
    return 1;
  }
}
