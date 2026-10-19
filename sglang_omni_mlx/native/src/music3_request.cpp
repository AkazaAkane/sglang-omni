// SPDX-License-Identifier: Apache-2.0
#include "music3_model.h"

#include <CoreFoundation/CoreFoundation.h>

#include <cmath>
#include <limits>
#include <regex>
#include <sstream>

namespace music3 {

std::string BuildPrompt(const std::string &caption, const std::string &lyrics) {
  static const std::regex special(R"(<\|([^|]*)\|>)");
  std::string cleaned;
  size_t offset = 0;
  for (std::sregex_iterator match(caption.begin(), caption.end(), special), end;
       match != end; ++match) {
    cleaned += caption.substr(offset, match->position() - offset);
    std::istringstream words((*match)[1].str());
    std::string first, rest;
    words >> first;
    std::getline(words >> std::ws, rest);
    const size_t last_word = rest.find_last_not_of(" \t\r\n\v\f");
    rest = last_word == std::string::npos ? "" : rest.substr(0, last_word + 1);
    cleaned += first + (rest.empty() ? "" : " is " + rest);
    offset = match->position() + match->length();
  }
  cleaned += caption.substr(offset);
  std::istringstream lines(cleaned);
  cleaned.clear();
  std::string line;
  bool first_line = true;
  while (std::getline(lines, line)) {
    line = std::regex_replace(line, std::regex(R"(^\s{0,3}#{1,6}\s+)"), "");
    line = std::regex_replace(line, std::regex(R"(^\s*[*+-]\s+)"), "");
    while (line.find("**") != std::string::npos) {
      const std::string replaced =
          std::regex_replace(line, std::regex(R"(\*\*([^*]+)\*\*)"), "$1");
      if (replaced == line) {
        break;
      } else {
        line = replaced;
      }
    }
    line = std::regex_replace(
        line, std::regex(R"((^|[^*])\*([^*\n]+)\*(?!\*))"), "$1$2");
    const size_t last = line.find_last_not_of(" \t\r\v\f");
    line = last == std::string::npos ? "" : line.substr(0, last + 1);
    if (std::regex_match(line, std::regex(R"(\s*[-*_]{3,}\s*)"))) {
      line.clear();
    } else {
    }
    cleaned += (first_line ? "" : "\n") + line;
    first_line = false;
  }
  cleaned = std::regex_replace(cleaned, std::regex("• |    "), "");
  cleaned = std::regex_replace(cleaned, std::regex("\n{2,}"), "\n");
  std::istringstream lyric_lines(lyrics);
  std::string normalized;
  first_line = true;
  while (std::getline(lyric_lines, line)) {
    std::smatch tags;
    if (std::regex_search(line, tags,
                          std::regex(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))"))) {
      line = tags[1].str();
      line.erase(line.find_last_not_of(" \t") + 1);
    } else {
    }
    normalized += (first_line ? "" : "\n") + line;
    first_line = false;
  }
  if (!lyrics.empty() && lyrics.back() == '\n') {
    normalized += '\n';
  } else {
  }
  normalized = std::regex_replace(normalized, std::regex(R"(\] )"), "]\n");
  normalized = std::regex_replace(normalized, std::regex(R"( \[)"), "\n[");
  normalized = std::regex_replace(normalized, std::regex(R"( \^ )"), "\n");
  static const std::regex tag(R"(\[([^\]]+)\])");
  std::string lowered;
  offset = 0;
  for (std::sregex_iterator match(normalized.begin(), normalized.end(), tag),
       end;
       match != end; ++match) {
    lowered += normalized.substr(offset, match->position() - offset);
    const std::string tag_text = (*match)[1].str();
    CFStringRef source = CFStringCreateWithBytes(
        kCFAllocatorDefault, reinterpret_cast<const UInt8 *>(tag_text.data()),
        static_cast<CFIndex>(tag_text.size()), kCFStringEncodingUTF8, false);
    if (source == nullptr) {
      throw std::invalid_argument("section tags must contain valid UTF-8");
    } else {
    }
    CFMutableStringRef lowercase =
        CFStringCreateMutableCopy(kCFAllocatorDefault, 0, source);
    CFRelease(source);
    // Note (Akazaakane): a null locale keeps tag casing independent of the
    // user's locale.
    CFStringLowercase(lowercase, nullptr);
    const CFRange characters = CFRangeMake(0, CFStringGetLength(lowercase));
    CFIndex byte_count = 0;
    CFStringGetBytes(lowercase, characters, kCFStringEncodingUTF8, 0, false,
                     nullptr, 0, &byte_count);
    std::string content(static_cast<size_t>(byte_count), '\0');
    CFStringGetBytes(lowercase, characters, kCFStringEncodingUTF8, 0, false,
                     reinterpret_cast<UInt8 *>(content.data()), byte_count,
                     nullptr);
    CFRelease(lowercase);
    lowered += "[" + content + "]";
    offset = match->position() + match->length();
  }
  lowered += normalized.substr(offset);
  return "<|im_start|><|caption_start|>" + cleaned +
         "<|caption_end|><|lyrics_start|>[start]\n" + lowered +
         "<|lyrics_end|><|im_end|><|audio_start|>";
}

Request ParseRequest(const nlohmann::json &body,
                     const qwen3_asr::Tokenizer &tokenizer,
                     const std::string &model_name, int context_tokens) {
  if (!body.is_object()) {
    throw std::invalid_argument("request must be a JSON object");
  } else {
  }
  const std::string lyrics = body.at("input").get<std::string>();
  const std::string caption = body.at("instructions").get<std::string>();
  if (lyrics.find_first_not_of(" \t\r\n") == std::string::npos ||
      caption.find_first_not_of(" \t\r\n") == std::string::npos) {
    throw std::invalid_argument(
        "input and instructions must be nonempty strings");
  } else if (body.contains("model") &&
             body.at("model").get<std::string>() != model_name) {
    throw std::invalid_argument("model does not match the served model");
  } else if (body.value("stream", false)) {
    throw std::invalid_argument("Music3 supports non-streaming output only");
  } else if (body.contains("speed") &&
             (!body.at("speed").is_number() ||
              body.at("speed").get<double>() != 1.0)) {
    throw std::invalid_argument("Music3 supports only speed=1");
  } else {
  }
  Request request;
  const auto integer = [&](const std::string &name, int fallback, int maximum) {
    if (!body.contains(name)) {
      return fallback;
    } else if (!body.at(name).is_number_integer() ||
               body.at(name).get<double>() < 1 ||
               body.at(name).get<double>() > maximum) {
      throw std::invalid_argument(name + " must be an integer in [1, " +
                                  std::to_string(maximum) + "]");
    } else {
      return body.at(name).get<int>();
    }
  };
  request.max_frames =
      integer("max_new_tokens", request.max_frames, kMaximumFrames);
  request.dit_steps = integer("dit_steps", request.dit_steps, 100);
  if (body.contains("seed")) {
    const auto &seed = body.at("seed");
    if (!seed.is_number_integer() || seed.get<double>() < 0 ||
        seed.get<uint64_t>() > std::numeric_limits<int64_t>::max()) {
      throw std::invalid_argument("seed must be an integer in [0, 2**63)");
    } else {
      request.seed = seed.get<uint64_t>();
    }
  } else {
  }
  if (body.contains("dit_cfg_scale")) {
    if (!body.at("dit_cfg_scale").is_number()) {
      throw std::invalid_argument("dit_cfg_scale must be numeric");
    } else {
      request.dit_guidance = body.at("dit_cfg_scale").get<float>();
    }
  } else {
  }
  if (!std::isfinite(request.dit_guidance) || request.dit_guidance < 0) {
    throw std::invalid_argument("dit_cfg_scale must be finite and nonnegative");
  } else {
  }
  const std::string format = body.value("response_format", std::string("wav"));
  if (format != "wav" && format != "pcm") {
    throw std::invalid_argument("response_format must be wav or pcm");
  } else {
    request.pcm = format == "pcm";
  }
  request.token_ids = tokenizer.Encode(BuildPrompt(caption, lyrics));
  if (request.token_ids.size() > kMaximumPromptTokens ||
      request.token_ids.size() + request.max_frames + 1 >
          static_cast<size_t>(context_tokens)) {
    throw std::invalid_argument(
        "Music3 prompt or requested generation exceeds the context limit");
  } else {
  }
  return request;
}

} // namespace music3
