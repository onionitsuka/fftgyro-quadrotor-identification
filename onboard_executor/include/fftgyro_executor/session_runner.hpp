#pragma once

#include <filesystem>
#include <string>

namespace fftgyro
{
/** @brief 設定検査・目標波形の出力・一連の動的試験の入口。 */
int sessionCommand(
  const std::string& _command,
  const std::filesystem::path& _config,
  const std::filesystem::path& _output = {});
}  // namespace fftgyro
