#pragma once

#include <stddef.h>
#include <stdint.h>

namespace rpm {
constexpr size_t kFrameSize = 22;

inline void put32(uint8_t* p, uint32_t value)
{
  for (uint8_t i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(value >> (8 * i));
}

inline uint32_t get32(const uint8_t* p)
{
  uint32_t value = 0;
  for (uint8_t i = 0; i < 4; ++i) value |= static_cast<uint32_t>(p[i]) << (8 * i);
  return value;
}

inline uint16_t crc16(const uint8_t* p, size_t size)
{
  uint16_t crc = 0xffff;
  while (size--) {
    crc ^= static_cast<uint16_t>(*p++) << 8;
    for (uint8_t bit = 0; bit < 8; ++bit)
      crc = static_cast<uint16_t>((crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1);
  }
  return crc;
}

inline void encode(uint8_t* p, char kind, uint32_t capture, uint32_t sequence, uint32_t time, uint32_t value)
{
  p[0] = 0xa5; p[1] = 0x5a; p[2] = 1; p[3] = kind;
  put32(p + 4, capture); put32(p + 8, sequence); put32(p + 12, time); put32(p + 16, value);
  const uint16_t crc = crc16(p, 20);
  p[20] = static_cast<uint8_t>(crc); p[21] = static_cast<uint8_t>(crc >> 8);
}

inline bool valid(const uint8_t* p)
{
  return p[0] == 0xa5 && p[1] == 0x5a && p[2] == 1 &&
         crc16(p, 20) == (static_cast<uint16_t>(p[20]) | static_cast<uint16_t>(p[21]) << 8);
}
}  // namespace rpm
