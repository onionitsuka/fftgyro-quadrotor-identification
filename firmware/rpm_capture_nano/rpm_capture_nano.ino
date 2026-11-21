#include <Arduino.h>
#include <util/atomic.h>
#include "rpm_wire.h"

// D2-D5 are inputs only. Never connect the Uno marker-output sketch to this circuit.
namespace {
struct Edge { uint32_t time; uint8_t channel; };
volatile Edge edges[256];
volatile uint8_t head = 0, tail = 0;
volatile uint32_t dropped = 0;
volatile bool recording = false;
uint32_t capture = 0, sequence = 0, heartbeat = 0;
bool ending = false;

void edge(uint8_t channel)
{
  if (!recording) return;
  const uint8_t next = static_cast<uint8_t>(head + 1);
  if (next == tail) { ++dropped; return; }
  edges[head].time = micros();
  edges[head].channel = channel;
  head = next;
}
void s1() { edge(1); }
void s2() { edge(2); }
void s3() { edge(3); }
void s4() { edge(4); }

void send(char kind, uint32_t time, uint32_t value)
{
  uint8_t frame[rpm::kFrameSize];
  rpm::encode(frame, kind, capture, sequence++, time, value);
  Serial.write(frame, sizeof(frame));
}

uint32_t losses()
{
  uint32_t value;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { value = dropped; }
  return value;
}

void command(const char* line)
{
  if (line[0] == 'B' && strlen(line) == 9 && !recording && !ending) {
    char* end;
    const uint32_t id = strtoul(line + 1, &end, 16);
    if (*end || !id) return;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { head = tail = 0; dropped = 0; }
    capture = id; sequence = 0; heartbeat = micros();
    send('B', heartbeat, 0);
    recording = true;
  } else if (line[0] == 'T' && strlen(line) == 9 && recording) {
    char* end;
    const uint32_t token = strtoul(line + 1, &end, 16);
    if (!*end) send('T', micros(), token);
  } else if (strcmp(line, "Z") == 0 && recording) {
    recording = false;
    ending = true;
  }
}
}  // namespace

void setup()
{
  Serial.begin(1000000);
  for (uint8_t pin = 2; pin <= 5; ++pin) pinMode(pin, INPUT);
  attachInterrupt(digitalPinToInterrupt(2), s1, FALLING);
  attachInterrupt(digitalPinToInterrupt(3), s2, FALLING);
  attachInterrupt(digitalPinToInterrupt(4), s3, FALLING);
  attachInterrupt(digitalPinToInterrupt(5), s4, FALLING);
}

void loop()
{
  // At most one short command per pass; reserve room for its reply first.
  static char line[12];
  static uint8_t length = 0;
  static bool invalid = false;
  for (uint8_t n = 0; n < 12 && Serial.available() && Serial.availableForWrite() >= 22; ++n) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      line[length] = 0;
      if (!invalid) command(line);
      length = 0; invalid = false;
      break;
    }
    if (length < sizeof(line) - 1) line[length++] = c;
    else invalid = true;
  }
  if (recording && static_cast<uint32_t>(micros() - heartbeat) >= 100000 && Serial.availableForWrite() >= 22) {
    heartbeat = micros();
    send('H', heartbeat, losses());
  }
  for (uint8_t n = 0; n < 16 && tail != head && Serial.availableForWrite() >= 22; ++n) {
    Edge value;
    ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
      value.time = edges[tail].time; value.channel = edges[tail].channel;
      tail = static_cast<uint8_t>(tail + 1);
    }
    send('E', value.time, value.channel);
  }
  if (ending && tail == head && Serial.availableForWrite() >= 22) {
    send('Z', micros(), losses());
    ending = false;
  }
}
