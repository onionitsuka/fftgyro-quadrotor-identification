// EE-SPY401 pulse and hardware-sync edge capture for Arduino Uno R3.
//
// Wiring:
//   D2 <- EE-SPY401 NPN output with an external pull-up to 5 V
//   D3 <- D4 through 1 kohm (physical COMMAND_ON/OFF edge loopback)
//   D4 -> 1 kohm -> HIOKI LR8450 EXT.I/O PULSE2 (as-built circuit)
//   Arduino GND -> HIOKI EXT.I/O GND at one point (non-isolated)
//
// Output at 1000000 baud:
//   event,label,micros
//   R,123456       (RPM reflection edge)
//   S,124000       (hardware synchronization edge)
//   M,COMMAND_ON,125000 (named software marker)
//   O,1            (one or more ring-buffer overflows occurred)
// Sending one of the five named marker commands records an M event at the Uno
// receive time. Legacy "MARK" still records an S event.

#include <Arduino.h>

namespace {

constexpr uint8_t kRpmPin = 2;
constexpr uint8_t kSyncPin = 3;
constexpr uint8_t kHiokiMarkerPin = 4;
constexpr uint16_t kBufferSize = 192;

struct EdgeEvent {
  uint32_t time_us;
  char kind;
  uint8_t marker_id;
};

volatile EdgeEvent events[kBufferSize];
volatile uint8_t head = 0;
volatile uint8_t tail = 0;
volatile bool overflowed = false;

void recordEvent(char kind, uint8_t marker_id = 0) {
  const uint8_t next = static_cast<uint8_t>((head + 1) % kBufferSize);
  if (next == tail) {
    overflowed = true;
    return;
  }
  events[head].time_us = micros();
  events[head].kind = kind;
  events[head].marker_id = marker_id;
  head = next;
}

void onRpmEdge() { recordEvent('R'); }
void onSyncEdge() { recordEvent('S'); }

bool popEvent(EdgeEvent* event) {
  noInterrupts();
  if (tail == head) {
    interrupts();
    return false;
  }
  event->time_us = events[tail].time_us;
  event->kind = events[tail].kind;
  event->marker_id = events[tail].marker_id;
  tail = static_cast<uint8_t>((tail + 1) % kBufferSize);
  interrupts();
  return true;
}

uint8_t markerId(const char* command) {
  if (strcmp(command, "MARK_START") == 0) return 1;
  if (strcmp(command, "COMMAND_ON") == 0) return 2;
  if (strcmp(command, "COMMAND_ACK") == 0) return 3;
  if (strcmp(command, "COMMAND_OFF") == 0) return 4;
  if (strcmp(command, "MARK_END") == 0) return 5;
  return 0;
}

const __FlashStringHelper* markerName(uint8_t marker_id) {
  switch (marker_id) {
    case 1: return F("MARK_START");
    case 2: return F("COMMAND_ON");
    case 3: return F("COMMAND_ACK");
    case 4: return F("COMMAND_OFF");
    case 5: return F("MARK_END");
    default: return F("UNKNOWN");
  }
}

void printMarker(uint8_t marker_id, uint32_t time_us) {
  // Markers must never compete with the high-rate RPM ring buffer. Emit them
  // immediately from loop() so command timing survives an RPM edge flood.
  Serial.print(F("M,"));
  Serial.print(markerName(marker_id));
  Serial.print(',');
  Serial.println(time_us);
}

void updateHiokiMarker(uint8_t marker_id) {
  // D4 is HIGH during the commanded motor run. With the as-built
  // non-isolated interface, HIOKI PULSE2 is active HIGH: rising edge is ON and
  // falling edge is OFF. MARK_START/MARK_END force D4 LOW between experiments.
  if (marker_id == 2) {
    digitalWrite(kHiokiMarkerPin, HIGH);  // COMMAND_ON
  } else if (marker_id == 1 || marker_id == 4 || marker_id == 5) {
    digitalWrite(kHiokiMarkerPin, LOW);   // START/OFF/END
  }
}

}  // namespace

void setup() {
  Serial.begin(1000000);
  pinMode(kRpmPin, INPUT);   // External 4.7--10 kohm pull-up is required.
  pinMode(kSyncPin, INPUT);  // Buffer/isolator must provide a defined level.
  pinMode(kHiokiMarkerPin, OUTPUT);
  digitalWrite(kHiokiMarkerPin, LOW);
  attachInterrupt(digitalPinToInterrupt(kRpmPin), onRpmEdge, FALLING);
  attachInterrupt(digitalPinToInterrupt(kSyncPin), onSyncEdge, CHANGE);
  Serial.println(F("event,label,micros"));
}

void loop() {
  while (Serial.available() > 0) {
    static char command[20];
    static uint8_t command_length = 0;
    const char value = static_cast<char>(Serial.read());
    if (value == '\r') {
      continue;
    }
    if (value == '\n') {
      command[command_length] = '\0';
      if (strcmp(command, "MARK") == 0) {
        recordEvent('S');
      } else {
        const uint8_t marker_id = markerId(command);
        if (marker_id != 0) {
          const uint32_t marker_time_us = micros();
          updateHiokiMarker(marker_id);
          printMarker(marker_id, marker_time_us);
        }
      }
      command_length = 0;
      continue;
    }
    if (command_length + 1 < sizeof(command)) {
      command[command_length++] = value;
    } else {
      command_length = 0;
    }
  }

  EdgeEvent event{};
  // Bound each drain pass so a continuous RPM edge flood cannot starve the
  // serial command parser and delay COMMAND_ACK/COMMAND_OFF indefinitely.
  for (uint8_t drained = 0; drained < 24 && popEvent(&event); ++drained) {
    Serial.print(event.kind);
    Serial.print(',');
    if (event.kind == 'M') Serial.print(markerName(event.marker_id));
    Serial.print(',');
    Serial.println(event.time_us);
  }

  noInterrupts();
  const bool report_overflow = overflowed;
  overflowed = false;
  interrupts();
  if (report_overflow) {
    Serial.println(F("O,,1"));
  }
}
