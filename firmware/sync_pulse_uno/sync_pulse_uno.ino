// FFT GYRO experiment synchronisation pulse generator for Arduino Uno R3.
//
// Electrical contract:
// - SYNC_PIN never drives HIGH. It either sinks LOW or becomes high impedance.
// - Use an external 3.3 V pull-up and an appropriate isolated/buffered fan-out.
// - Never connect an Uno 5 V HIGH output directly to Pixhawk 6X.

#include <Arduino.h>

namespace {

constexpr uint8_t SYNC_PIN = 8;
constexpr uint8_t ESTOP_PIN = 2;
constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long COMMAND_WATCHDOG_MS = 2000;
constexpr unsigned long MIN_PULSE_US = 100;
// Keep this below the AVR delayMicroseconds() practical range.
constexpr unsigned long MAX_PULSE_US = 10000;
constexpr size_t LINE_CAPACITY = 80;

char line_buffer[LINE_CAPACITY];
size_t line_length = 0;
bool armed = false;
unsigned long last_command_ms = 0;
char run_id[25] = "";

void release_sync_line() {
  digitalWrite(SYNC_PIN, LOW);
  pinMode(SYNC_PIN, INPUT);
}

void pull_sync_line_low() {
  digitalWrite(SYNC_PIN, LOW);
  pinMode(SYNC_PIN, OUTPUT);
}

void disarm(const __FlashStringHelper* reason) {
  release_sync_line();
  armed = false;
  run_id[0] = '\0';
  Serial.print(F("DISARMED "));
  Serial.println(reason);
}

bool estop_active() {
  return digitalRead(ESTOP_PIN) == LOW;
}

void handle_arm(char* token) {
  if (estop_active()) {
    Serial.println(F("ERR ESTOP_ACTIVE"));
    return;
  }
  if (token == nullptr || token[0] == '\0' || strlen(token) >= sizeof(run_id)) {
    Serial.println(F("ERR INVALID_RUN_ID"));
    return;
  }
  strncpy(run_id, token, sizeof(run_id) - 1);
  run_id[sizeof(run_id) - 1] = '\0';
  armed = true;
  last_command_ms = millis();
  Serial.print(F("ARMED "));
  Serial.println(run_id);
}

void handle_pulse(char* sequence_token, char* width_token) {
  if (!armed) {
    Serial.println(F("ERR NOT_ARMED"));
    return;
  }
  if (estop_active()) {
    disarm(F("ESTOP"));
    return;
  }
  if (sequence_token == nullptr || width_token == nullptr) {
    Serial.println(F("ERR PULSE_ARGUMENTS"));
    return;
  }

  char* sequence_end = nullptr;
  char* width_end = nullptr;
  const unsigned long sequence = strtoul(sequence_token, &sequence_end, 10);
  const unsigned long width_us = strtoul(width_token, &width_end, 10);
  if (*sequence_end != '\0' || *width_end != '\0') {
    Serial.println(F("ERR PULSE_ARGUMENTS"));
    return;
  }
  if (width_us < MIN_PULSE_US || width_us > MAX_PULSE_US) {
    Serial.println(F("ERR PULSE_WIDTH"));
    return;
  }

  noInterrupts();
  pull_sync_line_low();
  interrupts();
  delayMicroseconds(width_us);
  noInterrupts();
  release_sync_line();
  const unsigned long release_time_us = micros();
  interrupts();

  last_command_ms = millis();
  Serial.print(F("PULSED "));
  Serial.print(run_id);
  Serial.print(' ');
  Serial.print(sequence);
  Serial.print(' ');
  Serial.print(release_time_us);
  Serial.print(' ');
  Serial.println(width_us);
}

void handle_line(char* line) {
  char* command = strtok(line, " ");
  if (command == nullptr) {
    return;
  }
  last_command_ms = millis();

  if (strcmp(command, "PING") == 0) {
    Serial.println(F("PONG sync-pulse-uno 1"));
    return;
  }
  if (strcmp(command, "STATUS") == 0) {
    Serial.print(F("STATUS "));
    Serial.print(armed ? F("ARMED ") : F("DISARMED "));
    Serial.print(estop_active() ? F("ESTOP ") : F("ESTOP_CLEAR "));
    Serial.println(armed ? run_id : "-");
    return;
  }
  if (strcmp(command, "ARM") == 0) {
    handle_arm(strtok(nullptr, " "));
    return;
  }
  if (strcmp(command, "PULSE") == 0) {
    handle_pulse(strtok(nullptr, " "), strtok(nullptr, " "));
    return;
  }
  if (strcmp(command, "DISARM") == 0) {
    disarm(F("HOST"));
    return;
  }
  Serial.println(F("ERR UNKNOWN_COMMAND"));
}

void read_serial() {
  while (Serial.available() > 0) {
    const char value = static_cast<char>(Serial.read());
    if (value == '\r') {
      continue;
    }
    if (value == '\n') {
      line_buffer[line_length] = '\0';
      handle_line(line_buffer);
      line_length = 0;
      continue;
    }
    if (line_length + 1 >= LINE_CAPACITY) {
      line_length = 0;
      Serial.println(F("ERR LINE_TOO_LONG"));
      continue;
    }
    line_buffer[line_length++] = value;
  }
}

}  // namespace

void setup() {
  release_sync_line();
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
  }
  last_command_ms = millis();
  Serial.println(F("READY sync-pulse-uno 1"));
}

void loop() {
  read_serial();
  if (estop_active() && armed) {
    disarm(F("ESTOP"));
  }
  if (armed && millis() - last_command_ms > COMMAND_WATCHDOG_MS) {
    disarm(F("WATCHDOG"));
  }
}
