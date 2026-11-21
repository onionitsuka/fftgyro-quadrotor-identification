// Slow hand-test only: print the four receiver logic levels at 10 Hz.
// D2-D5 remain inputs. No motor, sync, or other control output is generated.
// External pull-ups and the 74HC14 circuit provide the input levels.

void setup() {
  for (int pin = 2; pin <= 5; ++pin) {
    pinMode(pin, INPUT);
  }
  Serial.begin(115200);
  Serial.println("ms,D2,D3,D4,D5");
}

void loop() {
  Serial.print(millis());
  for (int pin = 2; pin <= 5; ++pin) {
    Serial.print(',');
    Serial.print(digitalRead(pin));
  }
  Serial.println();
  delay(100);
}
