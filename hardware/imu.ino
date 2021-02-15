#include <Wire.h>
#include <Adafruit_Sensor.h>

#define NUM_SENSOR 5
#define TCAADDR 0x70

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#define PIN        40
#define NUMPIXELS 1

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

#define NXP_FXOS8700_FXAS21002      (2)

// Define your target sensor(s) here based on the list above!
// #define AHRS_VARIANT    ST_LSM303DLHC_L3GD20
#define AHRS_VARIANT   NXP_FXOS8700_FXAS21002

#if AHRS_VARIANT == NXP_FXOS8700_FXAS21002
#include <Adafruit_FXAS21002C.h>
#include <Adafruit_FXOS8700.h>
#else
#error "AHRS_VARIANT undefined! Please select a target sensor combination!"
#endif


// Create sensor instances.
Adafruit_FXAS21002C gyro[NUM_SENSOR];
Adafruit_FXOS8700 accelmag[NUM_SENSOR];

void tcaselect(uint8_t i) {
  if (i > 7) return;

  Wire.beginTransmission(TCAADDR);
  Wire.write(1 << i);
  Wire.endTransmission();
}

void pixel_blue() {
  pixels.setPixelColor(0, pixels.Color(0, 0, 255));
  pixels.show();
}

void pixel_green(float alpha=1) {
  pixels.setPixelColor(0, pixels.Color(0, 255*alpha, 0));
  pixels.show();
}

void pixel_red() {
  pixels.setPixelColor(0, pixels.Color(255, 0, 0));
  pixels.show();
}

void pixel_black() {
  pixels.clear();
}

void toggle_green() {
  static bool toggle = 0;
  toggle = !toggle;
  if (toggle) {
    pixel_green(0.1);
  } else {
    pixel_green(0.01);
  }
}

void setup()
{
  #if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
    clock_prescale_set(clock_div_1);
  #endif
  pixels.begin();
  pixels.clear();

  pixel_blue();
  
  Wire.begin();

  Serial.begin(1000000);

  // Wait for the Serial Monitor to open (comment out to run without Serial Monitor)
//  while (!Serial);

  for (int i = 0; i < NUM_SENSOR; i++) {
    tcaselect(i);

    gyro[i] = Adafruit_FXAS21002C(0x0021002C);
    accelmag[i] = Adafruit_FXOS8700(0x8700A, 0x8700B);

    // Initialize the sensors.
    if (!gyro[i].begin())
    {
      /* There was a problem detecting the gyro ... check your connections */
      Serial.println("Ooops, no gyro detected ... Check your wiring!");
      
      pixel_red();
      while (1);
    }

    if (!accelmag[i].begin(ACCEL_RANGE_4G))
    {
      Serial.println("Ooops, no FXOS8700 detected ... Check your wiring!");

      pixel_red();
      while (1);
    }
  }
}

void loop(void)
{
  toggle_green();
  
  sensors_event_t gyro_event;
  sensors_event_t accel_event;

//  Serial.print(millis());
//  Serial.print(';');

  for (int i = 0; i < 5; i++) {
    tcaselect(i);

    // Get new data samples
    gyro[i].getEvent(&gyro_event);
    accelmag[i].getEvent(&accel_event);

    Serial.print(accel_event.acceleration.x);
    Serial.print(',');
    Serial.print(accel_event.acceleration.y);
    Serial.print(',');
    Serial.print(accel_event.acceleration.z);
    Serial.print(',');
    Serial.print(gyro_event.gyro.x);
    Serial.print(',');
    Serial.print(gyro_event.gyro.y);
    Serial.print(',');
    Serial.print(gyro_event.gyro.z);
    Serial.print(';');
  }

  Serial.println();
}