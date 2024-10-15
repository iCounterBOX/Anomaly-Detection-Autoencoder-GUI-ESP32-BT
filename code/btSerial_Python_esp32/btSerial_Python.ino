/*
 This example code is in the Public Domain (or CC0 licensed, at your option.)
 By Evandro Copercini - 2018 - copy von dem arduino example  serialToSerialBT.ino

 This example creates a bridge between Serial and Classical Bluetooth (SPP)
 and also demonstrate that SerialBT have the same functionalities of a normal Serial
 Note: Pairing is authenticated automatically by this device

 COM port:  Silicon Labs CP210x USB to UART BRIDGE (COM8)
 D:\ALL_PROJECT\VS_vMicro_ESP\ES32_2024\simpleWifiScan\btSerial_Python\btSerial_Python.ino

 ..DIE BLAUE blink-led:
 * https://circuits4you.com/2018/02/02/esp32-led-blink-example/
 * ESP32 LED Blink Example - ON Board LED GPIO 2
 * 
 PYTHON  as RECEIVER :

 # terminal.py - simple terminal emulator - requires pyserial

import serial
import sys
import msvcrt
import time

serialPort = serial.Serial(
    port="COM8", baudrate=115200, bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE
)
serialPort.rtscts = False
serialPort.xonxoff = False
serialPort.dsrdtr = False
sys.stdout.write("Python terminal emulator \n")
serialPort.write('hello terminal\n'.encode())
while 1:
    try:
        # serial data received?  if so read byte and print it
        if serialPort.in_waiting > 0:
            char = serialPort.read()
            sys.stdout.write(str(char, 'ASCII'))
            serialPort.write(str(char, 'ASCII'))
        # keyboard hit? if so read key and transmit over serial
        if msvcrt.kbhit():
            char = msvcrt.getch()
            serialPort.write(char)
    except:
        pass


 ..is printing :
Hallo Welt outLoop 1159
Hallo Welt outLoop 1160
Hallo Welt outLoop 1161
....

 */


#include "BluetoothSerial.h"
String device_name = "ESP32Bluetooth";

// Check if Bluetooth is available
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to and enable it
#endif

// Check Serial Port Profile
#if !defined(CONFIG_BT_SPP_ENABLED)
#error Serial Port Profile for Bluetooth is not available or not enabled. It is only available for the ESP32 chip.
#endif

BluetoothSerial SerialBT;
#define LED 2

void setup() {

    // Set pin mode
    pinMode(LED, OUTPUT);

    Serial.begin(115200);
    SerialBT.begin(device_name);  //Bluetooth device name
    //SerialBT.deleteAllBondedDevices(); // Uncomment this to delete paired devices; Must be called after begin
    Serial.printf("The device with name \"%s\" is started.\nNow you can pair it with Bluetooth!\n", device_name.c_str());
}

int i = 0;
void loop() {
    delay(500);
    digitalWrite(LED, HIGH);
    
    if (Serial.available()) {
        SerialBT.write(Serial.read());
        Serial.printf("Serial.available() is ok..: %d \n" , i); // die msg erscheint ca 20 mal beim client/receiver ( python ) - dnch nur noch der loop unten
    }
    if (SerialBT.available()) {  //  IN diesem SKETCH spielt BT keine rolle -- NUR Serial
        //Serial.write(SerialBT.read());
        Serial.write("Hallo Welt %d", i);
        Serial.printf("Hallo Welt printf %d \n", i);
    }
    i = i + 1;
    Serial.printf("Hallo Welt outLoop %d \n", i); // läuft wunderbar und pausenlos
    delay(500);
    digitalWrite(LED, LOW);
}



