#include "modelo_rf.h"
#include "BluetoothSerial.h"
#include <stddef.h>
#include <Arduino.h>
#include <TFT_eSPI.h>

#define NUM_FEATURES 21

BluetoothSerial SerialBT;

TFT_eSPI tft = TFT_eSPI();

String etiquetas[]= {
  "SpO2", "FC Máximo", "FR", "VPI", "PI", "T manos", "T corp",
  "Máximo", "Mínimo", "RMS señal", "RMS V pico", "RMS V ancho",
  "RMS Vpp", "Prom señal", "Prom pico", "Prom ancho", "Prom prominencia",
  "STD señal", "STD V pico", "STD V ancho", "STD Vpp"

};

const int numCampos = 21;

bool btConectado = false;
float features[NUM_FEATURES];

int desconexiones =0;

void setup()
{
  Serial.begin(115200);
  SerialBT.begin("ESP32_BT");
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE,TFT_BLACK);
  tft.setTextSize(2);
  tft.setCursor(0,0);
  tft.println("Modelo de IA para");
  tft.setCursor(0,25);
  tft.println("determinar niveles");
  tft.setCursor(0,50);
  tft.println("de glucosa.");
  tft.println("");
  tft.println("");
  tft.setTextColor(TFT_CYAN,TFT_BLACK);
  tft.println("Bluetooth iniciado.");

  delay(1000);
}

void loop(){

  if (SerialBT.hasClient() && !btConectado){
    btConectado = true;
    Serial.println("Conexión Bluetooth establecida");
    tft.fillRect(0,90,240,30,TFT_BLACK);
    tft.setCursor(45,100);
    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.setTextSize(2);
    
    tft.println("BT Conectado!");
  }
  if (!SerialBT.hasClient() && btConectado){
    btConectado = false;
    desconexiones++;
    Serial.println("Desconexiones: ");
    Serial.println(desconexiones);
    Serial.println("Conexión Bluetooth finalizada");
    tft.fillRect(0, 100,200,20, TFT_BLACK);
    tft.setCursor(0,100);
    tft.setTextColor(TFT_CYAN, TFT_BLACK);
    tft.setTextSize(2);
    tft.println("Bluetooth iniciado.");


  }

  if (SerialBT.available()){
    String mensaje = SerialBT.readStringUntil('\n'); //recibe línea completa
    mensaje.trim();
    if (mensaje.length()==0){
      Serial.println("No se recibieron datos");
      return;
    }

    if (mensaje.equalsIgnoreCase("ping")){
      Serial.println("Mensaje de control ignorado");
      return;
    }

    int index = 0;
    char buffer[mensaje.length()+1];
    mensaje.toCharArray(buffer, mensaje.length()+1);

    char *token = strtok(buffer,","); 
    while (token != NULL && index < NUM_FEATURES){
      if (isdigit(token[0]) || token[0] == '-' || token[0] == '.'){
        features[index] = atof(token);
        Serial.print("Feature[");
        Serial.print(index);
        Serial.print("] = ");
        Serial.println(features[index]);
        index++;
      } else{
        Serial.println("Token ignorado: " + String(token));
      }
      token = strtok(NULL, ",");
    }

    if (index < NUM_FEATURES) {
      Serial.println("Datos incompletos, no se pueden predecir");
      return;
    }
    //limpiar pantalla
    tft.fillRect(0,0,240,200,TFT_BLACK);
    tft.setCursor(10,10);
    tft.setTextSize(2);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.println("Datos recibidos:");

    int yPos = 25; // posición inicial
    for (int i = 0; i < 4; i++) {
      if (yPos > 120) {
        yPos = 25;
        tft.fillRect(0,25,240,100,TFT_BLACK);
      }
      tft.setCursor(10, yPos);
      tft.setTextColor(TFT_YELLOW, TFT_BLACK);
      tft.setTextSize(2);
      tft.print(etiquetas[i]);
      tft.print(": ");
      tft.setTextColor(TFT_CYAN, TFT_BLACK);
      tft.println(features[i]);
      yPos += 21; // avanza hacia abajo
      //delay(1000); // pausa para que se vea el recorrido
    }

    unsigned long start = micros();  
    float resultado = predict(features);
    unsigned long end = micros();
    Serial.print("Prediccion: ");
    Serial.println(resultado, 2);
    String mensajeResultado = "Glucosa: " + String(resultado, 2) + "mg/dL";
    tft.setCursor(10,112);
    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.setTextSize(2);
    tft.print(mensajeResultado);
    Serial.print("Latencia (us): ");
    Serial.println(end-start);
    
    Serial.print("Heap libre: ");
    Serial.println(ESP.getFreeHeap());

  }
}
