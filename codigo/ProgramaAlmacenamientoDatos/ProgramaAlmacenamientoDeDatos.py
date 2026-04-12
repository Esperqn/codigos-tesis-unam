import pandas as pd
import serial
import time

#abrir puerto serie
ser= serial.Serial("COM3", 115200, timeout=2)
time.sleep(2)

#Listas para guardar resultados
valores_reales = []
predicciones = []

y_real = [93,93,
          131,131,131,
          140,140,140,
          91,91,91,
          119,119,119,
          103,103,103,
          97,97,
          114,114,114,
          93,93,93,
          80,80,80,
          107,107,107,
          98,98,98,
          84,84,84,
          101,101,101,
          106,106,106,
          75,75,75,
          110,110,110,
          100,100,100,
          85,85,85,85,85,85,85,85,85,85,
          108,108,108,108,108,108,108,108,108,108,
          102,102,102,102,102,102,102,102,102,102,
          92,92,92,92,92,92,92,92,92,92,
          125,125,125,125,125,125,125,125,125,125,
          91,91,91,91,91,91,91,91,91,91,
          89,89,89,89,89,89,89,89,89,89,
          109,109,109,109,109,109,109,109,109,109,
          87,87,87,87,87,87,87,87,87,87,
          86,86,86,86,86,86,86,86,86,86,
          95,95,95,95,95,95,95,95,95,95,
          96,96,96,96,96,96,96,96,96,96,
          85,85,85,85,85,85,85,85,85,85,
          84,84,84,84,84,84,84,84,84,84,
          85,85,85,85,85,85,85,85,85,85,
          83,83,83,83,83,83,83,83,83,83,
          96,96,96,96,96,96,96,96,96,96,
          92,92,92,92,92,92,92,92,92,92,
          72,72,72,72,72,72,72,72,72,72,
          129,129,129,129,129,129,129,129,129,129,
          85,85,85,85,85,85,85,85,85,85,
          118,118,118,118,118,
          100,100,100,100,100,100,100,100,100,100,
          96,96,96,96,96,96,96,96,96,96,
          115,155,115,115,115,115,115,115,115,115,
          101,101,101,101,101,101,101,101,101,101,
          119,119,119,119,119,119,119,119,119,119,
          82,82,82,82,82,82,82,82,82,82,
          90,90,90,90,90,90,90,90,90,90,
          100,100,100,100,100,100,100,100,
          90,90,90,90,90,90,90,90,90,90,
          101,101,101,101,101,101,101,101,101,101,
          86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,86,
          109,109,109,109,109,109,109,109,109,
          92,92,92,92,92,92,92,92,92,92,
          86,86,86,86,86,86,86,86,86,86,
          82,82,82,82,82,82,82,82,82,
          127,127,127,127,127,127,127,127,127,127,
          90,90,90,90,90,90,90,90,90,90,
          163,163,163,163,163,163,163,163,163,163,
          109,109,109,109,109,109,109,109,109,109,
          95,95,95,95,95,95,95,95,95,95,
          116,116,116,116,116,116,116,116,116,116,
          89,89,89,89,89,89,89,89,89,89,
          86,86,86,86,86,86,86,86,86,86,
          89,89,89,89,89,89,89,89,89,
          116,116,116,116,116,116,116,116,116,116,
        ]

latencias = []
heaps=[]
desconexiones=[]

print("Esperando predicciones del ESP32...")

fila = 0

while fila < len(y_real):
    resp = ser.read_until(b"\n").decode(errors="ignore").strip()
    print("resp: ", repr(resp))
    if "Prediccion:" in resp or "Predicción" in resp:
        try:
            pred = float(resp.split(":")[1].strip())
            predicciones.append(pred)
            print(f"Fila {fila}: real={y_real[fila]}, pred={pred}")
            fila += 1
        except:
            print("Error leyendo: ", resp)

    elif "Latencia" in resp:
        try:
            lat = float(resp.split(":")[1].strip())
            latencias.append(lat)
            print(f"latencia={lat} us")
            
        except:
            print("Error leyendo latencia: ", resp)

    elif "Heap libre" in resp:
        try:
            heap = int(resp.split(":")[1].strip())
            heaps.append(heap)
            print(f"heap libre={heap} bytes")
        except:
            print("Error leyendo heap: ", resp)

    elif "Desconexiones" in resp:
        try:
            desc = int(resp.split(":")[1].strip())
            desconexiones.append(desc)
            print(f"desconexiones={desc}")
        except:
            print("Error leyendo desconexiones: ", resp)

ser.close()

print("Longitudes:")
print("y_real:", len(y_real)) 
print("predicciones:", len(predicciones)) 
print("latencias:", len(latencias)) 
print("heaps:", len(heaps)) 
print("desconexiones:", len(desconexiones))
# Guardar resultados en CSV

if len(predicciones) >0: 
    df = pd.DataFrame({
        "y_real": y_real[:len(predicciones)], 
        "y_pred": predicciones
    })
    df.to_csv("predicciones_Modelo_rf50TC.csv", index=False) 
    print("✅ Predicciones guardadas")
else:
    print("⚠️ No se guardaron predicciones, lista vacía")
    

if len(latencias) > 0: 
    df_lat = pd.DataFrame({"Latencia_us": latencias}) 
    df_lat.to_csv("latencias_Modelo_rf50TC.csv", index=False) 
    print("✅ Latencias guardadas") 
else: print("⚠️ No se recibieron latencias")

if len(heaps) > 0: 
    df_heap = pd.DataFrame({"heap_libre": heaps})
    df_heap.to_csv("heaps_Modelo_rf50TC.csv", index=False)
    print("✅ Heaps guardados")
else: 
    print("⚠️ No se recibieron heaps")

if len(desconexiones) > 0: 
    df_desc = pd.DataFrame({"desconexiones": desconexiones})    
    df_desc.to_csv("desconexiones_Modelo_rf50TC.csv", index=False)
    print("✅ Desconexiones guardadas")
else: 
    print("⚠️ No se recibieron desconexiones")
    


print("Predicciones guardadas en predicciones_Modelo_rf50TC.csv")
print("Metricas guardadas en estadisticas_Modelo_rf50TC.csv")
