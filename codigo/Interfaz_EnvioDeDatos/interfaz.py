import tkinter as tk
from tkinter import filedialog
import pandas as pd
import serial.tools.list_ports
import serial

# Variables globales
columnas = ["Spo2","Fc_Masimo","Fr","Vpi","Pi","T_manos","T_corp",
            "Maximo","Minimo","RMS_Signal","RMS_V_pico","RMS_V_w","RMS_Vpp",
            "Promedio_Signal","Promedio_Pico","Promedio_Ancho","Promedio_Prominencia",
            "STD_Signal","STD_V_pico","STD_Vpp","STD_V_w"]
labels_text = [
    "Saturación de Oxigeno (SpO2)",
    "Frecuencia cardiaca (FC)",
    "Frecuencia respiratoria (FR)",
    "Variabilidad fotopletismografica (VPI)",
    "Indice de perfusión (PI)",
    "Temperatura de las manos (TM)",
    "Temperatura corporal (TC)",
    "Maximo",
    "Minimo",
    "RMS señal",
    "RMS Voltaje pico",
    "RMS Voltaje ancho",
    "RMS Voltaje pico-pico",
    "Promedio señal",
    "Promedio pico",
    "Promedio ancho",
    "Promedio prominencia",
    "STD señal",
    "STD Voltaje pico",
    "STD Voltaje ancho",
    "STD voltaje pico-pico"
]


entradas = {}
ESP32_COM = None
esp = None  # Objeto serial
estado_label = None

# 🔍 Listar puertos disponibles
def listar_puertos():
    puertos = serial.tools.list_ports.comports()
    return [p.device for p in puertos]

def puerto_estatus(nombre_puerto):
    puertos = [p.device for p in serial.tools.list_ports.comports()]
    return nombre_puerto in puertos

# 📂 Seleccionar archivo Excel
def select_file():
    file_selected = filedialog.askopenfilename(
        title="Selecciona base de datos (Excel)",
        filetypes=[("Archivo .xlsx", "*.xlsx"), ("Todos los archivos", "*.*")]
    )
    entradas["DBaddress"].delete(0, tk.END)
    entradas["DBaddress"].insert(0, file_selected)

# 📥 Cargar datos desde Excel
def load_data():
    try:
        df = pd.read_excel(entradas["DBaddress"].get())
        df.columns = df.columns.str.strip()
        fila = int(entradas["ID"].get())

        for col in columnas:
            entradas[col].delete(0, tk.END)
            entradas[col].insert(0, str(df.loc[fila, col]))
        print("✅ Datos cargados correctamente")
    except Exception as e:
        print(f"⚠️ Error al cargar datos: {e}")

# 📤 Enviar datos al ESP32
def send_data():
    global esp
    if esp and esp.is_open:
        mensaje = ",".join([entradas[col].get() for col in columnas])
        try:
            esp.write((mensaje + "\n").encode())
            print("📤 Enviado:", mensaje)
        except Exception as e:
            print(f"❌ Error al enviar: {e}")
    else:
        print("⚠️ ESP32 no conectado")

# 🔌 Conectar al ESP32 por puerto COM
def conectar_esp32():
    global esp
    try:
        esp = serial.Serial(ESP32_COM, 9600, timeout=1)
        print(f"✅ Conectado a {ESP32_COM}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        esp = None

# 🔄 Verificación periódica de conexión
import threading

def verificar_conexion_thread():
    def tarea():
        global esp
        estado = "⚠️ No conectado"
        color = "orange"

        if esp and esp.is_open and puerto_estatus(ESP32_COM):
            try:
                esp.write(b"ping\n")
                estado = "🔗 Conectado"
                color = "green"
            except:
                estado = "❌ Desconectado"
                color = "red"
        else:
            estado = "⚠️ No conectado"
            color = "orange"
        # Actualizar interfaz desde el hilo principal
        main_root.after(0, lambda: estado_label.config(text=estado, fg=color))

        # Repetir en 5 segundos
        threading.Timer(5, verificar_conexion_thread).start()

    threading.Thread(target=tarea, daemon=True).start()

# 🧩 Ventana principal
def lanzar_principal():
    global main_root, estado_label
    main_root = tk.Tk()
    main_root.title("Envío de datos")
    root = tk.Frame(main_root, padx=20, pady=20)
    root.grid(row=0, column=0)

    # Dirección base de datos
    tk.Label(root, text="Base de datos:").grid(row=0, column=0)
    entradas["DBaddress"] = tk.Entry(root)
    entradas["DBaddress"].grid(row=0, column=1)
    tk.Button(root, text="Seleccionar", command=select_file).grid(row=0, column=2)

    # ID
    tk.Label(root, text="ID:").grid(row=1, column=0)
    entradas["ID"] = tk.Entry(root)
    entradas["ID"].grid(row=1, column=1)
    tk.Button(root, text="Cargar", command=load_data).grid(row=1, column=2)
    tk.Button(root, text="Enviar", command=send_data).grid(row=11, column=5)

    # Etiquetas de sección
    tk.Label(root, text="Variables Fisiológicas").grid(row=2, column=0)
    tk.Label(root, text="Parámetros de señal PPG").grid(row=2, column=2)

    # Crear entradas dinámicamente
    for i, col in enumerate(columnas):
        fila = 3 + (i % 7)
        col_offset = (i // 7) * 2
        tk.Label(root, text=labels_text[i] + ":").grid(row=fila, column=col_offset)
        entradas[col] = tk.Entry(root)
        entradas[col].grid(row=fila, column=col_offset + 1, padx=5, pady=5)

    # Estado de conexión
    estado_label = tk.Label(root, text="⏳ Verificando conexión...", font=("Arial", 12))
    estado_label.grid(row=0, column=4, pady=10)

    conectar_esp32()
    verificar_conexion_thread()

    main_root.mainloop()

# ⚙️ Ventana de configuración inicial
def BAsave():
    global ESP32_COM
    ESP32_COM = puerto_seleccionado.get()
    print(f"🔧 Puerto configurado: {ESP32_COM}")
    cc.destroy()
    lanzar_principal()

cc = tk.Tk()
cc.title("Configuración")
c = tk.Frame(cc, padx=20, pady=20)
c.grid(row=0, column=0)

tk.Label(c, text="Puerto COM del ESP32:").grid(row=0, column=0)
puerto_seleccionado = tk.StringVar()
puertos_disponibles = listar_puertos()
puerto_seleccionado.set(puertos_disponibles[0] if puertos_disponibles else "")

menu_puertos = tk.OptionMenu(c, puerto_seleccionado, *puertos_disponibles)
menu_puertos.grid(row=0, column=1)
tk.Button(c, text="Aceptar", command=BAsave).grid(row=0, column=2)

cc.mainloop()
