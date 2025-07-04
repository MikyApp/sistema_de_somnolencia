from ultralytics import YOLO
import cv2
import time
import pyttsx3
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np

class FatigueDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección de somnolencia")
        self.root.geometry("640x640")
        self.root.resizable(True, True)
        
        # Variables de estado
        self.is_running = False
        self.thread = None
        self.closed_start_time = None
        self.yawn_count = 0
        self.last_yawn_time = 0
        self.alarm_triggered = False
        
        # Inicializar motor de voz
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Crear interfaz
        self.create_widgets()
        
        # Iniciar con la cámara apagada
        self.cap = None
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de video
        self.video_frame = ttk.LabelFrame(main_frame, text="Cámara en Vivo")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.root.geometry("640x800")

        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de información
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="Bostezos detectados:").pack(side=tk.LEFT, padx=(0, 5))
        self.yawn_counter = ttk.Label(info_frame, text="0/4", font=("Arial", 10, "bold"))
        self.yawn_counter.pack(side=tk.LEFT)
        
        ttk.Label(info_frame, text="Ojos cerrados:").pack(side=tk.LEFT, padx=(20, 5))
        self.closed_duration = ttk.Label(info_frame, text="0.0s", font=("Arial", 10, "bold"))
        self.closed_duration.pack(side=tk.LEFT)
        
        # Panel de alertas
        alert_frame = ttk.LabelFrame(main_frame, text="Alertas")
        alert_frame.pack(fill=tk.X, pady=5)
        
        self.alert_text = tk.StringVar()
        self.alert_text.set("Sistema detenido")
        alert_label = ttk.Label(alert_frame, textvariable=self.alert_text, foreground="red")
        alert_label.pack(padx=10, pady=5)
        
        # Panel de botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5 )
        
        self.start_btn = ttk.Button(button_frame, text="Iniciar Sistema", command=self.start_system)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Finalizar Sistema", command=self.stop_system, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Salir", command=self.close_app).pack(side=tk.RIGHT, padx=5)
        
    def speak(self, message, is_alarm=False):
        def run():
            if is_alarm:
                self.engine.setProperty('volume', 1.0)
            else:
                self.engine.setProperty('volume', 0.8)
                
            self.engine.say(message)
            self.engine.runAndWait()
        
        threading.Thread(target=run).start()
    
    def start_system(self):
        if not self.is_running:
            try:
                # Inicializar cámara
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "No se pudo abrir la cámara")
                    return
                
                # Inicializar modelo
                self.model = YOLO("modelo_entrenado/bestv2.pt")
                
                # Reiniciar contadores
                self.closed_start_time = None
                self.yawn_count = 0
                self.last_yawn_time = 0
                self.alarm_triggered = False
                
                # Actualizar interfaz
                self.is_running = True
                self.start_btn.config(state=tk.DISABLED)
                self.stop_btn.config(state=tk.NORMAL)
                self.alert_text.set("Sistema iniciado")
                self.yawn_counter.config(text="0/4")
                self.closed_duration.config(text="0.0s")
                
                # Iniciar hilo de detección
                self.thread = threading.Thread(target=self.run_detection, daemon=True)
                self.thread.start()
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo iniciar el sistema:\n{str(e)}")
                self.stop_system()
    
    def stop_system(self):
        if self.is_running:
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.alert_text.set("Sistema detenido")
            
            # Liberar recursos
            if self.cap and self.cap.isOpened():
                self.cap.release()
    
    def run_detection(self):
        try:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.root.after(0, lambda: self.alert_text.set("Error: No se pudo leer la cámara"))
                    break
                
                # Realizar inferencia
                results = self.model(frame, conf=0.3)
                annotated_frame = results[0].plot()
                
                # Procesar detecciones
                current_time = time.time()
                closed_detected = False
                yawn_detected = False
                
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    for box in detections:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        if class_name == "cerrados":
                            closed_detected = True
                        elif class_name == "bostezo":
                            if current_time - self.last_yawn_time > 2.0:
                                self.yawn_count += 1
                                self.last_yawn_time = current_time
                                yawn_detected = True
                
                # Actualizar contadores
                self.root.after(0, lambda: self.yawn_counter.config(text=f"{self.yawn_count}/4"))
                
                # Manejar ojos cerrados
                if closed_detected:
                    if self.closed_start_time is None:
                        self.closed_start_time = current_time
                        self.alarm_triggered = False
                    else:
                        duration = current_time - self.closed_start_time
                        self.root.after(0, lambda: self.closed_duration.config(text=f"{duration:.1f}s"))
                        
                        # Alertas
                        if 4 < duration <= 5:
                            self.speak("Te vez cansado, descansa un poco")
                        elif duration > 20 and not self.alarm_triggered:
                            self.speak("¡ALARMA! Llevas más de un minuto con los ojos cerrados", is_alarm=True)
                            self.alarm_triggered = True
                else:
                    self.closed_start_time = None
                    self.root.after(0, lambda: self.closed_duration.config(text="0.0s"))
                
                # Manejar bostezos
                if yawn_detected and self.yawn_count >= 4:
                    self.speak("Estás bostezando mucho, descansa un poco")
                    self.yawn_count = 0
                    self.root.after(0, lambda: self.yawn_counter.config(text="0/4"))
                
                # Mostrar video en la interfaz
                self.update_video(annotated_frame)
                
                # Pequeña pausa para no saturar
                time.sleep(0.01)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error en detección: {str(e)}"))
        finally:
            # Limpiar al finalizar
            if self.cap and self.cap.isOpened():
                self.cap.release()
        
    def update_video(self, frame):
        # Convertir imagen de OpenCV a formato compatible con Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        
        # Actualizar el widget de video
        self.video_label.configure(image=img)
        self.video_label.image = img
    
    def close_app(self):
        self.stop_system()
        self.root.destroy()


root = tk.Tk()
app = FatigueDetectionSystem(root)
root.protocol("WM_DELETE_WINDOW", app.close_app)
root.mainloop()