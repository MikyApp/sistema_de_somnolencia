import cv2
import os

def video_a_frames(ruta_video, carpeta_salida, paso=1, formato='jpg'):

    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Abrir el video
    video = cv2.VideoCapture(ruta_video)
    
    # Verificar si el video se abrió correctamente
    if not video.isOpened():
        raise ValueError("No se pudo abrir el video: " + ruta_video)
    
    contador = 0
    frame_actual = 0
    
    while True:
        # Leer siguiente frame
        exito, frame = video.read()
        
        # Terminar si no hay más frames
        if not exito:
            break
            
        # Procesar solo si es el frame adecuado según el paso
        if contador % paso == 0:
            # Guardar frame
            nombre_frame = os.path.join(carpeta_salida, f"frame_{frame_actual:06d}.{formato}")
            cv2.imwrite(nombre_frame, frame)
            frame_actual += 1
        
        contador += 1
    
    # Liberar recursos
    video.release()
    print(f"Extraídos {frame_actual} frames guardados en: {carpeta_salida}")

# Ejemplo de uso
video_a_frames(
    ruta_video = "videos/nini_cerrados.mp4",        
    carpeta_salida = "cerrados/nini",    
    paso = 1,                           # Extraer cada frame
    formato = "jpg"                      # Formato JPEG
)
