import cv2, cvzone, numpy as np  # Importa OpenCV, CVZone e NumPy
from cvzone.HandTrackingModule import HandDetector  # Detector de mãos do CVZone
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # Para controlar volume do Windows
from ctypes import cast, POINTER  # Para conversão de ponteiros usada pelo Pycaw
from comtypes import CLSCTX_ALL  # Contexto de execução COM necessário para Pycaw

# --- CONFIGURAÇÃO DO ÁUDIO ---
devices = AudioUtilities.GetSpeakers()  # Obtém os dispositivos de saída de áudio
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # Cria interface para controle de volume
volume = cast(interface, POINTER(IAudioEndpointVolume))  # Converte para ponteiro da interface de volume
volMin, volMax = volume.GetVolumeRange()[:2]  # Obtém o volume mínimo e máximo em decibéis

# --- CONFIGURAÇÃO DA CÂMERA ---
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Inicia a webcam (índice 0)
video.set(3, 1280); video.set(4, 720)  # Define largura e altura da câmera
detector = HandDetector(detectionCon=0.8, maxHands=1)  # Cria detector de mãos (80% confiança, 1 mão)

# --- CALIBRAÇÃO DISTÂNCIA PIXEL → CENTÍMETRO ---
coef = np.polyfit([300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57],
                  [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], 2)  # Ajuste polinomial 2º grau
minD, maxD = 57, 300  # Limites mínimos e máximos da distância em pixels

# --- LOOP PRINCIPAL ---
while True:
    ret, img = video.read()  # Captura o frame da câmera
    if not ret: break  # Se não conseguir capturar, sai do loop

    hands, img = detector.findHands(img, draw=False)  # Detecta mãos no frame

    if hands:  # Se uma mão for detectada
        l = hands[0]['lmList']  # Lista de landmarks da mão
        x, y, w, h = hands[0]['bbox']  # Caixa delimitadora da mão

        # Calcula distância entre base do dedo médio (ponto 5) e dedo mínimo (ponto 17)
        dist = np.clip(np.hypot(l[17][0]-l[5][0], l[17][1]-l[5][1]), minD, maxD)

        # Converte distância em pixels para centímetros usando o polinômio
        dcm = coef[0]*dist**2 + coef[1]*dist + coef[2]

        # Mapeia distância (20–100 cm) para volume (0.0–1.0) e ajusta limite do Windows
        v = np.interp(dcm, [20,100], [0.0,1.0])*1.13
        volume.SetMasterVolumeLevelScalar(min(v,1.0), None)  # Aplica o volume

        # Mostra valor do volume na tela
        cvzone.putTextRect(img, f'Vol: {int(min(v,1.0)*100)}%', (x+5, y-10))
        # Desenha um retângulo ao redor da mão
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 3)

    # Mostra o frame com sobreposição
    cv2.imshow('Controle de Volume', img)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- FINALIZAÇÃO ---
video.release()  # Libera a câmera
cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV
