# Importar las bibliotecas necesarias
import cv2  # OpenCV para el procesamiento de imágenes
import pytesseract  # pytesseract para la extracción de texto de imágenes

# Configurar la ubicación de Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Crear una lista vacía para almacenar la placa detectada
placa = []

# Leer la imagen del automóvil desde el archivo 'auto001.jpg'
image = cv2.imread('auto001.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de desenfoque a la imagen en escala de grises
gray = cv2.blur(gray, (3, 3))

# Aplicar el detector de bordes Canny a la imagen en escala de grises
canny = cv2.Canny(gray, 150, 200)

# Realizar una operación de dilatación en los bordes detectados
canny = cv2.dilate(canny, None, iterations=1)

# Encontrar los contornos en la imagen
cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Iterar a través de los contornos encontrados
for c in cnts:
    # Calcular el área del contorno
    area = cv2.contourArea(c)
    
    # Obtener las coordenadas del rectángulo delimitador del contorno
    x, y, w, h = cv2.boundingRect(c)
    
    # Calcular la longitud del perímetro del contorno con un pequeño factor de corrección
    epsilon = 0.09 * cv2.arcLength(c, True)
    
    # Aproximar el contorno con un polígono de vértices suavizados
    approx = cv2.approxPolyDP(c, epsilon, True)

    # Verificar si el polígono aproximado tiene 4 vértices y si el área es mayor a 9000
    if len(approx) == 4 and area > 9000:
        # Calcular la relación de aspecto del rectángulo delimitador
        aspect_ratio = float(w) / h
        
        # Si la relación de aspecto es mayor a 2.4, asumimos que es la placa del vehículo
        if aspect_ratio > 2.4:
            # Recortar la región de la placa de la imagen en escala de grises
            placa = gray[y:y+h, x:x+w]
            
            # Utilizar Tesseract OCR para extraer el texto de la placa
            text = pytesseract.image_to_string(placa, config='--psm 11')
            
            # Mostrar el texto de la placa y la placa recortada en una ventana
            print('PLACA: ', text)
            cv2.imshow('PLACA', placa)
            cv2.moveWindow('PLACA', 780, 10)
            
            # Dibujar un rectángulo alrededor de la placa y mostrar el texto
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(image, text, (x-20, y-10), 1, 2.2, (0, 255, 0), 3)

# Mostrar la imagen original con los resultados
cv2.imshow('Image', image)
cv2.moveWindow('Image', 45, 10)

# Esperar hasta que se presione una tecla
cv2.waitKey(0)
