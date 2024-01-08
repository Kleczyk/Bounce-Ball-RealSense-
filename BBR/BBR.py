import cv2
import numpy as np

# Funkcja do wykrywania piłki
def detect_ball(frame):
    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binaryzacja adaptacyjna
    # binary_adaptive = cv2.adaptiveThreshold(
    #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    # Opcjonalnie: zastosowanie rozmycia Gaussa do redukcji szumów
    # blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Wykrywanie okręgów za pomocą transformacji Hougha
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=200,
        param2=10,
        minRadius=10,
        maxRadius=200,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            # Zwróć pozycję pierwszego wykrytego okręgu
            return (x, y)

    return None  # Jeśli nie wykryto okręgów, zwróć None


# Funkcja do śledzenia piłki
def track_ball(ball_position, last_position):
    # Jeśli któraś z pozycji jest None, nie możemy wykryć zmiany kierunku
    if ball_position is None or last_position is None:
        return False

    # Obliczenie różnicy w położeniu między bieżącą a ostatnią pozycją
    delta_x = ball_position[0] - last_position[0]
    delta_y = ball_position[1] - last_position[1]

    # Ustalenie progu, aby określić, czy nastąpiła znacząca zmiana kierunku
    # Prog ten można dostosować w zależności od wymagań
    threshold = 100

    # Wykrywanie odbicia
    # Odbicie może być wykryte na podstawie znacznej zmiany w pionowym kierunku (delta_y)
    if abs(delta_y) > threshold:
        return True
    else:
        return False


# Inicjalizacja kamery
cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print("Aktualna wartość ekspozycji:", exposure)
# Ustawienie liczby klatek na sekundę
cap.set(cv2.CAP_PROP_FPS, 30)


last_position = None
bounce_count = 0



while True:
    ret, frame = cap.read()
    if not ret:
        break

    ball_position = detect_ball(frame)

    if ball_position is not None:
        # Rysowanie krzyżyka na piłce
        cv2.drawMarker(
            frame, ball_position, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2
        )

        bounce_detected = track_ball(ball_position, last_position)
        if bounce_detected:
            bounce_count += 1

    last_position = ball_position

    # Adaptacyjna binaryzacja
 
    # Wyświetlanie liczby odbić
    cv2.putText(
        frame,
        f"Bounces: {bounce_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
