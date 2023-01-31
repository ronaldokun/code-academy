# Operações Bitwise - AND, OR, XOR, and NOT


# Imports
import numpy as np
import cv2

# Primeiro, desenhamos um retângulo
rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Retangulo", rectangle)

# Em segundo lugar, vamos desenhar um círculo
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circulo", circle)

# A função bitwise_and examina cada pixel no retângulo e círculo. 
# Se ambos os pixels tiverem um valor maior do que zero, esse pixel é ativado 'ON' 
# (i.e configurado para 255 na imagem de saída). 
# Se ambos os pixels não são maiores do que zero, então o pixel de saída é deixado 'OFF' com um valor de 0.
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)

# A função bitwise_or examina cada pixel no retângulo e círculo. 
# Se qualquer pixel no retângulo ou círculo for maior do que zero, 
# então o pixel de saída tem um valor de 255, caso contrário, é 0.
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)

# A função bitwise_xor é idêntica à função 'OR'
# com uma exceção: tanto o retângulo como o círculo são
# não permitidos ter valores maiores que 0.
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)

# Finalmente, a função bitwise_not inverte os valores dos pixels. 
# Pixels com um valor de 255 tornam-se 0,
# e pixels com um valor de 0 tornam-se 255.
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey(0)