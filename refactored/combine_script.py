
# Объединяем все части скрипта улучшения детектора

with open('intellectshield/refactored/improve_detector_part1.py', 'r') as f:
    part1 = f.read()

with open('intellectshield/refactored/improve_detector_part2.py', 'r') as f:
    part2 = f.read()

with open('intellectshield/refactored/improve_detector_part3.py', 'r') as f:
    part3 = f.read()

with open('intellectshield/refactored/improve_detector_part4.py', 'r') as f:
    part4 = f.read()

with open('intellectshield/refactored/improve_detector.py', 'w') as f:
    f.write(part1 + part2 + part3 + part4)

print("Скрипт улучшения детектора успешно собран!")
