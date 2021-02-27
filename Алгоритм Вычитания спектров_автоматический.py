import numpy as np
import scipy
import librosa
from scipy.io.wavfile import write

# edit following wav file name
infile="C:\\Users\\PycharmProjects\\MIAODI\\venv\\files\\Чистка_Y_short_автоматизированная\\Y.wav"
outfile="C:\\Users\\PycharmProjects\\MIAODI\\venv\\files\\Чистка_Y_short_автоматизированная\\Y.wav"


# load input file, and stft (Short-time Fourier transform)
print ('load wav', infile)
w, sr = librosa.load( infile, sr=None, mono=True) # Загрузите аудиофайл как временной ряд с плавающей запятой.
                                                  # сохранение частоты дискретезации, заложенной в записи
                                                  # (default sr=22050).
s= librosa.stft(w) # Кратковременное преобразование Фурье
# STFT представляет сигнал в частотно-временной области путем вычисления дискретных преобразований Фурье (ДПФ) в коротких перекрывающихся окнах.
ss= np.abs(s) # Модуль сигнала (абсолютное значение)
angle= np.angle(s) # Получим угол (фазу) (Вычисление фазового спектра)
b=np.exp(1.0j* angle) # использовать эту информацию о фазе при обратном преобразовании

# Загруцим оцененный шум, проведем  Кратковременное преобразование Фурье, возьмем среднее
# print ('Read only noise')
nw=w[0:10000] # участок, содержащий только шум
nsr=sr
ns= librosa.stft(nw)
nss= np.abs(ns)
mns= np.mean(nss, axis=1) #вычисляет среднее арифметическое значений элементов массива.

print(sr, nsr)
# вычтем среднее спектральное значение шума из входного спектрального и проведем istft (обратное кратковременное преобразование Фурье)
sa= ss - mns.reshape((mns.shape[0],1)) # изменяет форму массива без изменения его данных( необходимо для вычитанция) (если в кратце-делаем столбцом))
sa0= sa * b # применяем фазовую информацию
y= librosa.istft(sa0) # назад к сигналу временной области (обратное преобразование Фурье)

# # save as a WAV file
scipy.io.wavfile.write(outfile, sr, (y * 32768).astype(np.int16)) # сохранить в формате WAV и количество бит на отсчет=16
#librosa.output.write_wav(outfile, y , sr) # save 32-bit floating-point WAV format, due to y is float
print ('write wav', outfile)
