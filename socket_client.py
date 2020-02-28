
import numpy as np
import pyaudio
import socket               # 导入 socket 模块
import keyboard

s = socket.socket()         # 创建 socket 对象
host = socket.gethostname()  # 获取本地主机名
port = 12345                # 设置端口号

s.connect((host, port))
# print(bytes.decode(s.recv(1024)))



CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
sr = RATE = 16000
RECORD_SECONDS = 100



p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
result = []
last_num = 0


mark=True
states = None
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

    data = stream.read(CHUNK)
    
    if keyboard.is_pressed('space'):

        frames.append(data)

        # audiodata = np.frombuffer(data, dtype=np.int16).tobytes()

        print('sending+' if mark else 'sending-', end='\r')
        s.send(data)
        mark = not mark

s.close()
