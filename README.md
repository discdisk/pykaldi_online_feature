# pykaldi_online_feature

### *online.py*, *offline.py*

different in kaldi feature extraction, model behavior is both in offline processing

### cmvn.py

python implement of cmvn from [espnet](https://github.com/espnet/espnet/blob/master/espnet/transform/cmvn.py)

parameter file will be created after espnet data preparation,

saved in `espnet-master/egs/csj/asr1/data/train_nodup_sp/`

### socket_client.py, socket_server.py

client and server connect by socket connection.

**client side** simply collect audio data using pyaudio and send it to server.

**server side** accept data and preprocess it with kaldi online feature extraction, then feed to Model.

### Model

In order to simplify the streaming processing,

`subsample rate = 1/ sub`, `attention window = L`

then the input frame size every step should be = `N*sub>attention` (N is Integer)

### How to run

since pykaldi didn't support windows, the server side need to run on [WSL](https://www.microsoft.com/ja-jp/p/ubuntu-1804-lts/9n9tngvndl3q?rtc=1&activetab=pivot:regionofsystemrequirementstab),

and since WSL can't get audio device input, so the client side need to run on windows.

### Todo

- [ ] CNN model state return (streaming processing)
- [ ] beam search decoding



```sequence
client->server: send a chunk of audio data
server->pykaldi: online feature extraction
pykaldi->server: feature
server->Model: model states, feature
Model->server:model states, recognition result
Note over server: show result or send to client


```



