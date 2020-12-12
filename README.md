Client.py defines the client side partial model
Server.py defines the server side partial model

client.ipynb is the client side program. 
server.ipynb is the server side program. 
GPU_CPU_compare.ipynb is training time comparison in different model complexity in GPU vs. CPU setting.

1. Run server.ipynb first three cells' codes to create a server,
2. Run client.ipynb codes to generate a sample dataset and create two clients and do the model partial
training, connect to the server and send the intermediate values to server.

If you test the GPU training codes in server.ipynb, please use tensorflow version >= 2.0
