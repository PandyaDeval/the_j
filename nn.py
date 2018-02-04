import numpy as np
import csv

#X= np.array([[1,1],[1,0],[0,0]])
#Y= np.array([[0],[1],[0]])
X= np.genfromtxt("ocr_train_input.csv", delimiter= ",")
W= np.genfromtxt("ocr_train_output.csv", delimiter=",")
X/=100

Y= np.zeros(shape=(1000,10))

for i in range(1000):
	if (W[i]==0):
		Y[i]=[1,0,0,0,0,0,0,0,0,0]
	elif (W[i]==1):
		Y[i]=[0,1,0,0,0,0,0,0,0,0]
	elif (W[i]==2):
		Y[i]=[0,0,1,0,0,0,0,0,0,0]
	elif (W[i]==3):
		Y[i]=[0,0,0,1,0,0,0,0,0,0]
	elif (W[i]==4):
		Y[i]=[0,0,0,0,1,0,0,0,0,0]
	elif (W[i]==5):
		Y[i]=[0,0,0,0,0,1,0,0,0,0]
	elif (W[i]==6):
		Y[i]=[0,0,0,0,0,0,1,0,0,0]
	elif (W[i]==7):
		Y[i]=[0,0,0,0,0,0,0,1,0,0]
	elif (W[i]==8):
		Y[i]=[0,0,0,0,0,0,0,0,1,0]
	elif (W[i]==9):
		Y[i]=[0,0,0,0,0,0,0,0,0,1]
		


input_layerSize= X.shape[1]
hidden_layerSize= 30
output_layerSize= Y.shape[1]
epoch =3000
lr= 0.0003


def sigmoid(x):
	np.clip(x, -1000, 1000)
	return 1/(1+np.exp(-x))
	
def derivative_sigmoid(x):
	return x*(1-x)
	
def cost(y, t):
	return np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

wh=  np.random.normal(scale= 0.1, size= (input_layerSize, hidden_layerSize))
wout= np.random.normal(scale= 0.1, size= (hidden_layerSize, output_layerSize))
bh= np.random.normal(scale=0.1, size= (1, hidden_layerSize))
bout= np.random.normal(scale= 0.1, size= (1, output_layerSize))
#print(wh)
#print(X.shape)
#print(wh.shape)

for i in range(epoch):
	
	#forward feedback
	hidden_layer_input1= np.dot(X, wh)
	hidden_layer_input= hidden_layer_input1 + bh
	hidden_layer_activation = sigmoid(hidden_layer_input)
	output_layer_input1 = np.dot(hidden_layer_activation, wout)
	output_layer_input = output_layer_input1+  bout
	output= sigmoid(output_layer_input)

	#back propogation
	E= Y- output
	error= np.sum(E**2)/1000
	print("Error at %d = %d", i, error)
	d_output= E* derivative_sigmoid(output)
	wout+= hidden_layer_activation.T.dot(d_output)*lr
	error_hidden_layer= d_output.dot(wout.T)
	d_hidden_layer= error_hidden_layer* derivative_sigmoid(hidden_layer_activation)
	wh+= X.T.dot(d_hidden_layer)*lr
	bout= bout+ np.sum(d_output, axis= 0)*lr
	bh+= np.sum(d_hidden_layer, axis=0)* lr
	
X= np.genfromtxt("ocr_test_input.csv", delimiter= ",")
W= np.genfromtxt("ocr_test_output.csv", delimiter= ",")
X/=100

Y= np.zeros(shape=(550,10))

for i in range(550):
	if (W[i]==0):
		Y[i]=[1,0,0,0,0,0,0,0,0,0]
	elif (W[i]==1):
		Y[i]=[0,1,0,0,0,0,0,0,0,0]
	elif (W[i]==2):
		Y[i]=[0,0,1,0,0,0,0,0,0,0]
	elif (W[i]==3):
		Y[i]=[0,0,0,1,0,0,0,0,0,0]
	elif (W[i]==4):
		Y[i]=[0,0,0,0,1,0,0,0,0,0]
	elif (W[i]==5):
		Y[i]=[0,0,0,0,0,1,0,0,0,0]
	elif (W[i]==6):
		Y[i]=[0,0,0,0,0,0,1,0,0,0]
	elif (W[i]==7):
		Y[i]=[0,0,0,0,0,0,0,1,0,0]
	elif (W[i]==8):
		Y[i]=[0,0,0,0,0,0,0,0,1,0]
	elif (W[i]==9):
		Y[i]=[0,0,0,0,0,0,0,0,0,1]
		



hidden_layer_input1= np.dot(X, wh)
hidden_layer_input= hidden_layer_input1 + bh
hidden_layer_activation = sigmoid(hidden_layer_input)
output_layer_input1 = np.dot(hidden_layer_activation, wout)
output_layer_input = output_layer_input1+  bout
output= sigmoid(output_layer_input)

c=0
for i in range(550):
	for j in range(10):
		if(output[i][j]== np.amax(output[i])):
			if(j==W[i]):
				c+=1
			break
print(c/(214*5))
			

