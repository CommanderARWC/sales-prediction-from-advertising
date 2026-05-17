import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

df=pd.read_csv("advertising.csv")

#input and output
Y = df["Sales"].iloc[0:100]
X = df.drop("Sales", axis=1).iloc[0:100]
Y_TEST = df["Sales"].iloc[100:]
X_TEST = df.drop("Sales", axis=1).iloc[100:]


#Converting pandas data into numpy
Y = Y.values.astype(np.float32)
X = X.values.astype(np.float32)
Y_TEST = Y_TEST.values.astype(np.float32)
X_TEST = X_TEST.values.astype(np.float32)

#Converting it into tensors
X=torch.from_numpy(X)
Y=torch.from_numpy(Y)
X_TEST=torch.from_numpy(X_TEST)
Y_TEST=torch.from_numpy(Y_TEST)

Y=Y.view(-1,1)
Y_TEST=Y_TEST.view(-1,1)

class linearRegression(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear=nn.Linear(X.shape[1], 1)
	def forward(self,x):
		prediction=self.linear(x)
		return prediction

model=linearRegression()
loss_function=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
epochs=1000

for i in range(epochs):
	prediction=model(X)
	loss=loss_function(prediction,Y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if i%100==0:
		print(loss.item())



with torch.no_grad():
    predicted_value=model(X_TEST[0])
    print(predicted_value)
    print(Y_TEST[0])




















