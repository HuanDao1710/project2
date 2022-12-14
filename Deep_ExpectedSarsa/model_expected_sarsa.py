import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size = 18, hidden_size1 = 384, hidden_size2 = 384, output_size = 3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size1)
        #self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
       


    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


    def save(self, file_name='v1.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class QTrainer:
	def __init__(self,model,lr,gamma):
		#Learning Rate for Optimizer
		self.lr = lr
		#Discount Rate
		self.gamma = gamma
		#Linear NN defined above.
		self.model = model
		#optimizer for weight and biases updation
		self.optimer = optim.Adam(model.parameters(),lr = self.lr)
		#Mean Squared error loss function
		self.criterion = nn.MSELoss()

		self.epsilon = 0.1

	
	def train_step(self,state,action,reward,next_state,done):
		state = torch.tensor(state,dtype=torch.float)
		next_state = torch.tensor(next_state,dtype=torch.float)
		action = torch.tensor(action,dtype=torch.long)
		reward = torch.tensor(reward,dtype=torch.float)

		# if only one parameter to train , then convert to tuple of shape (1, x)
		if(len(state.shape) == 1):
			#(1, x)
			state = torch.unsqueeze(state,0)
			next_state = torch.unsqueeze(next_state,0)
			action = torch.unsqueeze(action,0)
			reward = torch.unsqueeze(reward,0)
			done = (done, )

		# 1. Predicted Q value with current state
		pred = self.model(state)
		target = pred.clone()
		for idx in range(len(done)):
			Q_new = reward[idx]	
			if not done[idx]:
				list_actions = self.model(next_state[idx]).detach().numpy()
				max_action = np.max(list_actions)
				index = np.where(list_actions == max_action)
				prob = torch.ones(3) * self.epsilon / 3
				prob[index] += 1 - self.epsilon
        		
				Q_new = reward[idx] + self.gamma *  np.dot(prob, list_actions)
			target[idx][torch.argmax(action[idx]).item()] = Q_new
		# 2. Q_new = reward + gamma * max(next_predicted Qvalue)
		#pred.clone()
		#preds[argmax(action)] = Q_new
		self.optimer.zero_grad()
		loss = self.criterion(target,pred)
		loss.backward() # backward propagation of loss

		self.optimer.step()



