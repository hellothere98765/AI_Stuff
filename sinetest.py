import torch #type: ignore
from torch import nn # type: ignore
import random

torch.manual_seed(420)


class functiondataload:
    def __init__(self, func, noise, batch_size=32, train_len=100): # Initializes the dataloader
        self.func=func #Defines which function we're trying to approximate
        self.noise=noise 
        self.batch_size=batch_size #Defines the batch sizes - so the data output from train_dataloader would be of shape (32, 1) here.
        self.train_len=train_len #Not necessary here, but for the sake of argument how many training samples we have. I've honestly put it in wrong here.

    def manual_train_dataloader(self):
        for _ in range(0, self.train_len):
            data=(2*torch.rand(self.batch_size)-1)#Random numbers between -1 and 1
            labels=self.func(data)+self.noise*torch.randn(data.shape) #Actual labels, with a little bit of error.
            yield data, labels #Generator function
    
    def train_dataloader(self):
        data=(2*torch.rand(self.batch_size*self.train_len)-1)
        data=data.reshape(-1, 1)
        labels=self.func(data)+self.noise*torch.randn(data.shape)
        training_set=torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(training_set, self.batch_size, shuffle=True)#Same as above, but uses the dataloader/dataset utilities. 
    
class linearRegressor(nn.Module):
    def __init__(self, inputs, lr, sigma=.05):
        super().__init__()
        self.inputs=inputs#Number of inputs - like for a linear function trying to approximate e^x, it'd be 1.
        self.lr=lr#Learning rate
        self.sigma=sigma #The std dev on the rngs.


        #Initial weights and biases.
        self.w = torch.normal(0, sigma, (inputs, 1), requires_grad=True)
        self.b=torch.zeros(1, requires_grad=True)
    
    def forward(self, inputs):
        #Calculates the output from the neural net. It's pretty simple here. 
        #I tried putting in a 2nd weight and bias, but there's not really a point - since there's one input and each layer so far has 1 neuron, it's literally just a linear function.
        #inputs and weights have different lengths - broadcasting is what makes them be able to be multiplied like this.
        return inputs@self.w + self.b
    
    def loss(self, predicted, actual):
        #Squared loss - works cause we're using normal error.
        loss_vector=((predicted-actual)**2) /2
        return loss_vector.mean()

    def training_step(self, data):
        inputs, labels=data
        #Uses the forward to calculate the loss
        return self.loss(self.forward(inputs), labels)
    
    def optimizers(self):
        #Defines our optimizer using the class below.
        return GradDescent([self.w, self.b], self.lr)



class GradDescent:
    def __init__(self, params, lr):
        self.params=params#Tells us what parameters we need to optimize. We could not include some, but there's not too much of a point.
        self.lr=lr #Gives the learning rate.
    
    def step(self):
        for param in self.params:
            param-=self.lr*param.grad#Gradient descent step.
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()#Zeros out the gradient array of each parameter so that the thing can restart. 

class Trainer:
    def __init__(self, model, data, max_epochs=100):
        self.model=model 
        self.data=data
        self.optim=self.model.optimizers() #Convenience
        self.max_epochs=max_epochs #Max number of iterations to go through
        self.train_batch_id=0
    def fit(self):
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()#Enables training on our model (Part of nn.Module's functionalities)
        for batch in self.data:#For one set of inputs and labels
            loss=self.model.training_step(batch) #Find the mean loss from the batch
            self.optim.zero_grad()#Zero out the gradient arrays
            loss.backward()#This can be inside or outside the no_grad - it lets you do backwards gradient finding.
            with torch.no_grad(): 
                self.optim.step()#The specific way we did our optimization step, we mess with a parameter that has a gradient matrix. This isn't allowed in pytorch without the no_grad, which is why this is here.
            self.train_batch_id+=1

model=linearRegressor(1, .01)

data=functiondataload(torch.sin, .02)

trainer = Trainer(model, data.train_dataloader())

trainer.fit()

with torch.no_grad():
    print(f"{model.w} {model.b}")
    for k in torch.arange(-1, 1, 0.05):
        print(f"({k}, {model.forward(torch.Tensor([k])).item()})")

#WORKS!!!! THIS IS A WORKING EXAMPLE!!!!!
#It might only be able to take in a number and spit it back out but we take all W's