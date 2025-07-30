import torch #type: ignore
from torch import nn # type: ignore
torch.manual_seed(420)

class functiondataload:
    def __init__(self, func, noise, batch_size=32, train_len=100): # Initializes the dataloader
        self.func=func #Defines which function we're trying to approximate
        self.noise=noise 
        self.batch_size=batch_size #Defines the batch sizes - so the data output from train_dataloader would be of shape (32, 1) here.
        self.train_len=train_len #Not necessary here, but for the sake of argument how many training samples we have. I've honestly put it in wrong here.

    def train_dataloader(self):
        data=(2*torch.rand(self.batch_size*self.train_len)-1)
        data=data.reshape(-1, 1)
        labels=self.func(data)+self.noise*torch.randn(data.shape)
        training_set=torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(training_set, self.batch_size, shuffle=True)#Same as above, but uses the dataloader/dataset utilities. 
    

class LinearRegressor(nn.Module):
    def __init__(self,lr):
        super().__init__() #Initializes the super class
        self.lr=lr #Learning rate
        
        #The following two are identical for this use case.
        #self.net=nn.LazyLinear(1) #Linear layer. Lazy means I don't have to specify the input dimensions.
        self.net=nn.Linear(1, 1)

        self.net.weight.data.normal_(0, .01)#Initializes weights with random
        self.net.bias.data.fill_(0) #Initializes biases to 0

    def forward(self, X):
        return self.net(X)#Built in function - we can just do this.
    
    def loss(self, predicted, actual):
        loss_fn=nn.MSELoss()
        return loss_fn(predicted, actual)#Mean Squared Error loss.

    def training_step(self, data):
       inputs, labels=data
       #Uses the forward to calculate the loss
       return self.loss(self.forward(inputs), labels)

    def optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr) #Stochastic gradient descent. self.parameters() gives us the parameters of the net.
    
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
            self.optim.step()#Since we're not doing this weirdly like in the from scratch implementation, we don't need torch.nograd.
            self.train_batch_id+=1

model=LinearRegressor(.01)

data=functiondataload(nn.Identity(), .02)

trainer = Trainer(model, data.train_dataloader())

trainer.fit()

while(True):
    print(next(iter(model.parameters())))
#with torch.no_grad():
#    print(f"{list(model.parameters())}")
#    for k in torch.arange(-1, 1, 0.05):
#        print(f"({k}, {model.forward(torch.Tensor([k])).item()})")

#WORKING EXAMPLE :3
#Most of the changes were done to the model and calls to the model. Everything else is kinda the same.