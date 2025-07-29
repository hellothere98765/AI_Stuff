import torch #type:ignore
import torchvision #type:ignore
from torchvision import transforms #type:ignore
from torch import nn #type:ignore
import time
import matplotlib.pyplot as plt #type:ignore
torch.manual_seed(2357)
#fff=0

storage="Datasets"
class EMNIST(): 
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.batch_size=batch_size #Makes batch sizes what's necessary

        self.resize=(28,28)
        
        
        #First resizes something to 28, 28, then converts whatever comes out as a tensor. The two ending lambdas are because the emnist database is for some reason flipped weirdly.
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), transforms.Lambda(lambda x: x.rot90(1, [1, 2])), transforms.Lambda(lambda x: x.flip(1))]) 
        
        self.train = torchvision.datasets.EMNIST(
            root=storage, split="byclass", train=True, transform=trans, download=True) #Opens the fashionmnist database, downloads it if necessary, and pipes it into a 28x28 tensor. 
        
        self.eval = torchvision.datasets.EMNIST(
            root=storage, split="byclass", train=False, transform=trans, download=True)#Same as above, but since it's evaluating here it
    
    def labels(self, indices):
        labels="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return [labels[int(i)] for i in indices] #Returns the labels based on the classification in the database
    
    def get_dataloader(self, train, num_workers=0):
        if train:
            used_data=self.train
        else:
            used_data=self.eval#Determines which dataset to use - eval or train.

        return torch.utils.data.DataLoader(used_data, self.batch_size, shuffle=train, num_workers=num_workers) #Shuffles dataset if training, otherwise not. 


class softmaxClassifier(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=62, lr=.01, sigma=.03):
        super().__init__()
        self.num_inputs=num_inputs #Here, number of pixels.
        self.num_outputs=num_outputs #Here, number of digits

        self.lr=lr #Learning rate
        self.sigma=sigma #Std dev on torch rng 

        self.weights = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)#Sets up weights
        self.biases=torch.normal(0, sigma, size=(num_outputs,), requires_grad=True) #Sets up biases

        self.loss_func=nn.NLLLoss() #Using NLLLoss, as cross-entropy loss in pytorch does softmax automatically on the input logits..
        self.softmax=nn.LogSoftmax(dim=1)#NLLLoss expects log probabilities, so - the softmax needs to be logged.

    def parameters(self):
        return [self.weights, self.biases]
    
    def forward(self, X):
        X=X.reshape((-1, self.weights.shape[0])) #Flattens input so that it can be multiplied by the weights - the -1 is there so we can change the batch size.
        #print(X.shape, self.weights.shape)
        return self.softmax(X@self.weights + self.biases) #Applies softmax to the output.
    
    def loss(self, predicted, actual):
        k=self.loss_func(predicted, actual) #Plugs the predicted and actual values into the loss function.
        #global fff
        #fff+=1
        #if (fff%1000==0):
        #    print(float(k)) #A sanity check to make sure the loss is going down.
        return k
        #Aka -torch.log(predicted[list(range(len(predicted))),actual]).mean(). This finds the correctly labeled probability, combines them, and then takes the mean. 
    
    def training_step(self, data):
        inputs, labels=data 
        return self.loss(self.forward(inputs), labels)#
    
    def optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
class Trainer:
    def __init__(self, model, data, max_epochs=6):
        self.model=model 
        self.data=data
        self.optim=self.model.optimizers() #Convenience
        self.max_epochs=max_epochs #Max number of iterations to go through
        self.train_batch_id=0
        
    def fit(self,num_data=1000):
        for self.epoch in range(self.max_epochs):
            print(f"On epoch {self.epoch}")
            self.fit_epoch(num_data)

    def fit_epoch(self, num_data):#TODO - num_data isn't being used here - it might be beneficial to figure out a way to only take like 10000 samples from the dataloader at a time. Maybe not though.
        self.model.train()#Enables training on our model (Part of nn.Module's functionalities)
        for batch in self.data: #DO NOT change this into a "for i in range" thing - it makes the code like 3x slower.
            loss=self.model.training_step(batch) #Find the mean loss from the batch
            self.optim.zero_grad()#Zero out the gradient arrays
            loss.backward()#This can be inside or outside the no_grad - it lets you do backwards gradient finding.
            self.optim.step()
            self.train_batch_id+=1

model=softmaxClassifier()
data=EMNIST()
training_data=data.get_dataloader(train=True, num_workers=0)#Lol the more cores I use the slower it gets - 0 cores gave me a time of 2.6 seconds for 50, while 2 gave me 5.6.
trainer = Trainer(model, training_data)
trainer.fit()


#Grapher
model.eval()
fig, axes = plt.subplots(8, 8, figsize=(8, 8))#makes an 8x8 grid
eval_data=data.get_dataloader(train=False)#Gets evaluation data
(pictures, actual_labels)=next(iter(eval_data))
actual_labels=data.labels(actual_labels)
pred_labels=model.forward(pictures)
_, pred_labels=torch.max(pred_labels, dim=1)#Since the output from the model is in the form of a bunch of probabilities, this chooses the highest one in each vector.
pred_labels=data.labels(pred_labels.tolist())
for i in range(64):
    picture=pictures[i]
    actual_label=actual_labels[i]
    pred_label=pred_labels[i]
    ax=axes[i//8,i%8]
    ax.imshow(picture.squeeze(), cmap='gray')
    ax.axis('off')
    ax.set_title(str((pred_label, actual_label)))

fig.tight_layout()
fig.savefig("EMNISTTest.png")


