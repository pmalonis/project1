require 'nn'

-- Define the two-layer perceptron and the loss function.
-- Your code goes here 
nInput = 4096
nHidden = 1000
nOutput = 20

dropoutRatio = 0.5

model = nn.Sequential()
model:add(nn.Reshape(nInput))
model:add(nn.Linear(nInput,nHidden))
model:add(nn.ReLU(true))
model:add(nn.Dropout(dropoutRatio))
model:add(nn.Linear(nHidden, nOutput))

model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()



