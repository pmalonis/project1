require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim' 


dofile('Model.lua') -- defining the two-layer perceptron and the loss function

opt = {}
opt.learningRate = 1e-4
opt.batchSize = 30
opt.weightDecay = 1e-2
opt.momentum = 0.9
opt.saveflag = true
opt.save = 'results' -- subdirectory to save model in
-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Retrieve parameters and gradients:
if model then
   parameters,gradParameters = model:getParameters()
end


optimState = nil
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  dampening = 0.0,
  learningRateDecay = 0
}
optimMethod = optim.sgd
opt.type = 'float'


function train() -- A function that handles the training and report the accuracy on the training set
	-- epoch tracker
	epoch = epoch or 1
	-- local vars
	local time = sys.clock()
	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training()
	-- shuffle at each epoch
	shuffle = torch.randperm(trsize)
	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,trainData:size(),opt.batchSize do
		-- disp progress
		xlua.progress(t, trainData:size())
		-- create mini batch
		local inputs = torch.Tensor(math.min(t+opt.batchSize,trainData:size()+1)-t, trainData.data:size()[2])
		local targets = torch.Tensor(math.min(t+opt.batchSize,trainData:size()+1)-t)
		count=0
		for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
			count=count+1
			-- load new sample
			inputs[{{count},{}}] = trainData.data[shuffle[i]]
			targets[count] = trainData.labels[shuffle[i]]
		end
		if opt.type == 'double' then 
			inputs = inputs:double()
			targets = targets:double()
        elseif opt.type == 'cuda' then 
			inputs = inputs:cuda()
			targets = targets:cuda()			
		else 
			inputs = inputs:float() 
			targets = targets:float()
		end
		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end
			-- reset gradients
			gradParameters:zero()
			-- f is the average of all criterions
			local f = 0
			-- evaluate function for complete mini batch
			local output = model:forward(inputs);
			local df_do = criterion:backward(output, targets);
			model:backward(inputs, df_do);
			f = criterion:forward(output, targets)
			for i = 1,inputs:size()[1] do
				confusion:add(output[i], targets[i])
			end
			return f,gradParameters
		end
		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else
			optimMethod(feval, parameters, optimState)
		end
	end
	-- time taken
	time = sys.clock() - time
	time = time / trainData:size()
	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
	-- print confusion matrix
	print(confusion)
	if opt.saveflag then
		local filename = paths.concat(opt.save, 'model.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		print('==> saving model to '..filename)
		torch.save(filename, model) -- saving the model after each epoch
	end
	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end
