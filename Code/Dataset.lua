--require 'mattorch'
matio = require 'matio'

subset = 'train'

filePath = '/share/project/vision-winter16/voc12-' .. subset ..'.mat'


loaded = matio.load(filePath)  -- Loading lables and image names
imlist = loaded.Imlist
 t = {}
 s = {}
for i = 1, imlist:size()[1] do
  for j = 1, imlist:size()[2]  do
    t[j] = string.char(imlist[{i,j}])
  end
  s[i]=table.concat(t);
end

imlist = s
numimages = #imlist
labels = loaded.labels


datasetfeats = torch.Tensor(numimages,4096) -- Represent each image with 4096-dimensional feature vector
datasetfeats = torch.load('/share/project/vision-winter16/datasetfeats-voc12-' .. subset .. '.t7')


if subset == 'train' then
	trsize = datasetfeats:size()[1]
	trainData = {
	   data = datasetfeats,
	   labels = labels,
	   size = function() return trsize end
	}
elseif subset == 'val' then
	trsize = datasetfeats:size()[1]
	valData = {
	   data = datasetfeats,
	   labels = labels,
	   size = function() return trsize end
	}
elseif subset == 'test' then
	trsize = datasetfeats:size()[1]
	testData = {
	   data = datasetfeats,
	   size = function() return trsize end
	}
else 
	error("invalid subset")
end

