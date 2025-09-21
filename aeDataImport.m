%%%%%% Import the ape call data to Matlab

% Inputs to this script: original ASCII files apecalls.train and apecalls.test
% downloaded from XXX (will be disclosed at end of project)
% Data have been donated by XXX (will be disclosed at end of project)

% This script assigns global Matlab cell arrays trainInputs, testInputs,
% trainOutputs, testOutputs with cell sizes (270,1), (370,1), (270,1),
% (370,1) respectively. Each cell contains the corresponding
% multidimensional time series (inputs 12 dim, outputs 9 dim)



load ae.train -ascii;
aeTrain = ae;
load ae.test -ascii;
aeTest = ae;

% aeTrain and aeTest contain the 12-dim time series, which have
% different lengthes, concatenated vertically and separated by ones(1,12)
% rows. We now sort them into cell arrays, such that each cell represents
% one time series
trainInputs = cell(270,1);
readindex = 0;
for c = 1:270
    readindex = readindex + 1;
    l = 0;    
    while aeTrain(readindex, 1) ~= 1.0
        l = l+1;
        readindex = readindex + 1;
    end
    trainInputs{c,1} = aeTrain(readindex-l:readindex-1,:);    
end

testInputs = cell(370,1);
readindex = 0;
for c = 1:370
    readindex = readindex + 1;
    l = 0;    
    while aeTest(readindex, 1) ~= 1.0
        l = l+1;
        readindex = readindex + 1;
    end
    testInputs{c,1} = aeTest(readindex-l:readindex-1,:);    
end

% produce teacher signals. For each input time series of size N x 12 this
% is a time series of size N x 9, all zeros except in the column indicating
% the speaker, where it is 1.
trainOutputs = cell(270,1);
for c = 1:270
    l = size(trainInputs{c,1},1);
    teacher = zeros(l,9);
    speakerIndex = ceil(c/30);
    teacher(:,speakerIndex) = ones(l,1);
    trainOutputs{c,1} = teacher;
end

testOutputs = cell(370,1);
speakerIndex = 1;
blockCounter = 0;
blockLengthes = [31 35 88 44 29 24 40 50 29];
for c = 1:370
    blockCounter = blockCounter + 1;
    if blockCounter == blockLengthes(speakerIndex)+ 1
        speakerIndex = speakerIndex + 1;
        blockCounter = 1;
    end
    l = size(testInputs{c,1},1);
    teacher = zeros(l,9);    
    teacher(:,speakerIndex) = ones(l,1);
    testOutputs{c,1} = teacher;
end


