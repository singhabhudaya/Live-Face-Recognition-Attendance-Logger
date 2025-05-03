% Load the dataset
imds = imageDatastore('dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
augimds = augmentedImageDatastore([227 227 3], imds);

% Load pretrained AlexNet
net = alexnet;
featureLayer = 'fc7';

% Extract features
features = activations(net, augimds, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
labels = imds.Labels;

% Train SVM classifier
classifier = fitcecoc(features, labels);

% Save the trained model for future use
save('faceClassifier.mat', 'classifier', 'net');

disp('✅ faceClassifier.mat saved successfully.');
net = alexnet;