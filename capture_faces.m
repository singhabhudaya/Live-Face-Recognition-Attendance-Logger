cam = webcam;
faceDetector = vision.CascadeObjectDetector();
userName = 'USER 1';  % CHANGE THIS for each person
mkdir(fullfile('dataset', userName));

for i = 1:15
    img = snapshot(cam);
    bbox = step(faceDetector, img);
    if ~isempty(bbox)
        face = imcrop(img, bbox(1,:));
        face = imresize(face, [227 227]);  % Resize for CNN
        filename = fullfile('dataset', userName, sprintf('%d.jpg', i));
        imwrite(face, filename);
        imshow(face); title(['Captured Image ' num2str(i)]);
        pause(0.3);
    end
end

clear cam

cam = webcam;
faceDetector = vision.CascadeObjectDetector();
userName = 'USER 2';  % CHANGE THIS for each person
mkdir(fullfile('dataset', userName));

for i = 1:15
    img = snapshot(cam);
    bbox = step(faceDetector, img);
    if ~isempty(bbox)
        face = imcrop(img, bbox(1,:));
        face = imresize(face, [227 227]);  % Resize for CNN
        filename = fullfile('dataset', userName, sprintf('%d.jpg', i));
        imwrite(face, filename);
        imshow(face); title(['Captured Image ' num2str(i)]);
        pause(0.3);
    end
end

clear cam
imds = imageDatastore('dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

countEachLabel(imds)