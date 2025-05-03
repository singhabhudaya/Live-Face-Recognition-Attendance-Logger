clc;
clear;
load('faceClassifier.mat');  % Load the trained model and AlexNet
x4
cam = webcam;
faceDetector = vision.CascadeObjectDetector();

attendanceLog = {};  % To avoid duplicate marking
fprintf('Starting webcam... Press Ctrl+C to stop\n');

while true
    img = snapshot(cam);
    bbox = step(faceDetector, img);

    if ~isempty(bbox)
        face = imcrop(img, bbox(1,:));
        face = imresize(face, [227 227]);

        % Extract features
        features = activations(net, face, 'fc7', 'OutputAs', 'rows');
        label = predict(classifier, features);

        % Annotate and show result
        annotatedImg = insertObjectAnnotation(img, 'rectangle', bbox, char(label), 'LineWidth', 2, 'Color', 'green');
        imshow(annotatedImg);
        title(['Recognized: ' char(label)]);

        % Log attendance if not already present
        if ~any(strcmp(attendanceLog, label))
            attendanceLog{end+1} = char(label);
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            fid = fopen('attendance.csv','a');
            fprintf(fid, '%s,%s\n', char(label), timestamp);
            fclose(fid);
            disp(['✅ Attendance marked for: ' char(label)]);
        end
    else
        imshow(img);
        title('No Face Detected');
    end

    pause(0.3);  % Slight delay for smooth loop
end

clear cam