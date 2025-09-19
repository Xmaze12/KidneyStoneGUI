function kidneyStoneApp_noDL()
% KIDNEYSTONEAPP_NODL - GUI for kidney stone detection without Deep Learning Toolbox.
%
% This application builds a user interface and calls an external Python 
% script ('classify_image.py') to classify ultrasound images. It then
% performs image segmentation using only MATLAB's Image Processing Toolbox.

% Clear workspace, command window, and close existing figures for a clean start
clear; clc; close all;

% --- GUI SETUP ---
% Create the main figure window for the application
fig = figure('Name', 'Kidney Stone Detection GUI (Hybrid Model)', ...
             'Position', [200, 200, 800, 600], ...
             'NumberTitle', 'off', ...
             'MenuBar', 'none', ...
             'Color', [0.94 0.94 0.94]);

% --- UI Components ---
% Axes to display the ultrasound image and results
h.axes = axes('Parent', fig, 'Position', [0.05, 0.15, 0.7, 0.8]);
title(h.axes, 'Please Load an Ultrasound Image');
set(h.axes, 'XTick', [], 'YTick', []);

% Button to load and analyze an image
h.loadButton = uicontrol('Parent', fig, 'Style', 'pushbutton', ...
                         'String', 'Load & Analyze Image', ...
                         'Position', [580, 500, 200, 40], ...
                         'FontSize', 10, ...
                         'Callback', @loadImageCallback);

% Static text labels to display the results
uicontrol('Parent', fig, 'Style', 'text', ...
          'String', 'Classification Result:', ...
          'Position', [580, 440, 200, 25], ...
          'HorizontalAlignment', 'left', ...
          'FontSize', 11, 'FontWeight', 'bold');

h.resultText = uicontrol('Parent', fig, 'Style', 'text', ...
                         'String', 'N/A', ...
                         'Position', [580, 410, 200, 25], ...
                         'HorizontalAlignment', 'left', ...
                         'FontSize', 12, 'ForegroundColor', 'blue');

uicontrol('Parent', fig, 'Style', 'text', ...
          'String', 'Stone Diameter:', ...
          'Position', [580, 360, 200, 25], ...
          'HorizontalAlignment', 'left', ...
          'FontSize', 11, 'FontWeight', 'bold');

h.diameterText = uicontrol('Parent', fig, 'Style', 'text', ...
                           'String', 'N/A', ...
                           'Position', [580, 330, 200, 25], ...
                           'HorizontalAlignment', 'left', ...
                           'FontSize', 12, 'ForegroundColor', 'red');

% Store handles in the figure's UserData for access in the callback
fig.UserData.handles = h;
end

% --- CALLBACK FUNCTION for the "Load & Analyze Image" button ---
function loadImageCallback(src, ~)
% This function executes when the user clicks the button.

% Get handles from the figure's UserData
fig = src.Parent;
h = fig.UserData.handles;

% Open a file dialog to let the user select an image
[fileName, pathName] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'}, 'Select an Ultrasound Image');

% If the user clicks "Cancel", exit the function
if isequal(fileName, 0)
    return;
end
fullImagePath = fullfile(pathName, fileName);

% --- CLASSIFICATION STEP (Calling Python) ---
% Construct the command to call the python script.
% Note: Your 'python' executable must be in your system's PATH.
% Enclosing the image path in double quotes ("") is crucial to handle
% file paths that contain spaces.
command = ['python classify_image.py "' fullImagePath '"'];

% Update GUI to show that processing has started
set(h.resultText, 'String', 'Classifying...');
set(h.diameterText, 'String', 'N/A');
drawnow; % Force the GUI to update immediately

% Execute the system command and capture the text output
[status, cmdout] = system(command);

% Clean up the captured text (remove any trailing newline characters)
classificationResult = strtrim(cmdout);

% Check for errors from the Python script
if status ~= 0 || contains(classificationResult, 'Error', 'IgnoreCase', true)
    errordlg(['Python script execution failed. Output: ' classificationResult], 'Classification Error');
    set(h.resultText, 'String', 'Error');
    return;
end

% Update the result text field with the classifier's output
set(h.resultText, 'String', classificationResult);

% --- SEGMENTATION STEP (Conditional on classification) ---
imgToDisplay = imread(fullImagePath); % Read the image for processing/display

if strcmpi(classificationResult, 'Stone')
    % If the classifier found a stone, run the segmentation logic
    set(h.resultText, 'ForegroundColor', 'red'); % Set text color to red for "Stone"
    
    % --- IMPORTANT: CALIBRATION ---
    % Set this ratio based on your ultrasound machine's scale.
    % This is crucial for accurate real-world measurement.
    pixel_to_mm_ratio = 0.2; % Example value: 0.2 millimeters per pixel
    
    % Call the helper function to find, mark, and measure the stone
    [segmentedImage, diameterText] = segmentAndMeasureStone(fullImagePath, pixel_to_mm_ratio);
    
    % Display the final processed image and the calculated diameter
    imshow(segmentedImage, 'Parent', h.axes);
    title(h.axes, 'Stone Detected and Segmented');
    set(h.diameterText, 'String', diameterText);
else
    % If the classifier said "No Stone"
    set(h.resultText, 'ForegroundColor', 'green'); % Set text to green
    imshow(imgToDisplay, 'Parent', h.axes); % Display the original image
    title(h.axes, 'Classifier Result: No Stone Detected');
    set(h.diameterText, 'String', 'N/A'); % No diameter to show
end
end

% --- HELPER FUNCTION for Segmentation and Measurement ---
function [finalImage, diameter_text] = segmentAndMeasureStone(imagePath, pixel_to_mm_ratio)
% This function contains the MATLAB image processing code to locate, mark,
% and measure the most likely stone candidate within the kidney.

% --- Parameters for Segmentation ---
kidneyBinarizationThreshold = 0.3;
minKidneyArea = 5000;
stoneBrightnessPercentile = 0.99;
minStoneCandidateArea = 15;
weight_shape = 0.5;
weight_location = 0.3;
weight_size = 0.2;
diameter_text = 'Measurement Failed'; % Default text

% --- Load and Pre-process ---
originalImage = imread(imagePath);
if size(originalImage, 3) == 3, grayImage = rgb2gray(originalImage); else, grayImage = originalImage; end
enhancedImage = adapthisteq(grayImage);

% --- Segment the Kidney ---
binaryImage = imbinarize(enhancedImage, kidneyBinarizationThreshold);
binaryImage = bwareaopen(binaryImage, minKidneyArea);
se = strel('disk', 10);
kidneyMask = imclose(binaryImage, se);
kidneyMask = imfill(kidneyMask, 'holes');
[labels, ~] = bwlabel(kidneyMask);
regionProps = regionprops(labels, 'Area');
if ~isempty(regionProps)
    [~, maxIndex] = max([regionProps.Area]);
    kidneyMask = (labels == maxIndex);
else
    finalImage = originalImage;
    diameter_text = 'Kidney region not found';
    return;
end

% --- Score and Select Best Stone Candidate ---
kidneyROI = enhancedImage;
kidneyROI(~kidneyMask) = 0;
finalStoneMask = false(size(grayImage));

if any(kidneyROI(:))
    intensityValues = kidneyROI(kidneyROI > 0);
    stoneThreshold = quantile(double(intensityValues), stoneBrightnessPercentile);
    potentialStonesMask = imbinarize(kidneyROI, stoneThreshold / 255);
    potentialStonesMask = bwareaopen(potentialStonesMask, minStoneCandidateArea);
    
    stoneCandidates = regionprops(potentialStonesMask, 'Area', 'Centroid', 'Circularity');
    
    if ~isempty(stoneCandidates)
        numCandidates = length(stoneCandidates);
        scores = zeros(numCandidates, 1);
        kidneyProps = regionprops(kidneyMask, 'Centroid');
        kidneyCentroid = kidneyProps.Centroid;
        
        all_distances = zeros(numCandidates, 1);
        all_areas = [stoneCandidates.Area];
        for i = 1:numCandidates
            all_distances(i) = pdist([kidneyCentroid; stoneCandidates(i).Centroid]);
        end
        
        for i = 1:numCandidates
            score_shape = stoneCandidates(i).Circularity;
            score_location = 1 - (all_distances(i) / max(all_distances));
            if isnan(score_location), score_location = 1; end
            score_size = all_areas(i) / max(all_areas);
            if isnan(score_size), score_size = 1; end
            scores(i) = (weight_shape * score_shape) + (weight_location * score_location) + (weight_size * score_size);
        end
        
        [~, best_index] = max(scores);
        [stoneLabels, ~] = bwlabel(potentialStonesMask);
        finalStoneMask = (stoneLabels == best_index);
        
        % Calculate Diameter
        stone_props = regionprops(finalStoneMask, 'Area');
        if ~isempty(stone_props)
            area_px = stone_props.Area;
            diameter_px = sqrt(4 * area_px / pi);
            diameter_mm = diameter_px * pixel_to_mm_ratio;
            diameter_text = sprintf('%.1f mm', diameter_mm);
        end
    end
end

% --- Create Final Image Display ---
finalImage = imoverlay(grayImage, finalStoneMask, 'red');
if any(finalStoneMask(:))
    finalImage = insertText(finalImage, [10 10], ['Diameter: ', diameter_text], ...
        'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.6);
end

end