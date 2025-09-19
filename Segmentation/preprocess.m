% Preprocessing of Medical Images
% Noise Filtering + Contrast Adjustment + Optional Sharpening
% ============================

% Input dataset paths
normalFolder = 'F:\madical Imaging LAB\project\my dataset final 512x512(implemented)\Normal';
stoneFolder  = 'F:\madical Imaging LAB\project\my dataset final 512x512(implemented)\stone';

% Output dataset paths
outNormalFolder = 'F:\madical Imaging LAB\project\my dataset final 512x512(implemented)\processed_Normal';
outStoneFolder  = 'F:\madical Imaging LAB\project\my dataset final 512x512(implemented)\processed_stone';

% Create output folders if they don’t exist
if ~exist(outNormalFolder, 'dir')
    mkdir(outNormalFolder);
end
if ~exist(outStoneFolder, 'dir')
    mkdir(outStoneFolder);
end

% -------- Run Processing --------
processImages(normalFolder, outNormalFolder);
processImages(stoneFolder, outStoneFolder);

disp('✅ Preprocessing complete! Check processed_Normal and processed_stone folders.');
% ============================

% -------- Processing Function --------
function processImages(inputFolder, outputFolder)
    % Accept multiple image formats
    exts = {'*.JPG','*.jpeg','*.png','*.bmp','*.tif','*.tiff','*.dcm'};
    imgFiles = [];
    for i = 1:numel(exts)
        imgFiles = [imgFiles; dir(fullfile(inputFolder, exts{i}))]; %#ok<AGROW>
    end
    
    for k = 1:length(imgFiles)
        % Read image
        imgPath = fullfile(inputFolder, imgFiles(k).name);
        
        % Handle DICOM separately
        [~,~,ext] = fileparts(imgPath);
        if strcmpi(ext, '.dcm')
            img = dicomread(imgPath);
            img = mat2gray(img); % scale to 0-1
        else
            img = imread(imgPath);
        end

        % Convert grayscale to RGB if needed
        if size(img,3) == 1
            imgRGB = cat(3, img, img, img);
        else
            imgRGB = img;
        end

        % Step 1: Noise removal using edge-preserving Wiener filter
        imgFiltered = zeros(size(imgRGB), 'like', imgRGB);
        for c = 1:3
            imgFiltered(:,:,c) = wiener2(imgRGB(:,:,c), [2 2]);
        end

        % Step 2: Contrast adjustment
        imgContrast = zeros(size(imgFiltered), 'like', imgFiltered);
        for c = 1:3
            imgContrast(:,:,c) = imadjust(imgFiltered(:,:,c));
        end

        % Step 3 (Optional): Light sharpening to restore edges
        imgFinal = zeros(size(imgContrast), 'like', imgContrast);
        for c = 1:3
           imgFinal(:,:,c) = imsharpen(imgContrast(:,:,c), 'Radius',1,'Amount',0.5);
        end

        % Save processed image (keep same filename)
        outPath = fullfile(outputFolder, imgFiles(k).name);
        imwrite(imgFinal, outPath);
    end
end

