
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    SYNTHETIC DATA GENERATOR                             %
%          FOR FEATURE SELECTION ANALYSIS IN OSA RESEARCH                 %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Purpose:
%   Generate synthetic sample data that mimics the structure and 
%   characteristics of real obstructive sleep apnoea (OSA) acoustic 
%   feature data for testing and validation of the feature selection 
%   algorithm.
%
% Based on Publication:
%   Sebastian, A., Cistulli, P. A., Cohen, G., & de Chazal, P. (2021). 
%   "Unsupervised Approach for the Identification of the Predominant 
%   Site of Upper Airway Collapse in Obstructive Sleep Apnoea Patients 
%   Using Snore Signals." 
%   43rd Annual International Conference of the IEEE Engineering in 
%   Medicine & Biology Society (EMBC), 2021, pp. 961-964.
%   DOI: 10.1109/EMBC46164.2021.9630150
% =========================================================================
% SITE-OF-COLLAPSE LABELS
% =========================================================================
%
% Event-Level Labels (in sampleData column 57):
%   0 = Lateral wall collapse
%   1 = Palate collapse
%   2 = Tongue-base collapse
%
% Patient-Level Labels (in PSOC_M):
%   'N' = Non-tongue predominant (Lateral wall or Palate)
%   'T' = Tongue-base predominant
%
% Predominance Rule:
%   If ≥60% of patient's events are of one type, that type is 
%   considered the predominant site-of-collapse for that patient
%
% =========================================================================
% DATA GENERATION METHODOLOGY
% =========================================================================
%
% Feature Generation:
%   - Base features: Random normal distribution N(0,1)
%   - Cluster separation: Added mean offsets based on labels
%   - Noise: Realistic variability to mimic real data
%   - Correlations: Some features correlated to simulate real acoustics
%
% Label Assignment:
%   - Patient-level: Random assignment maintaining ~55% tongue prevalence
%   - Event-level: Generated with some variability around patient type
%   - Predominance: Enforced ≥60% rule for patient classification
%
% =========================================================================
% OUTPUT FILES
% =========================================================================
%
% Saved File: 'generatedData.mat'
% Contains:
%   - sampleData  : Feature matrix with labels
%   - data_len    : Events per patient
%   - PSOC_M      : Predominant site-of-collapse per patient
%
% File Size: ~2-5 MB (depending on total events)
%
% =========================================================================
clear all
close all
rng(42)  % For reproducibility

fprintf('=== Generating Sample Data ===\n\n');

%% Configuration
num_patients = 49;
excluded_patients = [25, 34, 37, 39];  % Problematic files to exclude
min_events = 50;
max_events = 200;

%% Generate data_len (number of events per patient)
data_len = randi([min_events, max_events], 1, num_patients);

fprintf('Generated data_len:\n');
fprintf('  Min events: %d\n', min(data_len));
fprintf('  Max events: %d\n', max(data_len));
fprintf('  Mean events: %.1f\n', mean(data_len));
fprintf('  Total events: %d\n\n', sum(data_len));

% Save data_len
save('data_len.mat', 'data_len');
fprintf('Saved: data_len.mat\n');

%% Generate PSOC_M (predominant site-of-collapse labels)
% Create a balanced mix of 'T' (Tongue) and 'N' (Non-tongue)
num_tongue = round(num_patients * 0.5);
PSOC_M = cell(1, num_patients);

% Assign labels
tongue_patients = randperm(num_patients, num_tongue);
for i = 1:num_patients
    if ismember(i, tongue_patients)
        PSOC_M{i} = 'T';  % Tongue
    else
        PSOC_M{i} = 'N';  % Non-tongue
    end
end

fprintf('Generated PSOC_M:\n');
fprintf('  Tongue (T): %d patients\n', sum(strcmp(PSOC_M, 'T')));
fprintf('  Non-tongue (N): %d patients\n\n', sum(strcmp(PSOC_M, 'N')));

% Save PSOC_M
save('PSOC_M.mat', 'PSOC_M');
fprintf('Saved: PSOC_M.mat\n\n');

%% Generate individual patient files (001.mat to 049.mat)
fprintf('Generating patient data files...\n');

for patient_id = 1:num_patients
    % Skip excluded patients
    if ismember(patient_id, excluded_patients)
        fprintf('  Skipping patient %03d (excluded)\n', patient_id);
        continue;
    end
    
    num_events = data_len(patient_id);
    
    % Determine predominant label for this patient
    if strcmp(PSOC_M{patient_id}, 'T')
        predominant_label = 2;  % Tongue
        % 70-80% of events will be tongue
        tongue_ratio = 0.7 + 0.1 * rand();
    else
        predominant_label = randi([0, 1]);  % Lateral wall or Palate
        % 70-80% of events will be non-tongue
        tongue_ratio = 0.1 + 0.2 * rand();
    end
    
    % Generate labels for events
    labels = zeros(num_events, 1);
    num_tongue_events = round(num_events * tongue_ratio);
    tongue_indices = randperm(num_events, num_tongue_events);
    labels(tongue_indices) = 2;  % Tongue
    
    % Assign non-tongue labels
    non_tongue_indices = setdiff(1:num_events, tongue_indices);
    if predominant_label < 2
        labels(non_tongue_indices) = predominant_label;
    else
        % Mix of lateral wall and palate
        half = floor(length(non_tongue_indices) / 2);
        labels(non_tongue_indices(1:half)) = 0;  % Lateral wall
        labels(non_tongue_indices(half+1:end)) = 1;  % Palate
    end
    
    % Generate 46 features per event
    % Features 1-20: MFCC-like features (mean and variance patterns)
    % Features 21-45: Other acoustic features
    % Feature 46: Label
    
    FeatureData = zeros(num_events, 46);
    
    % Generate features with different characteristics for each label
    for event_idx = 1:num_events
        label = labels(event_idx);
        
        % MFCC coefficients (features 1-20)
        if label == 0  % Lateral wall
            mfcc_base = randn(1, 20) * 2 + [linspace(5, -2, 10), linspace(-1, 3, 10)];
        elseif label == 1  % Palate
            mfcc_base = randn(1, 20) * 2.5 + [linspace(3, 1, 10), linspace(2, -2, 10)];
        else  % Tongue (label == 2)
            mfcc_base = randn(1, 20) * 1.8 + [linspace(-1, 4, 10), linspace(3, 0, 10)];
        end
        
        FeatureData(event_idx, 1:20) = mfcc_base;
        
        % Additional acoustic features (21-45)
        % These represent spectral, energy, and temporal features
        for feat_idx = 21:45
            if label == 2  % Tongue - distinct pattern
                FeatureData(event_idx, feat_idx) = randn() * 1.5 + 5 * sin(feat_idx/5);
            else  % Non-tongue
                FeatureData(event_idx, feat_idx) = randn() * 2 + 3 * cos(feat_idx/4);
            end
        end
        
        % Add some cross-feature correlations for realism
        if label == 2
            FeatureData(event_idx, 25) = FeatureData(event_idx, 10) * 0.7 + randn() * 0.5;
            FeatureData(event_idx, 35) = FeatureData(event_idx, 15) * 0.6 + randn() * 0.6;
        end
        
        % Store label
        FeatureData(event_idx, 46) = label;
    end
    
    % Save patient file
    filename = sprintf('%03d.mat', patient_id);
    save(filename, 'FeatureData');
    
    if mod(patient_id, 10) == 0
        fprintf('  Generated %d/%d files...\n', patient_id, num_patients);
    end
end

fprintf('Completed! Generated %d patient files.\n\n', num_patients - length(excluded_patients));

%% Summary Statistics
fprintf('=== Data Generation Summary ===\n');
fprintf('Total patients: %d\n', num_patients);
fprintf('Excluded patients: %d\n', length(excluded_patients));
fprintf('Valid patient files: %d\n', num_patients - length(excluded_patients));
fprintf('Total events across all patients: %d\n', sum(data_len));
fprintf('Features per event: 46\n');
fprintf('Label distribution:\n');
fprintf('  - Tongue (T): %d patients (%.1f%%)\n', sum(strcmp(PSOC_M, 'T')), ...
        100*sum(strcmp(PSOC_M, 'T'))/num_patients);
fprintf('  - Non-tongue (N): %d patients (%.1f%%)\n\n', sum(strcmp(PSOC_M, 'N')), ...
        100*sum(strcmp(PSOC_M, 'N'))/num_patients);

fprintf('Files created:\n');
fprintf('  - data_len.mat\n');
fprintf('  - PSOC_M.mat\n');
fprintf('  - 001.mat to 049.mat (excluding %d files)\n', length(excluded_patients));
fprintf('\nYou can now run the main feature selection script!\n');

%% Verify one sample file
fprintf('\n=== Sample File Verification ===\n');
load('001.mat');
fprintf('File 001.mat:\n');
fprintf('  Events: %d\n', size(FeatureData, 1));
fprintf('  Features: %d\n', size(FeatureData, 2));
fprintf('  Label distribution:\n');
fprintf('    Lateral wall (0): %d\n', sum(FeatureData(:,46)==0));
fprintf('    Palate (1): %d\n', sum(FeatureData(:,46)==1));
fprintf('    Tongue (2): %d\n', sum(FeatureData(:,46)==2));
fprintf('  Feature value ranges:\n');
fprintf('    Min: %.2f\n', min(FeatureData(:)));
fprintf('    Max: %.2f\n', max(FeatureData(:)));
fprintf('    Mean: %.2f\n', mean(FeatureData(:)));

%% Combine all 49 patient files into one matrix
fprintf('\n=== Combining All Patient Data ===\n');

Files = setdiff(1:49, excluded_patients);
X1_combined = zeros(1, 46);  % Initialize with empty row

for i = 1:length(Files)
    load(sprintf('%03d.mat', Files(i)));      
    X1_combined = [X1_combined; FeatureData(:, 1:46)];
end

% Remove initial empty row
X1_combined = X1_combined(2:end, :);

fprintf('Combined dataset:\n');
fprintf('  Total events: %d\n', size(X1_combined, 1));
fprintf('  Total features + label: %d\n', size(X1_combined, 2));
fprintf('  Label distribution:\n');
fprintf('    Lateral wall (0): %d (%.1f%%)\n', sum(X1_combined(:,46)==0), ...
        100*sum(X1_combined(:,46)==0)/size(X1_combined,1));
fprintf('    Palate (1): %d (%.1f%%)\n', sum(X1_combined(:,46)==1), ...
        100*sum(X1_combined(:,46)==1)/size(X1_combined,1));
fprintf('    Tongue (2): %d (%.1f%%)\n', sum(X1_combined(:,46)==2), ...
        100*sum(X1_combined(:,46)==2)/size(X1_combined,1));

% Save combined data
save('combined_data.mat', 'X1_combined');
fprintf('\nSaved: combined_data.mat\n');
fprintf('  Variable name: X1_combined\n');
fprintf('  Size: %d x %d\n', size(X1_combined, 1), size(X1_combined, 2));

fprintf('\n=== All Data Generation Complete ===\n');