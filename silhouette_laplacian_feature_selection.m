%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%              OPTIMAL CLUSTER NUMBER AND FEATURE SELECTION               %
%                   USING SILHOUETTE ANALYSIS                             %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Description:
%   This script implements an unsupervised feature selection algorithm 
%   that combines silhouette analysis with Laplacian score to identify 
%   the optimal features for clustering analysis. The algorithm is 
%   particularly useful for identifying the predominant site-of-collapse 
%   in obstructive sleep apnoea patients using acoustic features from 
%   snore signals.
%
% Based on Publication:
%   Sebastian, A., Cistulli, P. A., Cohen, G., & de Chazal, P. (2021). 
%   "Unsupervised Approach for the Identification of the Predominant 
%   Site of Upper Airway Collapse in Obstructive Sleep Apnoea Patients 
%   Using Snore Signals." 
%   43rd Annual International Conference of the IEEE Engineering in 
%   Medicine & Biology Society (EMBC), 2021, pp. 961-964.
%   DOI: 10.1109/EMBC46164.2021.9630150
%
% Algorithm Overview:
%   1. Find optimal cluster number for each feature (k=1 to 6)
%   2. Filter features based on cluster uniformity (40%-60% threshold)
%   3. Rank selected features using Laplacian Score
%   4. Sequential feature addition until first maximum silhouette score
%   5. Evaluate clustering performance with optimal features
%
% Key Features:
%   - Unsupervised approach (no ground truth labels required)
%   - Automatic feature selection combining multiple criteria
%   - Built-in cluster validation using silhouette analysis
%   - Locality preservation through Laplacian Score
%
% Input Data Requirements:
%   - SampleData can be genereted using SampleDataGenerator.m
%   - Feature matrix: n_samples × n_features
%   - Optional: Labels in last column for validation
%   - Data should be numerical and properly normalized
%
% Output:
%   - Selected feature indices
%   - Optimal number of clusters
%   - Silhouette scores throughout selection process
%   - Cluster assignments for samples
%   - Comprehensive visualization plots
%   - Results saved to .mat file
%
% Performance Metrics (from paper):
%   - Optimal cluster number: 2 (tongue/non-tongue collapse)
%   - Features selected: 17 out of 50
%   - Mean silhouette coefficient: 0.79
%   - Classification accuracy: 68% for predominant site-of-collapse
%
% Dependencies:
%   - Statistics and Machine Learning Toolbox
%   - MATLAB R2019b or later
%   - Function: fsulaplacian (for Laplacian Score calculation)
%
% Author:
%   Original Algorithm: Arun Sebastian
% 


clear all
close all
rng(11)

%% Step 1: Data Loading and Preprocessing
load('generatedData.mat') % This data is generated using Matlab code
NF = length(data_len);
num_feat = size(sampleData, 2) - 1; % Last column is labels

fprintf('Loaded %d patients with %d features\n', NF, num_feat);
fprintf('Total samples: %d\n\n', size(sampleData, 1));

%% Step 2: Find Optimal Cluster Number for Each Feature
fprintf('=== Step 2: Finding Optimal Cluster Number ===\n');

opt1 = zeros(2, num_feat);

for i = 1:num_feat
    X = sampleData(:, i);
    % Evaluate silhouette scores for k=1 to 6 clusters
    E = evalclusters(X, 'kmeans', 'Silhouette', 'klist', [1:6]);
    
    opt1(1, i) = E.OptimalK;  % Optimal cluster number for this feature
    opt1(2, i) = E.CriterionValues(E.OptimalK);  % Silhouette score
    
    % Optional: Plot silhouette curves
    % sh = E.CriterionValues;
    % plot(1:6, sh, '-o')
    % hold on
end

% Determine overall optimal k using majority voting
optk = mode(opt1(1, :));
optk_rep = length(find(opt1(1, :) == optk));

fprintf('Optimal cluster number (majority vote): %d\n', optk);
fprintf('Number of features agreeing: %d/%d (%.1f%%)\n\n', ...
        optk_rep, num_feat, 100*optk_rep/num_feat);

%% Step 3: Feature Selection Based on Cluster Uniformity
fprintf('=== Step 3: Feature Selection (Uniformity Test) ===\n');

opt = zeros(4, num_feat);
opt(1, :) = opt1(1, :);

for i = 1:num_feat
    X = sampleData(:, i);
    
    % Perform k-means clustering with k=2
    idx = kmeans(X, 2, 'Distance', 'sqeuclidean', 'Display', 'off');
    
    % Calculate silhouette values
    silh5 = silhouette(X, idx, 'sqeuclidean');
    
    opt(2, i) = mean(silh5);  % Mean silhouette score
    opt(3, i) = sum(silh5 > mean(silh5));  % Points above mean
    opt(4, i) = sum(idx == 1);  % Size of cluster 1
end

% Select features with uniform cluster thickness
% Discard if any cluster has >60% or <40% of data
feat_sel = find(opt(4, :) < 0.6*size(sampleData, 1) & opt(4, :) > 0.4*size(sampleData, 1));

fprintf('Features passing uniformity test: %d/%d\n', length(feat_sel), num_feat);
fprintf('Selected feature indices: ');
fprintf('%d ', feat_sel);
fprintf('\n\n');

close all
clearvars -except sampleData feat_sel NF opt PSOC_M data_len

%% Step 4: Ranking Selected Features Using Laplacian Score
fprintf('=== Step 4: Feature Ranking (Laplacian Score) ===\n');

X = sampleData(:, feat_sel);

% Calculate Laplacian scores for feature ranking
[rnk_feat, scores] = fsulaplacian(X, 'Distance', 'seuclidean');

% Get features in ranked order (best to worst)
feat_relev = feat_sel(rnk_feat);

fprintf('Features ranked by Laplacian score (best first):\n');
for i = 1:min(10, length(feat_relev))
    fprintf('  %2d. Feature %2d (score: %.4f)\n', i, feat_relev(i), scores(rnk_feat(i)));
end
fprintf('\n');

%% Step 5: Sequential Feature Addition Until First Maximum
fprintf('=== Step 5: Sequential Feature Addition ===\n');

opt = zeros(6, length(feat_relev));

% Load ground truth labels if available
data_len = data_len(1:NF);
PSOC_M = PSOC_M(1:NF);

for i = 1:length(feat_relev)
    % Select top i features
    data = sampleData(:, feat_relev(1:i));
    
    % Evaluate optimal cluster number
    E = evalclusters(data, 'kmeans', 'silhouette', 'klist', [1:6]);
    opt(1, i) = E.OptimalK;
    
    % Perform k-means with k=2
    idx = kmeans(data, 2, 'Distance', 'sqeuclidean', 'Display', 'off');
    
    % Calculate silhouette metrics
    silh5 = silhouette(data, idx, 'sqeuclidean');
    opt(2, i) = mean(silh5);  % Mean silhouette score
    opt(3, i) = sum(silh5 > 0.7);  % Points with high silhouette
    
    % Evaluate against ground truth (predominant site-of-collapse)
    Y = mat2cell(idx(:, :).', 1, data_len);
    predom_val = 0.6;
    
    for j = 1:NF
        k = Y{j};
        rep = mode(k);
        num = sum(k(:) == rep);
        
        if (mode(k) == 1 && num > predom_val * length(k))
            PSOC_A{j} = 'N';  % Non-tongue
        else
            PSOC_A{j} = 'T';  % Tongue
        end
    end
    
    % Calculate accuracy
    opt(4, i) = sum(strcmp(PSOC_M, PSOC_A));  % Patient-level accuracy
    
    % Event-level accuracy
    target = sampleData(:, end);
    target = target + 1;
    target(target == 2) = 1;
    target(target == 3) = 2;
    opt(5, i) = sum(target == idx) / size(target, 1);
    opt(6, i) = sum(idx == 1);  % Cluster 1 size
    
    fprintf('Features: %2d | Silhouette: %.3f | Patient Acc: %d/%d (%.1f%%) | Event Acc: %.1f%%\n', ...
            i, opt(2, i), opt(4, i), NF, 100*opt(4, i)/NF, 100*opt(5, i));
end

close all

% Find the first maximum in silhouette scores
[max_silh, best_n_feat] = max(opt(2, :));
fprintf('\n=== Results ===\n');
fprintf('Best number of features: %d\n', best_n_feat);
fprintf('Maximum silhouette score: %.3f\n', max_silh);
fprintf('Patient-level accuracy: %d/%d (%.1f%%)\n', ...
        opt(4, best_n_feat), NF, 100*opt(4, best_n_feat)/NF);
fprintf('Event-level accuracy: %.1f%%\n\n', 100*opt(5, best_n_feat));

% Display selected features
fprintf('Final selected features:\n');
for i = 1:best_n_feat
    fprintf('  Feature %d\n', feat_relev(i));
end

%% Visualization: Silhouette Score vs Number of Features
figure('Position', [100, 100, 900, 600]);

subplot(2, 2, 1)
plot(1:length(feat_relev), opt(2, :), 'ro-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(best_n_feat, 'g--', 'LineWidth', 2);
xlabel('Number of Features', 'FontSize', 12);
ylabel('Mean Silhouette Score', 'FontSize', 12);
title('Feature Selection Process', 'FontSize', 14);
grid on;
legend('Silhouette Score', 'Optimal Features', 'Location', 'best');

subplot(2, 2, 2)
plot(1:length(feat_relev), 100*opt(4, :)/NF, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(best_n_feat, 'g--', 'LineWidth', 2);
xlabel('Number of Features', 'FontSize', 12);
ylabel('Patient Accuracy (%)', 'FontSize', 12);
title('Patient-Level Classification', 'FontSize', 14);
grid on;

subplot(2, 2, 3)
plot(1:length(feat_relev), 100*opt(5, :), 'mo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(best_n_feat, 'g--', 'LineWidth', 2);
xlabel('Number of Features', 'FontSize', 12);
ylabel('Event Accuracy (%)', 'FontSize', 12);
title('Event-Level Classification', 'FontSize', 14);
grid on;

subplot(2, 2, 4)
plot(1:length(feat_relev), opt(1, :), 'ko-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(best_n_feat, 'g--', 'LineWidth', 2);
yline(2, 'r--', 'LineWidth', 1.5);
xlabel('Number of Features', 'FontSize', 12);
ylabel('Optimal K', 'FontSize', 12);
title('Optimal Cluster Number', 'FontSize', 14);
grid on;
ylim([1, 6]);

%% Final Clustering Visualization (2D projection)
data = sampleData(:, feat_relev(1:best_n_feat));
idx = kmeans(data, 2, 'Distance', 'sqeuclidean', 'Display', 'off');

% Use first two features for visualization
figure('Position', [100, 100, 800, 600]);
gscatter(data(:, 1), data(:, 2), idx, 'br', '*+', 8);
xlabel(sprintf('Feature %d', feat_relev(1)), 'FontSize', 12);
ylabel(sprintf('Feature %d', feat_relev(2)), 'FontSize', 12);
title('2D Clustering Visualization (First Two Features)', 'FontSize', 14);
legend('Cluster 1', 'Cluster 2', 'Location', 'best');
grid on;

% Highlight correctly classified points
target = sampleData(:, end);
target = target + 1;
target(target == 2) = 1;
target(target == 3) = 2;
idxBoth = find(target == idx);
hold on;
plot(data(idxBoth, 1), data(idxBoth, 2), 'ok', 'MarkerSize', 10, ...
     'LineWidth', 1.5);
legend('Cluster 1', 'Cluster 2', 'Correct Classification', 'Location', 'best');

%% Silhouette Diagram for Final Model
figure('Position', [100, 100, 900, 700]);
[silh_vals, h] = silhouette(data, idx, 'sqeuclidean');
title(sprintf('Silhouette Plot (Mean: %.3f, %d Features)', mean(silh_vals), best_n_feat), ...
      'FontSize', 14);
xlabel('Silhouette Coefficient', 'FontSize', 12);
ylabel('Cluster', 'FontSize', 12);

%% Save Results
save('optimal_feature_selection_results.mat', 'feat_relev', 'best_n_feat', ...
     'opt', 'max_silh', 'data', 'idx', 'PSOC_A', 'PSOC_M');

fprintf('\n=== Analysis Complete ===\n');
fprintf('Results saved to: optimal_feature_selection_results.mat\n');