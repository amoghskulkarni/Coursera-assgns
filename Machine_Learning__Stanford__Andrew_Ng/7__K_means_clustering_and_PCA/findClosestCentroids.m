function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i = 1:size(X,1)
  x_i = X(i,:);
  
  diff_v = x_i - centroids(1,:);
  closest_c_dist = sum(sum(diff_v .* diff_v));
  closest_c = 1;
  
  for j = 2:K
    c_j = centroids(j,:);
    
    diff_v = x_i - c_j;
    c_dist = sum(sum(diff_v .* diff_v));
    
    if c_dist < closest_c_dist
      closest_c_dist = c_dist;
      closest_c = j;
    endif
  
  endfor

  idx(i) = closest_c;


% =============================================================

end

