function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hypothesis = sigmoid(X * theta);
log_hypo = log(hypothesis);
log_one_minus_hypo = log(1 - hypothesis);

for idx = 1:m
  J = J + (-1 * y(idx))*log_hypo(idx) - (1 - y(idx))*log_one_minus_hypo(idx);
endfor

J = J / m;

errors =  hypothesis - y;

for feature_id = 1:size(theta)(1)
  grad(feature_id) = (1 / m) * (errors' * X(:, feature_id));
endfor

% =============================================================

end
