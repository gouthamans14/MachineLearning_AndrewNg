function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h_tet = sigmoid(X*theta);
reg=((lambda/(2*m))*(sum(theta(2:end).^2)));
J = ((1/m)*((-y'*log(h_tet))-((1-y)'*log(1-h_tet))))+reg;
err = h_tet -y;
l= ((1/m)*(err'*X));


l1=((1/m)*(err'*X))'+((lambda/m)*theta);
grad=[l(1);l1(2:length(l1))];


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
