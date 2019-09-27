function [theta1, theta2] = nueronNetwork(hiddenLayerSize, x, y)
	#x = [1 3 2 4 3; 2 2 2 3 3; 3 2 4 1 3; 1 2 1 4 3]'
	#y = [1 0 1 0 1; 0 1 0 1 1; 1 0 1 0 1]'
	x = double(x)' / 255;
	y = y';
	xSize = size(x,1); ### input layer length;
	ySize = size(y,1); ### output layer length;
	theta1 = rand(xSize + 1, hiddenLayerSize);
	theta2 = rand(hiddenLayerSize + 1, ySize);


	theta = [theta1(:); theta2(:)];
	lam = 0.0002; 
	
	warning('off', 'Octave:possible-matlab-short-circuit-operator');
	#options = optimset('MaxIter', '1000');
	theta = fmincg(@(theta) costFunction(theta, x, y, xSize, hiddenLayerSize, lam), theta);
	
	theta1 = reshape(theta(1:hiddenLayerSize * (xSize + 1)), (xSize + 1), hiddenLayerSize);
	theta2 = reshape(theta((1 + (hiddenLayerSize * (xSize + 1))):end), (hiddenLayerSize + 1), size(y,1));
endfunction


function [jVal, gradient] = costFunction(theta, x, y, xSize, hiddenLayerSize, lam)
	theta1 = reshape(theta(1:hiddenLayerSize * (xSize + 1)), (xSize + 1), hiddenLayerSize);
	theta2 = reshape(theta((1 + (hiddenLayerSize * (xSize + 1))):end), (hiddenLayerSize + 1), size(y,1));
	
	m = size(x,2);
	z2 = calcZ(x, theta1);
	a2 = sigmoid(z2);
	z3 = calcZ(a2, theta2);
	a3 = sigmoid(z3);
	
	jVal = sum(-(1/m) * sum(y .* log(a3) + (1 - y) .* log(1 .- a3)) + (lam/2/m) * (sum(sum(theta1(2:end,:) .^ 2)) + sum(sum(theta2(2:end,:) .^ 2))));
	
	delta3 = a3 - y;
	delta2 = theta2 * delta3 .* addBias(sigmoidGrad(z2));
	delta2 = delta2(2:end,:);
	
	theta2Grad = (1/m) * addBias(a2) * delta3' + lam * [zeros(1, size(theta2, 2)); theta2(2:end,:)];	
	theta1Grad = (1/m) * addBias(x) * delta2' + lam * [zeros(1, size(theta1, 2)); theta1(2:end,:)];
	gradient = [theta1Grad(:); theta2Grad(:)];
	
	####checkDerivative(theta1, theta2, theta1Grad, theta2Grad, x, y, lam);
endfunction


###Checks to see if the derivative term is correct
function [dtheta1, dtheta2] = checkDerivative(theta1, theta2, theta1Grad, theta2Grad, x, y, lam)
	
	test = theta1;	
	[m1, n1] = size(theta1);
	[m2, n2] = size(theta2);
	dtheta1 = zeros(m1, n1);
	dtheta2 = zeros(m2, n2);
	
	for i=1:m1
		for j=1:n1
			
			theta1(i, j) += 0.0000001;
			m = size(x,2);
			z2 = calcZ(x, theta1);
			a2 = sigmoid(z2);
			z3 = calcZ(a2, theta2);
			a3 = sigmoid(z3);

			jValplus = sum(-(1/m) * sum(y .* log(a3) + (1 - y) .* log(1 .- a3)) + (lam/2/m) * (sum(sum(theta1(2:end,:) .^ 2)) + sum(sum(theta2(2:end,:) .^ 2))));
				
			theta1(i, j) -= 0.0000002;
			m = size(x,2);
			z2 = calcZ(x, theta1);
			a2 = sigmoid(z2);
			z3 = calcZ(a2, theta2);
			a3 = sigmoid(z3);
			
			jValminus = sum(-(1/m) * sum(y .* log(a3) + (1 - y) .* log(1 .- a3)) + (lam/2/m) * (sum(sum(theta1(2:end,:) .^ 2)) + sum(sum(theta2(2:end,:) .^ 2))));
			dtheta1(i, j) = (jValplus - jValminus) / 0.0000002 - theta1Grad(i,j)
		end
	end
	dtheta1;

	for i=1:m2
		for j=1:n2
			
			theta2(i, j) += 0.0000001;
			m = size(x,2);
			z2 = calcZ(x, theta1);
			a2 = sigmoid(z2);
			z3 = calcZ(a2, theta2);
			a3 = sigmoid(z3);

			jValplus = sum(-(1/m) * sum(y .* log(a3) + (1 - y) .* log(1 .- a3)) + (lam/2/m) * (sum(sum(theta1(2:end,:) .^ 2)) + sum(sum(theta2(2:end,:) .^ 2))));
				
			theta2(i, j) -= 0.0000002;
			m = size(x,2);
			z2 = calcZ(x, theta1);
			a2 = sigmoid(z2);
			z3 = calcZ(a2, theta2);
			a3 = sigmoid(z3);
			
			jValminus = sum(-(1/m) * sum(y .* log(a3) + (1 - y) .* log(1 .- a3)) + (lam/2/m) * (sum(sum(theta1(2:end,:) .^ 2)) + sum(sum(theta2(2:end,:) .^ 2))));
			dtheta2(i, j) = (jValplus - jValminus) / 0.0000002 - theta2Grad(i,j);
		end
	end
	dtheta2
endfunction

function z = calcZ(a, theta)
	z = theta' * addBias(a);
endfunction

function a = sigmoid(z)
	a = 1 ./ (1 .+ e .^ (-1 * z));
endfunction

function x = sigmoidGrad(z)
	x = sigmoid(z) .* (1 - sigmoid(z));
endfunction

function a = addBias(a)
	a = [ones(1, size(a, 2)); a];
endfunction

function x = removeTopLayer(x)
	x = x(2:size(x, 1), :);
endfunction