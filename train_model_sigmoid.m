function model = train_model_sigmoid(data, params)
%% Initialize
rng(params.seed)

[num_samples, num_features] = size(data.X);
% initialize W to random values

theta =  rand(1,num_features);

%% SGD with sigmoid
figure(1)
clf; hold on
xlabel('learning epoch'); ylabel('train error');
title(['Train model with epoch=' num2str(params.max_epoch) ' and alpha=' num2str(params.alpha)]);
error = [];
for epoch = 1:params.max_epoch
    fprintf('\nEpoch #%i: ', epoch)
    
    % Arrange samples in random order for each learning epoch
    epoch_order = randperm(num_samples);
    % Initialize the error
    error =zeros(1, params.max_epoch);
    Hx = 0;
    loss = 0;
    % we will use stochastic gradient descent to train our model
    for iter = 1:num_samples
        % Get current sample
        sample_index = epoch_order(iter);
        current_sample = data.X(sample_index,:);  % COMPLETE CODE HERE
        current_label = data.Y(sample_index);    % COMPLETE CODE HERE
        
        % Update weights
        Hx = dot(current_sample, theta);
        Gx = sigmf(Hx,[1,0]); 
        if current_label == 0
            loss = -log(1-Gx);
        else
            loss = -log(Gx);
        end
        lambda_change = 0;
        if params.regulation~=0
            regulation = 0.5*params.lambda*sum(theta.*theta);
            loss = loss + regulation;
            lambda_change = -params.alpha*params.lambda*theta;
        end
        error(epoch) = error(epoch) + (loss/num_samples);
        theta = theta-params.alpha*(Gx-current_label).*current_sample + lambda_change;
    end
    
    % Plot average error
    if epoch>params.convergence_window
        plot(epoch, mean(error(epoch-params.convergence_window:epoch)), '.', 'MarkerSize', 10)
        title(['Train model with epoch=' num2str(params.max_epoch) ' and alpha=' num2str(params.alpha)]);
        drawnow
    end
    
    % Stopping criteria
    if epoch>params.max_epoch
        prevStart=epoch-params.convergence_window*2; prevEnd=epoch-params.convergence_window;
        currStart=epoch-params.convergence_window+1;
        if (mean(error(prevStart:prevEnd))<mean(error(currStart:epoch)))
            break
        end
    end
end
if epoch < params.max_epoch
    fprintf('\nStopped after %i epochs\n', epoch)
else
    fprintf('\nNo convergence - epoch number exceeded maximal number of epochs\n/')
end
%% Output model
model.theta = theta;
model.training_error = error;
model.num_of_epochs = epoch;
end