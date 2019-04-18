function model = train_model(data, params)
%% Initialize
rng(params.seed)

[num_samples, num_features] = size(data.X);
% initialize W to random values

theta =  rand(1,num_features);

%% SGD with hinge-loss
figure(1)
clf; hold on
xlabel('learning epoch'); ylabel('train error');
title(['Train model with epoch=' num2str(params.max_epoch)]);
error = [];
for epoch = 1:params.max_epoch
    fprintf('\nEpoch #%i: ', epoch)
    
    % Arrange samples in random order for each learning epoch
    epoch_order = randperm(num_samples);
    % Initialize the error
    error(epoch) = 0;
    
    % we will use stochastic gradient descent to train our model
    for iter = 1:num_samples
        % Get current sample
        sample_index = epoch_order(iter);
        current_sample = data.X(sample_index,:);   % COMPLETE CODE HERE
        current_label = data.Y(sample_index);    % COMPLETE CODE HERE
        % Update weights
        condition=0;
        for i=1:length(current_sample)
            condition= condition + (current_sample(i)*theta(i)*current_label);
        end
        if   condition<0 
            change_vec= -current_label.*current_sample; % COMPLETE CODE HERE
            theta = theta - (params.alpha.*change_vec);
            error(epoch) =  error(epoch)-condition;% COMPLETE CODE HERE
        end
    end
    error(epoch)=(1/num_samples)*error(epoch);
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