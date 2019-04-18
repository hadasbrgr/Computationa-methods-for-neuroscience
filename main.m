% INIT
clear; close all; clc

%calculate parameters with HINGE
%Settings & Parameters
[settings, params] = load_settings_params();
%Data
[data_train, data_test] = load_data(settings);

%Train model
model  = train_model(data_train, params);
model_train_error=mean(model.training_error);
%Predict
prediction = predict_y(data_train.X, model.theta);
%evaluate
precision = evaluate_model(prediction, data_train.Y);

% Test set
%Predict
prediction_test = predict_y(data_test.X, model.theta);
%evaluate
precision_test = evaluate_model(prediction_test, data_test.Y);

%%
% Calculate parameters with SIGMOID function
% Settings & Parameters
[settings, params] = load_settings_params();
% Data
[data_train, data_test] = load_data_sigmoid(settings);

% Train model
model  = train_model_sigmoid(data_train, params);
% Predict
prediction = predict_y_sigmoid(data_train.X, model.theta);
% evaluate
precision_sigmoid = evaluate_model(prediction, data_train.Y);

%Test set
% Predict
prediction_test_for_sigmoid = predict_y_sigmoid(data_test.X, model.theta);
% evaluate
precision_test_sigmoid = evaluate_model(prediction_test_for_sigmoid , data_test.Y);
