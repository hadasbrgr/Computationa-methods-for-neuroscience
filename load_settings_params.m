function [settings, params] = load_settings_params()

    %%  SETTINGS
    settings.dataset_file = 'Neurons.txt';
    settings.path2data = fullfile('..', 'data',settings.dataset_file);

    %% MODEL PARAMETERS
    params.seed = 1;
    params.alpha = 0.1;                
    params.max_epoch = 1000;
    params.convergence_window = 50;
    params.CV_k = 5;
    params.regulation = 1;
    params.lambda = 0.2;
end