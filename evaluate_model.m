function precision = evaluate_model(predicted_Y, true_Y)
	precision = 0;
    for i=1:length(predicted_Y)
        if(predicted_Y(i)==true_Y(i))
            precision=precision+1;
        end
    end
    precision=100*(precision/length(predicted_Y));
end