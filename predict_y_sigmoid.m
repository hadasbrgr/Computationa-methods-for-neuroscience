function predicted = predict_y_sigmoid(X, theta)
    predicted = X*theta';
    for i=1:length(X)
         if predicted(i)>0.5
              predicted(i)=1;
         else
              predicted(i)=0;
         end
    end
end