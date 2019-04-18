function predicted = predict_y(X, theta)
  for i=1:size(X,1)
    predicted(i,:) = sign(sum((theta.*X(i,:))));
  end
end