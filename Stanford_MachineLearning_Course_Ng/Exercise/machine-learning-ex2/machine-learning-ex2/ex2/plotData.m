function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure;
hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
accept_arg=find(y==1);
reject_arg=find(y==0);
plot(X(accept_arg,1),X(accept_arg,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(reject_arg,1),X(reject_arg,2),'ko','MarkerFaceColor','y','MarkerSize',7);
xlabel('Exam1','FontSize',10);
ylabel('Exam2','FontSize',10);
title('Data graph','FontSize',18);
legend('Admitted','Not admitted');
axis tight;
axis equal;
% =========================================================================



hold off;

end
