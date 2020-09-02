k = 7;
rng(1980);
[~,Xpca] = pca(X,'NumComponents',100);
pccres = zeros(100);
for i=1:100
    slabel = slabelgen(label,0.2);
    for p=1:100        
        owner = pccmex(Xpca(:,1:p), slabel, k, 'euclidean');
        pccres(i,p) = stmwevalk(label,slabel,owner);
        fprintf('Round: %3.0f  PCA components: %3.0f  Error: %0.4f  Mean: %0.4f\n',i,p,pccres(i,p),mean(pccres(1:i,p)));
    end
end
save pccres pccres