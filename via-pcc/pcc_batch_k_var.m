rng(1980);
[~,Xpca] = pca(X,'NumComponents',10);
pccres = zeros(100);
for i=1:100
    slabel = slabelgen(label,0.1);
    for k=1:100        
        owner = pccmex(Xpca, slabel, k, 'euclidean');
        pccres(i,k) = stmwevalk(label,slabel,owner);
        fprintf('Rodada: %3.0f K: %3.0f  Erro: %0.4f  Média: %0.4f\n',i,k,pccres(i,k),mean(pccres(1:i,k)));
    end
end
save pccres pccres