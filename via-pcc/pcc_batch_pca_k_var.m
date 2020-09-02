rng(1980);
[~,Xpca] = pca(X,'NumComponents',20);
pccres = zeros(20,20,100);
for i=1:100
    slabel = slabelgen(label,0.2);
    for p=1:20
        parfor k=1:20
            owner = pccmex(Xpca(:,1:p), slabel, k, 'euclidean');
            pccres(p,k,i) = stmwevalk(label,slabel,owner);
            %fprintf('Round: %3.0f  PCA components: %3.0f  k: %3.0f  Acc: %0.4f\n',i,p,k,pccres(p,k,i));
        end        
        [val, ind] = max(pccres(p,:,i));
        fprintf('Round %3.0f  PCA components: %3.0f  Best k: %3.0f  Acc: %0.4f\n',i,p,ind,val)
    end
    [vallist, indlist] = max(mean(pccres(:,:,1:i),3));
    [val, ind] = max(vallist);
    indp = indlist(ind);
    fprintf('PARTIAL %i/100: Best # PCA comp.: %3.0f  Best k: %3.0f  Mean Acc: %0.4f\n',i,indp,ind,val)
    save(sprintf('pccres-%s',getenv('computername')),'pccres');
end
[vallist, indlist] = max(mean(pccres,3));
[val, indk] = max(vallist);
indp = indlist(indk);
stddev = std(pccres(indp,indk,:));
fprintf('FINAL: Best #PCA Comp. %3.0f  Best k: %3.0f  Mean Acc: %0.4f  Std. Dev.: %0.4f\n',indp,indk,val,stddev)
save(sprintf('pccres-%s',getenv('computername')),'pccres','indk','indp','val','stddev');