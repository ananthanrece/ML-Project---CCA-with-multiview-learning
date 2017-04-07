clc;
clear all;
close all;

dataset = 'VIPeR';
%dataset = 'PRID';

if strcmp(dataset,'VIPeR')
    data = load( 'data/VIPeR_split.mat');
    n_gallery = 316;
    n_train = 316;
    n_test = 316;
end

if strcmp(dataset,'PRID')
    data = load( 'data/PRID_split.mat');
    n_gallery = 649;
    n_train = 100;
    n_test = 100;
end


tic
for round = 1:1
    
    train_a = data.trials(round).featAtrain; %dxN
    train_b = data.trials(round).featBtrain; %dxN
    
%     idxTrain_a = data.trials(round).labelsAtrain; %1xN
%     idxTrain_b = data.trials(round).labelsBtrain; %1xN
    m=1;
    [w_a,w_b] = fn_cca(train_a,train_b,m);
    w_a_cca = real(w_a);
    w_b_cca = real(w_b);
    
    probe = data.trials(round).featAtest;
    gallery = data.trials(round).featBtest;
    idxProbe = randperm(n_test);
    idxGallery = randperm(n_gallery);
    test_a = probe(:,idxProbe);
    test_b = gallery(:,idxGallery);
    
    proj_test_probe = test_a'*w_a_cca;
    proj_test_gallery = test_b'*w_b_cca;
    
    score_cca = pdist2(proj_test_gallery,proj_test_probe,'cosine');
    cmc_cca = zeros(n_gallery,3);
    cmc_cca(:,1) = 1:n_gallery;
    for k=1:n_test
        final_score = score_cca(:,k);
        [sort_score, idx] = sort(final_score);
        cmc_cca = fn_eval_cmc(idxProbe(k),idxGallery(idx),cmc_cca);
    end
    

end
figure;
fn_plotcmc(cmc_cca,'r');

toc
