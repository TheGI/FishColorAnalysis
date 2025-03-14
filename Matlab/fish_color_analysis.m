%% Step 0: adjust these values based on the dataset
% path to the sample images
image_path = strcat(pwd,'/example_data/');
% number of groups under test
n_group = 4;
% number of samples per group
n_indiv = 9;

%% Step 1: Preprocessing
% in this step, we read all image files and convert rgb to hsv
% then we take histogram of hues for each image data

images=[dir([image_path '*.jpg']);dir([image_path '*.png']);dir([image_path '*.tif'])];
n_color = 256;
cmap = hsv(n_color);
mask_hsv_value = [1,1,1];
for i = 1:length(images)
    im_rgb = imread([image_path images(i).name]);
    im_hsv = rgb2hsv(im_rgb);
    if(i == 1) 
        imshow(im_hsv);
        [x,y] = ginput(1);
        x = round(x); y = round(y);
        mask_hsv_value = im_hsv(y, x, :);
        close;
    end

    mask = im_hsv(:,:,1) >= mask_hsv_value(:,:,1) * 0.98 & ...
           im_hsv(:,:,1) <= mask_hsv_value(:,:,1) * 1.02;
    im_hsv_mask = bsxfun(@times, im_hsv, cast(~mask, class(im_hsv)));
    hue = im_hsv_mask(:,:,1);
    hue = hue(hue~=0);
    [hue_hist(i,:), ~] = hist(hue(:), n_color);
    hue_hist_normal(i,:) = hue_hist(i,:)/norm(hue_hist(i,:));
end

%% Step 2: Overview of Group Color Histogram
hue_mean_idx = 1;
hue_mean = [];
hue_err = [];
for i = 1:n_indiv:n_group*n_indiv
    hue_mean(hue_mean_idx,:) = mean(hue_hist_normal(i:i+n_indiv-1,:));
    hue_err(hue_mean_idx,:) = std(hue_hist_normal(i:i+n_indiv-1,:))/sqrt(n_indiv);
    hue_mean_idx = hue_mean_idx+1;
end

i_c = find(sum(hue_mean,1) >0.0005);
i_c1 = find(hue_mean(1,:) >0.0005);
i_c2 = find(hue_mean(2,:) >0.0005);
i_c3 = find(hue_mean(3,:) >0.0005);
i_c4 = find(hue_mean(4,:) >0.0005);

hold on;
for i = 1:length(i_c)
    patch([i-.5,i+.5,i+.5,i-.5],[0,0,-.01,-.01],cmap(i_c(i),:),'FaceAlpha', 1);
end
p1 = plot(hue_mean(1,i_c),'k^-','MarkerSize',8,'MarkerFaceColor','r');
p2 = plot(hue_mean(2,i_c),'ko-','MarkerSize',8,'MarkerFaceColor','g');
p3 = plot(hue_mean(3,i_c),'ks-','MarkerSize',8,'MarkerFaceColor','b');
p4 = plot(hue_mean(4,i_c),'kp-','MarkerSize',8,'MarkerFaceColor','y');
hold off;

lgd = legend([p1,p2,p3,p4],...
    'Group 1','Group 2','Group 3','Group 4',...
    'Location','Best');
xlabel('Colors'), ylabel('Count');
ylim([-0.01,Inf]);
title('Histogram of Before and After Treatment for IRL and PBS');

%% Step 3: Which hues pass ANOVA?
% in this step, we run ANOVA to determine significant hues
% that are different amongst groups
hueHistNorm3D = reshape(hue_hist_normal,n_indiv,n_group,n_color);
i_cSig = [];
for j = 1:n_color
    p = anova1(hueHistNorm3D(:,:,j),[],'off');
    if p < 0.00001 && max(mean(hueHistNorm3D(:,:,j))) > 0.05
        i_cSig(end+1) = j;
    end
end

%% Step 4: Compare groups
% plot histograms of hue distribution between a pair of groups
i_cSig12 = [];
for j = i_cSig
    [hue,p12] =ttest2(hueHistNorm3D(:,2,j), hueHistNorm3D(:,3,j));
    if p12 < 0.001
        i_cSig12(end+1) = j;
    end
end

figure;
hold on;
for i = 1:length(i_cSig12)
    patch([i-.5,i+.5,i+.5,i-.5],[0,0,-.01,-.01],cmap(i_cSig12(i),:),'FaceAlpha', 1);
end
p2Sig = errorbar(hue_mean(2,i_cSig12),hue_err(2,i_cSig12));
p3Sig = errorbar(hue_mean(3,i_cSig12),hue_err(3,i_cSig12));
p2Sig = area(hue_mean(2,i_cSig12),'FaceColor','b','FaceAlpha',0.4);
p3Sig = area(hue_mean(3,i_cSig12),'FaceColor','y','FaceAlpha',0.4);
hold off;
lgd = legend([p2Sig,p3Sig],...
    'T1','T2',...
    'Location','Best');
set(lgd,'FontSize',30);
xlabel('Colors', 'FontSize', 16), ylabel('Proportion', 'FontSize', 16);
ylim([-0.01,Inf]);
title('Pair-wise T-Test between T1 and T2 (p < 0.01)','fontsize',20);

%
i_cSig23 = [];
for j = i_cSig
    [hue,p23] =ttest2(hueHistNorm3D(:,3,j), hueHistNorm3D(:,4,j));
    if p23 < 0.001
        i_cSig23(end+1) = j;
    end
end

figure;
hold on;
for i = 1:length(i_cSig23)
    patch([i-.5,i+.5,i+.5,i-.5],[0,0,-.01,-.01],cmap(i_cSig23(i),:),'FaceAlpha', 1);
end
p3Sig = errorbar(hue_mean(3,i_cSig23),hue_err(3,i_cSig23));
p4Sig = errorbar(hue_mean(4,i_cSig23),hue_err(4,i_cSig23));
p3Sig = area(hue_mean(3,i_cSig23),'FaceColor','y','FaceAlpha',0.4);
p4Sig = area(hue_mean(4,i_cSig23),'FaceColor','b','FaceAlpha',0.4);
hold off;
lgd = legend([p3Sig,p4Sig],...
    'T2','T3',...
    'Location','Best');
set(lgd,'FontSize',30);
xlabel('Colors', 'FontSize', 16), ylabel('Proportion', 'FontSize', 16);
ylim([-0.01,Inf]);
title('Pair-wise T-Test between T2 and T3 (p < 0.01)','fontsize',20);

%
i_cSig13 = [];
for j = i_cSig
    [hue,p13] =ttest2(hueHistNorm3D(:,2,j), hueHistNorm3D(:,4,j));
    if p13 < 0.001
        i_cSig13(end+1) = j;
    end
end

figure;
hold on;
for i = 1:length(i_cSig13)
    patch([i-.5,i+.5,i+.5,i-.5],[0,0,-.01,-.01],cmap(i_cSig13(i),:),'FaceAlpha', 1);
end
p2Sig = errorbar(hue_mean(2,i_cSig13),hue_err(2,i_cSig13));
p4Sig = errorbar(hue_mean(4,i_cSig13),hue_err(4,i_cSig13));
p2Sig = area(hue_mean(2,i_cSig13),'FaceColor','b','FaceAlpha',0.4);
p4Sig = area(hue_mean(4,i_cSig13),'FaceColor','r','FaceAlpha',0.4);
hold off;
lgd = legend([p2Sig,p4Sig],...
    'T1','T3',...
    'Location','Best');
set(lgd,'FontSize',30);
xlabel('Colors', 'FontSize', 16), ylabel('Proportion', 'FontSize', 16);
ylim([-0.01,Inf]);
title('Pair-wise T-Test between T1 and T3 (p < 0.01)','fontsize',20);

%% Step 5: plot hue histogram of all groups
figure;
hold on;
for i = 1:length(i_cSig)
    patch([i-.5,i+.5,i+.5,i-.5],[0,0,-.01,-.01],cmap(i_cSig(i),:),'FaceAlpha', 1);
end

p2Sig = errorbar(hue_mean(2,i_cSig),hue_err(2,i_cSig));
p3Sig = errorbar(hue_mean(3,i_cSig),hue_err(3,i_cSig));
p4Sig = errorbar(hue_mean(4,i_cSig),hue_err(4,i_cSig));
p2Sig = area(hue_mean(2,i_cSig),'FaceColor','b','FaceAlpha',0.4);
p3Sig = area(hue_mean(3,i_cSig),'FaceColor','y','FaceAlpha',0.4);
p4Sig = area(hue_mean(4,i_cSig),'FaceColor','r','FaceAlpha',0.4);

hold off;
lgd = legend([p2Sig,p3Sig,p4Sig],...
    'T1','T2','T3',...
    'Location','Best');
set(lgd,'FontSize',30);
xlabel('Colors', 'FontSize', 16), ylabel('Proportion', 'FontSize', 16);
ylim([-0.01,Inf]);
title('ANOVA among T1, T2 and T3 (p < 0.00001)','fontsize',20);


%% Step 6: plot Chi-Square Histogram Distance
dist_func=@chi_square_statistics;
D=pdist2(hue_hist_normal,hue_hist_normal,dist_func);
hm = HeatMap(D);
hm.addTitle('Chi-Square Distance Heat Map');

% %% Optional: Eigenspace Analysis
% cSig = cmap(i_cSig,:);
% c1 = cmap(i_c1,:);
% c2 = cmap(i_c2,:);
% c3 = cmap(i_c3,:);
% c4 = cmap(i_c4,:);
% cSigM = mean(cSig);
% cSigV = cov(cSig);
% [V,D] = eig(cSigV);
% S = zeros(3,3);
% S(:,1) = V(:,3)*D(3,3);
% S(:,2) = V(:,2)*D(2,2);
% S(:,3) = V(:,1)*D(1,1);
% 
% S = S/norm(S);
% 
% scatter3(cSig(:,1),cSig(:,2),cSig(:,3),200,cSig,'filled');
% xlabel('red'), ylabel('green'), zlabel('blue');
% title('Most Significantly Different Colors with Eigenvectors');
% axis equal;
% view(19,31);
% hold on;
% quiver3(repmat(cSigM(1),1,3),repmat(cSigM(2),1,3),repmat(cSigM(3),1,3),...
%     S(1,:),S(2,:),S(3,:));
% hold off;
% grid on;
% 
% %% Optional: plot mean color point clouds for T0, T1, T2, T3
% figure;
% hold on;
% scatter3(c1(:,1) + (rand(length(c1(:,1)),1)-0.5)*0.05,...
%     c1(:,2) + (rand(length(c1(:,1)),1)-0.5)*0.05,...
%     c1(:,3) + + (rand(length(c1(:,1)),1)-0.5)*0.05,...
%     100,'r','filled','^');
% scatter3(c2(:,1) + (rand(length(c2(:,1)),1)-0.5)*0.05,...
%     c2(:,2) + (rand(length(c2(:,1)),1)-0.5)*0.05,...
%     c2(:,3) + (rand(length(c2(:,1)),1)-0.5)*0.05,...
%     100,'g','filled','s');
% scatter3(c3(:,1) + (rand(length(c3(:,1)),1)-0.5)*0.05,...
%     c3(:,2) + (rand(length(c3(:,1)),1)-0.5)*0.05,...
%     c3(:,3) + (rand(length(c3(:,1)),1)-0.5)*0.05,...
%     100,'b','filled','^');
% scatter3(c4(:,1) + (rand(length(c4(:,1)),1)-0.5)*0.05,...
%     c4(:,2) + (rand(length(c4(:,1)),1)-0.5)*0.05,...
%     c4(:,3) + (rand(length(c4(:,1)),1)-0.5)*0.05,...
%     100,'y','filled','^');
% legend('T0','T1','T2','T3','Location','Best');
% % legend('Yellow Group','Blue Grouop','Location','Best');
% quiver3(repmat(cSigM(1),1,3),repmat(cSigM(2),1,3),repmat(cSigM(3),1,3),...
%     S(1,:),S(2,:),S(3,:));
% hold off;
% xlabel('red'), ylabel('green'), zlabel('blue');
% title('Mean Per Group Colors with Eigenvectors');
% axis equal;
% view(17,18);
% grid on;
% figure;
% 
% % Principal Component Color Complements
% for i = 1:3
%     ind = dsearchn(cSig,S(:,i)'*10);
%     eigcolors1(i,:)=cSig(ind,:);
%     ind = dsearchn(cSig,-S(:,i)'*10);
%     eigcolors2(i,:)=cSig(ind,:);
% end
% 
% for i = 1:3
%     patch([i-1 i i i-1], [0 0 0.5 0.5], eigcolors1(i,:))
%     patch([i-1 i i i-1], [0.5 0.5 1 1], eigcolors2(i,:))
% end
% 
