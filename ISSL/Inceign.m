function uu2 = Inceign(vtmp,k,bins,g,ii,jj)
% LOWER = 1/100; %% clip lowest CLIP_MARIN percent
% UPPER = 1-LOWER; % clip symmetrically
% 
% for a=1:size(vtmp,2)
%     [clip_lower(a),clip_upper(a)] = percentile(vtmp(:,a),LOWER,UPPER);
%     q = vtmp(:,a)<clip_lower(a);
%     %set all values below threshold to be constant
%     vtmp(q,a) = clip_lower(a);
%     q2 = vtmp(:,a)>clip_upper(a);
%     %set all values above threshold to be constant
%     vtmp(q2,a) = clip_upper(a);
% 
% end

uu1 = zeros(k,k);
uu2 = zeros(size(vtmp,1),k);
bins_out = bins(:,jj(1:k));

for a=1:k
    uu1(:,a) = g(:,ii(a),jj(a));
    uu2(:,a) = interp1(bins_out(:,a),uu1(:,a),vtmp(:,jj(a)),'linear','extrap');
end
suu2 = sqrt(sum(uu2.^2));
uu2 = uu2 ./ (ones(size(vtmp,1),1) * suu2);