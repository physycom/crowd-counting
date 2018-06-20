SW_VER_CROWD = 100;


addpath('../MRF');
MRFParams = single([105 200 1.0]);% Shanghaitech Part_A
%MRFParams = single([200 200 8]);% Shanghaitech Part_B

load([base '.phase_0.mat']);
load([base '.phase_1.mat']);

%%
[height, width, mille] = size(features);
p = reshape(predictions, width, height);
    
% The marginal data of the predicted count matrix is 0 after apply MRF, 
% so first extending the predicted count matrix by copy marginal data.
p = uint8(p)';
p = [p(1,:); p];
p = [p ;p(end,:)];
p = [p(:, 1) p];
p = [p p(:, end)];
% apply MRF
p = MRF(p, MRFParams);
p = p(2:end-1, 2: end-1);

[row, column] = size(p);
C = p;
C(2:2:(row - 1), :) = 0;
C(:, 2 : 2 : (column - 1)) = 0;
if mod(row, 2) == 0
    C(row, :) = C(row, :) / 2;  
end
if mod(column, 2) == 0
    C(:, column) = C(:, column) / 2;  
end

finalcount = sum(sum(C));

disp(finalcount)

tok = strsplit(base, '_');
loctag = tok{1};
timestamp = tok{2};
jsonout = strcat(base, '.json');
info_json = fopen(jsonout,'w');
fprintf(info_json, '{\n');
fprintf(info_json, '\t\"frame_00000\" : {\n');
fprintf(info_json, '\t\t\"timestamp\" : %s,\n', timestamp);
fprintf(info_json, '\t\t\"id_box\" : \"%s\",\n', loctag);
fprintf(info_json, '\t\t\"detection\" : \"%s\",\n', "crowd2");
fprintf(info_json, '\t\t\"sw_ver\" : %d,\n', SW_VER_CROWD);
fprintf(info_json, '\t\t\"people_count\" : [{\"id\" : \"%s\", \"count\" : %u}],\n', loctag, finalcount);
fprintf(info_json, '\t\t\"diagnostics\" : [{\"id\" : \"coming\", \"value\" : \"soon\"}]\n');
fprintf(info_json, '\t}\n');
fprintf(info_json, '}');


