clc;
img_dir = 'C:\Users\adith\Downloads\FrogAnalysis';%image folder directory
cd (img_dir)

img_list_v=[dir(['*ropped*/*v.*']); dir(['*ropped*/*Ven*']);dir(['*ropped*/* V *'])]
img_list_d=[dir(['*ropped*/*d.*']); dir(['*ropped*/*Dor*']);dir(['*ropped*/* D *'])]
v={img_list_v.name} %list all "ventral" images.
v_num= numel(v);

for i = 1: v_num
    temp = strsplit(v{i},{'v.','Ventr',' V'});
    v_split{i} = temp{1};
end
v_split
    
  
d={img_list_d.name} %list all "dorsal" images.
d_num= numel(d)
for i = 1: d_num
    temp = strsplit(d{i},{'d.','Dors',' D'});
    d_split{i} = temp{1};
end
d_split

for i = 1: v_num
 if any(strcmp(v_split{i}, d_split)) == 0
   disp(['missng d: ' v_split{i}])
 end
end


 for i = 1: d_num
 if any(strcmp(d_split{i}, v_split)) == 0
   disp(['missing v: ' d_split{i}])
 end
end
