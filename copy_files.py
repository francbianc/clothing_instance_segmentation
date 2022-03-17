import os
import shutil

src = '.../' #@@@ OVERRIDE: Path of Source folder
dst = '.../' #@@@ OVERRIDE: Path of Destination folder

with open('.../names.txt', 'r') as f:
    names_train = [n.strip() for n in f.readlines()]

files_tr = [i for i in names_train]
index = 0
for f in files_tr:
    shutil.copy(os.path.join(src, f), os.path.join(dst, str(index).zfill(6)+'.jpg')) # copy and rename from 0
    index += 1

print('Done')
print(len(os.listdir(dst)))