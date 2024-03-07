import os
path=("notebook/research.ipynb")
dir,file=os.path.split(path)
print(dir)
print(file)
os.makedirs(dir,exist_ok=True)  
with open(file,'w') as f:
    pass
