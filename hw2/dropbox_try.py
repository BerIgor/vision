from pathlib import Path
import urllib.request
from os import getcwd

pwd = getcwd().replace('\\','//')
model_path = pwd + '/q3/pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_68.pth'

my_file = Path(model_path)
if not my_file.is_file():
    print("Downloading File...")
    url = "https://www.dropbox.com/s/91wcu6bpqezu4br/resnet_34_8s_68.pth?dl=1"  # dl=1 is important
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    with open("weights.pth", "wb") as f:
        f.write(data)
        
    input("Press Enter to continue...")

else:
    print("File exists")
    input("Press Enter to continue...")
