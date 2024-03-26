import subprocess
import fileinput
import os

# The file containing the list of values
list_file = [
        "dataloaders/dn-real+dn-sd14.pkl",
        "dataloaders/dn-real+dn-glide.pkl",
        "dataloaders/dn-real+dn-mj.pkl",
        "dataloaders/dn-real+dn-dallemini.pkl",
        "dataloaders/dn-real+dn-tt.pkl",
        "dataloaders/dn-real+dn-sd21.pkl",
        "dataloaders/dn-real+dn-cips.pkl",
        "dataloaders/dn-real+dn-biggan.pkl",
        "dataloaders/dn-real+dn-vqdiff.pkl",
        "dataloaders/dn-real+dn-diffgan.pkl",
        "dataloaders/dn-real+dn-sg3.pkl",
        "dataloaders/dn-real+dn-gansformer.pkl",
        "dataloaders/dn-real+dn-dalle2.pkl",
        "dataloaders/dn-real+dn-ld.pkl",
        "dataloaders/dn-real+dn-eg3d.pkl",
        "dataloaders/dn-real+dn-projgan.pkl",
        "dataloaders/dn-real+dn-sd1.pkl",
        "dataloaders/dn-real+dn-ddg.pkl",
        "dataloaders/dn-real+dn-ddpm.pkl",
]

# The script file
script_file = "scripts/syn2_udil.sh"

# copy the content of the script file to a new file
with open(script_file, 'r') as script_file:
    filedata = script_file.read()
    new_file = "scripts/syn2_udil_new.sh"
    for line in list_file:
        with open(new_file, 'w') as new:
            new.write(filedata)
            new.write("--new-pkl=" + line)
        os.system("chmod +x " + new_file)
        os.system("scripts/syn2_udil_new.sh")
