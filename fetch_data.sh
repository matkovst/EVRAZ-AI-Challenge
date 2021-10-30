#!/bin/bash

# Download pretrained networks
wget https://www.dropbox.com/s/6ttxi3vb6e7kx4t/cdcl_pascal_model.zip?dl=1 -O ./CDCL/weights/cdcl_pascal_model.zip
unzip CDCL/weights/cdcl_pascal_model.zip -d CDCL/weights && rm CDCL/weights/cdcl_pascal_model.zip
wget https://www.dropbox.com/s/sknafz1ep9vds1r/cdcl_model.zip?dl=1 -O ./CDCL/weights/cdcl_model.zip
unzip CDCL/weights/cdcl_model.zip -d CDCL/weights && rm CDCL/weights/cdcl_model.zip