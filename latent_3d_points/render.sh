#! /bin/bash

conda activate PointFlow

class_name=chair

pcl_file=$class_name/single_class_ae_chamfer/ae_npy/ae.npy
out_name=${class_name}_ae
xml_folder=$class_name/single_class_ae_chamfer/render/xml
img_folder=$class_name/single_class_ae_chamfer/render/img


echo "============= mitsuba setpath ============="
cd ~/tmp/mitsuba/
export MITSUBA_PYVER=3.6
source setpath.sh
echo "                    OK"
echo ""
echo ""

echo "============== npy  -->  xml =============="
cd ~/coding/PointCloud/lgan/latent_3d_points/data
#python data/npy2xml4render.py
python npy2xml4render.py -i $pcl_file -o $out_name -xf $xml_folder -if $img_folder
echo ""
echo ""


echo "============== xml  -->  png =============="
cd ~/coding/PointCloud/lgan/latent_3d_points/data
#mitsuba test.xml
# python render/batchrender.py -i render/xml -o render/img
#python data/xml2png4render.py -i render/xml -o render/img
python xml2png4render.py -i $xml_folder -o $img_folder
echo ""
echo ""

sudo rm -rf mitsuba.IPIS-sohee.log

cd ~/coding/PointCloud/lgan/latent_3d_points




