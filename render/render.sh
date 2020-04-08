#! /bin/bash

conda activate PointFlow


echo "============= mitsuba setpath ============="
cd ~/tmp/mitsuba/
export MITSUBA_PYVER=3.6
source setpath.sh
echo "                    OK"
echo ""
echo ""

echo "============== npy  -->  xml =============="
cd ~/coding/PointCloud/render
python npy2xml4render.py
echo ""
echo ""


echo "============== xml  -->  png =============="
cd ~/coding/PointCloud/render
#mitsuba test.xml
# python render/batchrender.py -i render/xml -o render/img
python xml2png4render.py -i res/xml -o res/img
echo ""
echo ""



sudo rm -rf mitsuba.IPIS-sohee.log




