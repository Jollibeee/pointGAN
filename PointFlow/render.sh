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
cd ~/coding/PointCloud/PointFlow
python npy2xml4render.py
echo ""
echo ""


echo "============== xml  -->  png =============="
cd ~/coding/PointCloud/PointFlow/
#mitsuba test.xml
# python render/batchrender.py -i render/xml -o render/img
python render/xml2png4render.py -i render/xml -o render/img
echo ""
echo ""



sudo rm -rf mitsuba.IPIS-sohee.log




