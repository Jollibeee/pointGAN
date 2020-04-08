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
cd ~/coding/PointCloud/PointFlowRenderer
python npy2xml.py
echo ""
echo ""


echo "============== xml  -->  png =============="
mitsuba test.xml
echo ""
echo ""
