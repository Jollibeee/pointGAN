# %%
import argparse
import numpy as np
import os
from tqdm import tqdm
import time
from os.path import isfile, isdir, join


# ## input
# # pcl_file = 'pretrained_models/gen/render_0.npy'
# pcl_file = 'chair/single_class_ae/ae_npy/ae_recon_00000.npy'
# out_name = 'ae_recon_00000'

# xml_folder = 'render/xml'
# img_f = 'render/img'




def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    # print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result

#http://latexcolor.com/

xml_head = \
    """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
    
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="1600"/>
                <integer name="height" value="1200"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
    
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>   
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->            
        </bsdf>
    
    """

xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>                
            </bsdf>
        </shape>
    """

xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
    
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


# xml_segments = [xml_head]




def save_xml_dim2(pcl):
    pcl = standardize_bbox(pcl, 2048)
    # pcl = pcl[:, [2, 0, 1]]
    pcl = pcl[:, [1, 0, 2]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        # color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125)
        color = [0.63, 0.79, 0.95]
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    xml_file = xml_folder + "/" + out_name + '.xml'
    with open(xml_file, 'w') as f:
        f.write(xml_content)

# [0.96, 0.76, 0.76] # BABY PINK

def save_xml_dim3(pcl):
    for n in tqdm(range(pcl.shape[0])):

        pcl2 = pcl[n]
        pcl2 = standardize_bbox(pcl2, 2048)
        # pcl2 = pcl2[:, [2, 0, 1]]
        pcl2 = pcl2[:, [1, 0, 2]]
        pcl2[:, 0] *= -1
        pcl2[:, 2] += 0.0125

        xml_segments = [xml_head]

        for i in range(pcl2.shape[0]):
            # color = colormap(pcl2[i, 0] + 0.5, pcl2[i, 1] + 0.5, pcl2[i, 2] + 0.5 - 0.0125)
            color = [0.63, 0.79, 0.95] # BABY BLUE EYES
            xml_segments.append(xml_ball_segment.format(pcl2[i, 0], pcl2[i, 1], pcl2[i, 2], *color))
        xml_segments.append(xml_tail)

        xml_content = str.join('', xml_segments)

        xml_file = xml_folder + "/" + out_name + "_" + str(n).zfill(4) + '.xml'
        # if isfile(xml_file) == True:
        # 	continue
        with open(xml_file, 'w') as f:
            f.write(xml_content)

        xml_segments.clear()
        xml_content = ""

       	time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch render with mitsuba')
    parser.add_argument('-i', '--pcl_file', help="input npy path", required=True)
    parser.add_argument('-o', '--out_name', help="output name", required=True)

    parser.add_argument('-xf', '--xml_folder', help="output xml path", required=True)
    parser.add_argument('-if', '--img_folder', help="output image path", required=True)

    args = parser.parse_args()

    pcl_file = args.pcl_file
    out_name = args.out_name
    xml_folder = args.xml_folder
    img_folder = args.img_folder

    createFolder(xml_folder)
    createFolder(img_folder)

    pcl = np.load(pcl_file)
    if pcl.ndim == 2:
        save_xml_dim2(pcl)

    elif pcl.ndim == 3:
        save_xml_dim3(pcl)

