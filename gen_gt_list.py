import codecs

def gennerate_gt(gt, Annotation, frame, filename, width, height):
    gt_lines = gt

    gt_fram = []
    for line in gt_lines:
        fram_id = int(line.split(',')[0])
        if fram_id == frame:
            visible = float(line.split(',')[8])
            label_class = line.split(',')[7]
            if (label_class == '1' or label_class == '2' or label_class == '7') and visible > 0.3:
                gt_fram.append(line)

    with codecs.open(Annotation + filename + '.xml', 'w') as xml:
        xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'voc' + '</folder>\n')
        xml.write('\t<filename>' + filename + '.jpg' + '</filename>\n')
        # xml.write('\t<path>' + path + "/" + info1 + '</path>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database> The MOT-Det </database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + '3' + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for bbox in gt_fram:
            x1 = int(bbox.split(',')[2])
            y1 = int(bbox.split(',')[3])
            x2 = int(bbox.split(',')[4])
            y2 = int(bbox.split(',')[5])

            xml.write('\t<object>\n')
            xml.write('\t\t<name>person</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>0</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(x1) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(y1) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(x1 + x2) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(y1 + y2) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
        xml.write('</annotation>')

import os
folder = 'data/MOT20/train/MOT20-02/'
imgs = []
fp_gt = open(folder+'gt/gt.txt')
gt_lines = fp_gt.readlines()
entries = []
for (dirpath, dirnames, filenames) in os.walk(folder+'img1'):
    imgs.extend(filenames)
for i in imgs:
    gennerate_gt(gt_lines,folder+'/ann/',int(i[:-4]),i[:-4],19dsad20,10dds80)
    entries.append(folder+'img1/'+i+' '+folder+'ann/' + i[:-4] + '.xml\n')

with open(folder+'manifest.txt','w') as f:
    f.writelines(entries)