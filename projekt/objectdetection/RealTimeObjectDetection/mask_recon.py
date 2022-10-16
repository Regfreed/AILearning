#%%
#POSTAVLJANJE PATHOVA

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/'


#postavljanje labela

labels = [{'name':'mask', 'id':1}, {'name':'no mask', 'id':2}]

with open(ANNOTATION_PATH + '/label_map.txt', 'w') as writer:
    for label in labels:
        writer.writelines('item{\n')
        writer.writelines('\tname:\'{}\'\n'.format(label['name']))
        writer.writelines('\tid:{}\n'.format(label['id']))
        writer.writelines('}\n')


#TF records
!python {SCRIPTS_PATH +'/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH +'/label_map.txt'} -o {ANNOTATION_PATH +'/train.record'}
!python {SCRIPTS_PATH +'/generate_tfrecord.py'} -x {IMAGE_PATH + '/test'} -l {ANNOTATION_PATH +'/label_map.txt'} -o {ANNOTATION_PATH +'/test.record'}

# %%
