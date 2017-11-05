import numpy as np
import tensorflow as tf
import PIL.Image
import argparse
from io import BytesIO

parser = argparse.ArgumentParser(description='Deep dream generator')
parser.add_argument('img_prefix', type=str)
parser.add_argument('save_prefix', type=str)
parser.add_argument('num_images', type=int)

args = parser.parse_args()
img_prefix = args.img_prefix
save_prefix = args.save_prefix
num_images = args.num_images

model_file = 'tensorflow_inception_graph.pb'

# Start a TF session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# Load graph from file
with tf.gfile.FastGFile(model_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create input layer
input_tensor = tf.placeholder(np.float32, name='input')

# Create/transform tensor for input
imagenet_mean = 117.0
preprocessed_tensor = tf.expand_dims(input_tensor - imagenet_mean, 0)

# Import graph with input mapping
tf.import_graph_def(graph_def, {'input': preprocessed_tensor})


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session = kw.get('session'))
        return wrapper
    return wrap


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def get_output_tensor(layer):
    return graph.get_tensor_by_name("import/{}:0".format(layer))


def calc_grad_tiled(img, grad_tensor, tile_size=512):
    sz = tile_size
    height, width = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(height - sz // 2, sz), sz):
        for x in range(0, max(height - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(grad_tensor, {input_tensor: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def save_img(a, filename, format='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    PIL.Image.fromarray(a).save(filename, format)


def render_dream(tensor_object, img0, filename='results/dream', num_iter=10, step=1.5, num_octaves=4, octave_scale=1.4):
    score_tensor = tf.reduce_mean(tensor_object)
    grad_tensor = tf.gradients(score_tensor, input_tensor)[0]

    img = img0
    octaves = []

    for i in range(num_octaves - 1):
        width = img.shape[:2]
        low = resize(img, np.int32(np.float32(width) / octave_scale))
        high = img - resize(low, width)
        img = low
        octaves.append(high)

    for octave in range(num_octaves):
        if octave > 0:
            high = octaves[-octave]
            img = resize(img, high.shape[:2]) + high
        for i in range(num_iter):
            grad = calc_grad_tiled(img, grad_tensor)
            img += grad * (step / (np.abs(grad).mean()+ 1e-7))
        save_img(img / 255.0, filename)

for i in range(1, num_images + 1):
    start_image = np.float32(PIL.Image.open('{}{}.jpg'.format(img_prefix, i)))
    try:
        render_dream(tf.square(get_output_tensor('mixed4c')), img0=start_image, filename='{}{}.jpg'.format(save_prefix, i))
        print('Image {} complete'.format(i))
    except:
        print('Failed to process image {}, skipping...'.format(i))
