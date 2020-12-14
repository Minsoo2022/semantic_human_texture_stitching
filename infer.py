import step1_make_unwraps, step2_segm_vote_gmm, step3_stitch_texture
import os
import argparse
import numpy as np
from opendr.topology import loop_subdivider
from psbody.mesh import Mesh
from glob import glob





parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str,)
args = parser.parse_args()
name = ' '.join((args.dir).split('/')).split()[-1]
os.makedirs(os.path.join(args.dir,'unwarps'),exist_ok=True)
step1_make_unwraps.main(os.path.join(args.dir,'frame_data.pkl'),os.path.join(args.dir,'frames'),os.path.join(args.dir,'segmentations'),os.path.join(args.dir,'unwarps'))
step2_segm_vote_gmm.main(os.path.join(args.dir,'unwarps'),os.path.join(args.dir,'segm.png'),os.path.join(args.dir,'gmm.pkl'))
step3_stitch_texture.main(os.path.join(args.dir,'unwarps'), os.path.join(args.dir,'segm.png'), os.path.join(args.dir,'gmm.pkl'), os.path.join(args.dir,name+'.jpg'),50)

filename = os.path.join(args.dir,name+'.obj')
body = Mesh(filename=filename)
body_tex = filename.replace('.obj', '.jpg')
if not os.path.exists(body_tex):
    body_tex = 'tex_{}'.format(body_tex)

v, f = body.v, body.f
(mapping, hf) = loop_subdivider(v, f)
hv = mapping.dot(v.ravel()).reshape(-1, 3)
body_hres = Mesh(hv, hf)

vt, ft = np.hstack((body.vt, np.ones((body.vt.shape[0], 1)))), body.ft
(mappingt, hft) = loop_subdivider(vt, ft)
hvt = mappingt.dot(vt.ravel()).reshape(-1, 3)[:, :2]
body_hres.vt, body_hres.ft = hvt, hft

body_hres.set_texture_image(body_tex)
body_hres.write_obj(os.path.join(filename.replace('.obj', '_hres.obj')))
print(filename)
