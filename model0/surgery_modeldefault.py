# Make sure that caffe is on the python path:

caffe_root = './caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
net_full_conv = caffe.Net('deploy.prototxt', 
                          '/lustre/yixi/face_segmentation_finetune/flow/modeldefault/snapshots_camvid200200/train_lr1e-10/_iter_77000.caffemodel')
net_full_conv.set_phase_test()


caffe_root = './caffe-future/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
net = caffe.Net('modeldefault_deploy_ori.prototxt', 
                '/lustre/yixi/face_segmentation_finetune/flow/modeldefault/snapshots_camvid200200/train_lr1e-10/_iter_77000.caffemodel', caffe.TEST)
#net.set_phase_test()



params_full_conv = ['fc6', 'fc7', 'score-fr-camvid', 'score2-camvid', 'score-pool4-camvid', 'score4-camvid', 'score-pool3-camvid', 'upsample-camvid']
conv_params = {pr: [net_full_conv.params[pr][i].data for i in range(len(net_full_conv.params[pr]))] for pr in params_full_conv}
for conv in params_full_conv:
	print '{} weights are {} dimensional'.format(conv, conv_params[conv][0].shape) + ('' if len(conv_params[conv])<=1 else 'and biases are {} dimensional'.format(conv_params[conv][1].shape))
	for i in range(2, len(conv_params[conv])):
		print 'additional weights are {}'.format(conv_params[conv][i].shape)



params = ['fc6-conv', 'fc7-conv', 'score59bg', 'upscore2bg', 'score-pool4bg', 'upsample-fused-16bg', 'score-pool3bg', 'upsamplebg']
fc_params = {pr: [net.params[pr][i].data for i in range(len(net.params[pr]))] for pr in params}
for fc in params:
	print '{} weights are {} dimensional'.format(fc, fc_params[fc][0].shape) + ('' if len(fc_params[fc])<=1 else 'and biases are {} dimensional'.format(fc_params[fc][1].shape))
	for i in range(2, len(fc_params[fc])):
		print 'additional weights are {}'.format(fc_params[fc][i].shape)





for pr, pr_conv in zip(params, params_full_conv):
	for i in range(len(conv_params[pr_conv])):
		conv_params[pr_conv][i][0:fc_params[pr][i].shape[0], 0:fc_params[pr][i].shape[1], 0:fc_params[pr][i].shape[2], 0:fc_params[pr][i].shape[3]] = fc_params[pr][i]




net_full_conv.save('modeldefault_finetuned.caffemodel')

