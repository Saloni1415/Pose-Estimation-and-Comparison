
import tensorflow as tf
import pickle
import cv2
import numpy as np
import posenet
from pose import Pose
from score import Score
from dtaidistance import dtw


class get_Score(object):
	def __init__(self, lookup='lookup.pickle'):
		self.a = Pose()
		self.s = Score()
		self.b = pickle.load(open(lookup, 'rb'))
		self.input_test = []

	def get_action_coords_from_dict(self,action):
			for (k,v) in self.b.items():
				if k==action:
					(model_array,no_of_frames) = (v,v.shape[0])
			return model_array,no_of_frames
	
	def calculate_Score(self,video,action):
		with tf.compat.v1.Session() as sess:
			model_cfg, model_outputs = posenet.load_model(101, sess)
			model_array,j = self.get_action_coords_from_dict(action)
			# cap = cv2.VideoCapture(0)
			cap = cv2.VideoCapture(video)
			i = 0
			if cap.isOpened() is False:
				print("error in opening video")
			frame_count = 1
			while cap.isOpened():
				ret_val, image = cap.read()
				# cv2.imshow("windnow", image)
				# cv2.waitKey(1)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					cap.release()
					# Destroy all the windows
					cv2.destroyAllWindows()
					break

				# print("ret value",ret_val)
				# print("image ===",image.shape)
				# image = np.transpose(image, (0, 1, 2))
				# print("image newe===", image.shape)

				if ret_val:
					print("image shape 42...=====",image.shape)
					# input_points= self.a.getpoints(image,sess,model_cfg,model_outputs)
					input_points, input_black_image = self.a.getpoints_vis(image, sess, model_cfg, model_outputs, frame_count)
					input_points= self.a.getpoints(cv2.resize(image,(372,495)),sess,model_cfg,model_outputs)
					if len(input_points) !=0:
					# print(" input_points ====",input_points)
						input_new_coords = np.asarray(self.a.roi(input_points)[0:34]).reshape(17,2)
						self.input_test.append(input_new_coords)
						i = i + 1
						# im_with_keypoints= cv2.drawKeypoints(image, input_points[0], image, color=(255, 0, 0))
						# cv2.imshow("keypoints",im_with_keypoints)
						# cv2.waitKey(0)

				else:
					break
			cap.release()
			final_score,score_list = self.s.compare(np.asarray(self.input_test),np.asarray(model_array),j,i)
		return final_score,score_list



	
	


