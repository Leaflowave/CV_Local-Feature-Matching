import cv2
import numpy as np
import match_functions as mf
from utils import *
import time

if __name__ == '__main__':
    t_start = time.clock()
    
    # 1) Loads and resizes images
#    image1 = cv2.imread('../data/Notre Dame/img1.jpg').astype('single') #2048*1536*3
#    image2 = cv2.imread('../data/Notre Dame/img2.jpg').astype('single') #2032*1524*3
#    eval_file = '../data/Notre Dame/img1_to_img2.npy'
    
#    image1 = cv2.imread('../data/Mount Rushmore/img1.jpg').astype('single') #2048*1536*3
#    image2 = cv2.imread('../data/Mount Rushmore/img2.jpg').astype('single') #2032*1524*3
#    eval_file = '../data/Mount Rushmore/img1_to_img2.npy'
#    
#    image1 = cv2.imread('../data/Episcopal Gaudi/img1.jpg').astype('single') #2048*1536*3
#    image2 = cv2.imread('../data/Episcopal Gaudi/img2.jpg').astype('single') #2032*1524*3
#    eval_file = '../data/Episcopal Gaudi/img1_to_img2.npy'
    
#    image1 = cv2.imread('../data/Office/img1.png').astype('single') #2048*1536*3
#    image2 = cv2.imread('../data/Office/img2.png').astype('single') #2032*1524*3
    
    image1 = cv2.imread('../data/Name board/img1.png').astype('single') #2048*1536*3
    image2 = cv2.imread('../data/Name board/img2.png').astype('single') #2032*1524*3

  

    image1 = image1 / 255.0
    image2 = image2 / 255.0
    
    # make images smaller to speed up the algorithm
    scale_factor = 0.5
    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR)

    # change to gray-scale images
    image1_bw = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) #1024*768
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) #1016*762

    # width and height of each local feature, in pixels. 
    feature_width = 16

    # 2) Find distinctive points in each image. Szeliski 4.1.1
    # TODO: You need to implement get_interest_points()
#     Uncomment this line when you finish
#    x1, y1 = mf.get_interest_points(image1_bw, feature_width)
#    x2, y2 = mf.get_interest_points(image2_bw, feature_width)

    # Use cheat_interest_points only for development and debugging!
    # Uncomment this line to use. Comment this line when you finish get_interest_points().
#    [x1, y1, x2, y2] = cheat_interest_points(eval_file, scale_factor);
     
    # Visualize ground truth correspondance. 
    # Comment this if your do not have ground truth
#    show_correspondence(image1, image2, x1, y1, x2, y2, 'truth')
#    show_correspondence2(image1, image2, x1, y1, x2, y2, 'truth')

#    # Evaluate ground truth correspondance. 
#    # Comment this if your do not have ground truth
#    evaluate_correspondence(image1, image2, eval_file, scale_factor, x1, y1, x2, y2, 'truth')
#
#     
#    # 3) Create feature vectors at each interest point. Szeliski 4.1.2
#    # TODO: You need to implement get_features().
    t_im_get_point = time.clock()
    x1, y1 = mf.get_interest_points(image1_bw, feature_width)
    x2, y2 = mf.get_interest_points(image2_bw, feature_width)
    t_im_get_point_done = time.clock()
   
#    np.savez('var',x1,y1,x2,y2)
    t_get_features = time.clock()
    image1_features = mf.get_features(image1_bw, x1, y1, feature_width)
    image2_features = mf.get_features(image2_bw, x2, y2, feature_width)
    t_get_features_done = time.clock()

    
    # 4) Match features. Szeliski 4.1.3
    # TODO: You need to implement match_features()
    t_match_features = time.clock()
    matches, confidences = mf.match_features(image1_features, image2_features)
    t_match_features_done = time.clock()

    
    # 5) Visualization
    # You might want to set 'num_pts_to_visualize' and 'num_pts_to_evaluate' to
    # some constant (e.g. 100) once you start detecting hundreds of interest
    # points, otherwise things might get too cluttered. You could also
    # threshold based on confidence.

    # There are two visualization functions. You can comment out one of both of
#    # them if you prefer.
    num_pts_to_visualize = 150
    x1_m = x1[matches[:num_pts_to_visualize, 0].astype(np.int32)]
    y1_m = y1[matches[:num_pts_to_visualize, 0].astype(np.int32)]
    x2_m = x2[matches[:num_pts_to_visualize, 1].astype(np.int32)]
    y2_m = y2[matches[:num_pts_to_visualize, 1].astype(np.int32)]
    
    
#    # visualize your matching result
    show_correspondence(image1, image2, x1_m, y1_m, x2_m, y2_m, 'result')
    show_correspondence2(image1, image2, x1_m, y1_m, x2_m, y2_m, 'result')

    # 6) Evaluation
    # This evaluation function will only work for the Notre Dame, Episcopal
    # Gaudi, and Mount Rushmore image pairs. Comment out this function if you
    # are not testing on those image pairs. Only those pairs have ground truth
    # available. You can use collect_ground_truth_corr.m to build the ground
    # truth for other image pairs if you want, but it's very tedious. It would
    # be a great service to the class for future years, though!
    num_pts_to_evaluate = matches.shape[0]
#    x1_m = x1[matches[:num_pts_to_evaluate, 0].astype(np.int32)]
#    y1_m = y1[matches[:num_pts_to_evaluate, 0].astype(np.int32)]
#    x2_m = x2[matches[:num_pts_to_evaluate, 1].astype(np.int32)]
#    y2_m = y2[matches[:num_pts_to_evaluate, 1].astype(np.int32)]
#    evaluate_correspondence(image1, image2, eval_file, scale_factor, \
#                        x1_m, y1_m, x2_m, y2_m, 'result')
    t_end = time.clock()
    print('The total time is %fs\n' % (t_end - t_start))
    print('The get_interest_points function processing time %fs\n'%(t_im_get_point_done-t_im_get_point))
    print('The get_features function processing time are %fs\n'% (t_get_features_done-t_get_features))
    print('The match_features function processing time are %fs\n'% (t_match_features_done-t_match_features))
