import numpy as np
import cv2

def cheat_interest_points(eval_file, scale_factor):
    """ 
    This function is provided for development and debugging but cannot be
    used in the final handin. It 'cheats' by generating interest points from
    known correspondences. It will only work for the three image pairs with
    known correspondences.

    Args:
        eval_file: the file path to the list of known correspondences
        scale_factor: needed to map from the original image coordinates to
            the resolution being used for the current experiment.

    Returns:
        x1 and y1: nx1 vectors of x and y coordinates of interest points
            in the first image.
        x2 and y2: mx1 vectors of x and y coordinates of interest points
            in the second image. 
        For convenience, n will equal m, but don't expect that to be the 
            case when interest points are created independently per image.
    """
    data = np.load(eval_file).tolist()
    x1 = data['x1']  
    y1 = data['y1']
    x2 = data['x2']
    y2 = data['y2']
    x1 = x1 * scale_factor
    y1 = y1 * scale_factor
    x2 = x2 * scale_factor
    y2 = y2 * scale_factor

    return x1, y1, x2, y2 #they are float...

def show_correspondence(imgA, imgB, X1, Y1, X2, Y2, file_name='result'):
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    if len(imgA.shape)==2:
        imgA = np.expand_dims(imgA, 2)
    if len(imgB.shape)==2:
        imgB = np.expand_dims(imgB, 2)
    numColors = imgA.shape[2]
    newImg = np.zeros((Height, Width, numColors))
    newImg[:imgA.shape[0],:imgA.shape[1],:] = imgA
    newImg[:imgB.shape[0],imgA.shape[1]:,:] = imgB
    shiftX = imgA.shape[1]
    for i in xrange(X1.shape[0]):
        cur_color = np.random.rand(3)
        cv2.circle(newImg, (X1[i], Y1[i]), 9, cur_color, -1)
        cv2.circle(newImg, (X1[i], Y1[i]), 9, [0, 0, 0], 2)
        cv2.circle(newImg, (X2[i]+shiftX, Y2[i]), 9, cur_color, -1)
        cv2.circle(newImg, (X2[i]+shiftX, Y2[i]), 9, [0, 0, 0], 2)

    print('Saving visualization to vis_dots_' + file_name + '.png')
    cv2.imwrite('vis_dots_'+file_name+'.png', newImg*255.0)

def show_correspondence2(imgA, imgB, X1, Y1, X2, Y2, file_name='result'):
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    if len(imgA.shape)==2:
        imgA = np.expand_dims(imgA, 2)
    if len(imgB.shape)==2:
        imgB = np.expand_dims(imgB, 2)
    numColors = imgA.shape[2]
    newImg = np.zeros((Height, Width, numColors))
    newImg[:imgA.shape[0],:imgA.shape[1],:] = imgA
    newImg[:imgB.shape[0],imgA.shape[1]:,:] = imgB
    shiftX = imgA.shape[1]
    for i in xrange(X1.shape[0]):
        cur_color = np.random.rand(3)
        cv2.circle(newImg, (X1[i], Y1[i]), 5, cur_color, -1)
        cv2.circle(newImg, (X2[i]+shiftX, Y2[i]), 5, cur_color, -1)
        cv2.line(newImg, (X1[i], Y1[i]), (X2[i]+shiftX, Y2[i]), cur_color, 2)

    print('Saving visualization to vis_arrows_' + file_name + '.png')
    cv2.imwrite('vis_arrows_'+file_name+'.png', newImg*255.0)

def evaluate_correspondence(imgA, imgB, ground_truth_correspondence_file, scale_factor, x1_est, y1_est, x2_est, y2_est, file_name='result'):
    """
    You do not need to modify anything in this function, although you can if
    you want to.

    """
    print('\n')
    print('-------------------------------- Start Evaluation --------------------------------\n')
    x1_est = x1_est / scale_factor
    y1_est = y1_est / scale_factor
    x2_est = x2_est / scale_factor
    y2_est = y2_est / scale_factor

    good_matches = np.zeros((x1_est.shape[0], 1)) # indicator vector

    # loads variables x1, y1, x2, y2
    data = np.load(ground_truth_correspondence_file).tolist()
    x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']

    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]
    numColors = imgA.shape[2]
    newImg = np.zeros((Height, Width, numColors))
    newImg[:imgA.shape[0],:imgA.shape[1],:] = imgA
    newImg[:imgB.shape[0],imgA.shape[1]:,:] = imgB
    shiftX = imgA.shape[1]

    for i in xrange(x1_est.shape[0]):

        # for each x1_est, find nearest ground truth point in x1
        x_dists = x1_est[i] - x1
        y_dists = y1_est[i] - y1
        dists = np.sqrt(x_dists**2 + y_dists**2)
        
        # sort the distance
        best_matches = np.argsort(dists, axis=0)[:, 0]
        dists = dists[best_matches][:, 0]
        
        current_offset = [x1_est[i] - x2_est[i], y1_est[i] - y2_est[i]]
        most_similar_offset = [x1[best_matches[0]] - x2[best_matches[0]], 
                               y1[best_matches[0]] - y2[best_matches[0]]];
        
        match_dist = np.sqrt(np.sum((np.array(current_offset) - np.array(most_similar_offset))**2))
        
        # A match is bad if there's no ground truth point within 150 pixels or
        # if nearest ground truth correspondence offset isn't within 25 pixels
        # of the estimated correspondence offset.        
        if (dists[0] > 150 or match_dist > 40):
            good_matches[i] = 0
            edgeColor = [0, 0, 1]
            flag_str = 'Wrong:   '
        else:
            good_matches[i] = 1
            edgeColor = [0, 1, 0]
            flag_str = 'Correct: '
        print('%s\t(%4.0f, %4.0f) to (%4.0f, %4.0f): \tg.t. point %.0f px. Match error %.0f px.' % \
            (flag_str, x1_est[i], y1_est[i], x2_est[i], y2_est[i], dists[0], match_dist))
        

        cur_color = np.random.rand(3)

        cv2.circle(newImg, (int(x1_est[i]*scale_factor), int(y1_est[i]*scale_factor)), 9, cur_color, -1)
        cv2.circle(newImg, (int(x1_est[i]*scale_factor), int(y1_est[i]*scale_factor)), 9, edgeColor, 2)
        cv2.circle(newImg, (int(x2_est[i]*scale_factor+shiftX), int(y2_est[i]*scale_factor)), 9, cur_color, -1)
        cv2.circle(newImg, (int(x2_est[i]*scale_factor+shiftX), int(y2_est[i]*scale_factor)), 9, edgeColor, 2)

    print('\n%d total good matches, %d total bad matches. %.2f%% accuracy.\n' % \
        (np.sum(good_matches), x1_est.shape[0] - np.sum(good_matches), np.sum(good_matches) / x1_est.shape[0]*100))
    print('-------------------------------- End Evaluation --------------------------------\n')

    print('Saving visualization to ' + 'eval_'+file_name+'.png\n')
    cv2.imwrite('eval_'+file_name+'.png', newImg*255.0)

