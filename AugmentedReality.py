import cv2
import numpy as np

# File paths
input_video_path = 'Multiple View.avi'
reference_frame_path = 'ReferenceFrame.png'
object_mask_path = 'ObjectMask.png'
augmented_layer_path = 'AugmentedLayer.png'
augmented_mask_path = 'AugmentedLayerMask.png'
output_video_path = 'Augmented Multiple View1.avi'

# Load images and masks
reference_frame = cv2.imread(reference_frame_path)
object_mask = cv2.imread(object_mask_path, 0)
augmented_layer = cv2.imread(augmented_layer_path)
augmented_mask = cv2.imread(augmented_mask_path, 0)


# Convert reference frame to grayscale for SIFT
reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

# Initialize video
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Initialize SIFT detector
sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01, edgeThreshold=10)
keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_gray, object_mask)

# Use BFMatcher with L2 norm for SIFT
bf = cv2.BFMatcher(cv2.NORM_L2)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale for SIFT
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

    if descriptors_frame is not None and len(descriptors_frame) >= 4:
        # KNN match and apply Lowe’s ratio test
        matches = bf.knnMatch(descriptors_ref, descriptors_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                # Warp the augmented layer onto the current frame
                warped_layer = cv2.warpPerspective(augmented_layer, H, (frame.shape[1], frame.shape[0]))
                warped_mask = cv2.warpPerspective(augmented_mask, H, (frame.shape[1], frame.shape[0]))

                mask_inv = cv2.bitwise_not(warped_mask)
                background = cv2.bitwise_and(frame, frame, mask=mask_inv)
                foreground = cv2.bitwise_and(warped_layer, warped_layer, mask=warped_mask)

                frame = cv2.add(background, foreground)
                

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
