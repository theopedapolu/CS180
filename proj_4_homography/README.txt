Name: Theophilus Pedapolu
SID: 3035826494
Email: theopedapolu@berkeley.edu

There are 2 code files in this submission:

homography.ipynb - jupyter notebook of all the code used to generate the images. The notebook is split up into sections, labeled
Part A and Part B for the different parts of Project 4. For Part A, the computeH() function takes in two sets of points and 
uses least squares to compute a 3x3 homography matrix, the warpImage() function takes in an image and a homography matrix and 
returns a warped image transformed by the homography, then image rectification is done on two images calling the earlier computeH and warpImage
functions, and finally the blend_mosaic() function take in a set of images and point correspondences and 
return a mosaic with the images manually stitched together. For Part B, the provided get_harris_corners() function returns the harris corners
and harris response for each interest point found, the ANMS() function takes in a harris response matrix and interest points coordinates
and returns the interest points found by adaptive non-maximal suppression, the extract_feature_patches() function takes in an image 
and a set of feature points and returns a list of feature descriptors (patches) for each points, the matching_outlier_rejection_function() 
takes in two sets of feature descriptors and returns two sets of matched points between the descriptors, rejecting points according to
Lowe thresholding, the RANSAC() function takes in two sets of matched points and implements RANSAC to return a robust homography matrix, 
the autoH() takes in two images and uses the previous functions to return a homography estimate from the first image to the second image, 
and finally the autoStitch() function takes in two images and a homography matrix and warps the images according to the homography and 
stitches them together to return an automatically stitched mosiac. 


point_select.py - simple python code that makes a figure for an image and allows you to select an arbitrary number of points on that image
and then prints the selected points in array form. I used this to output the points correspondences for manual stitching