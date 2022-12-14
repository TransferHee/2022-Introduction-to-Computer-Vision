# 2022-Introduction-to-Computer-Vision

## Score

    Assign1 - 100/100 (Average score: 85)
    Assign2 - 100/100 (Average score: 71)
    Assign3 - 100/100 (Average score: 78.6)
    Assign4 - 86/100  (Average score: 59.9)
	
    Grade: A+

## Assign 1

### A1_image_filtering

Perform Gaussian filtering to an image

![part_1_gaussian_filtered_lenna](https://user-images.githubusercontent.com/62924398/207543731-5cf15434-3a32-47e7-88d0-497a51a1ece7.png)

### A1_edge_detection

Perform edge detection without simplified NMS

![part_2_edge_raw_lenna](https://user-images.githubusercontent.com/62924398/207543944-51007721-0d5c-45ae-a862-d24b6950f46e.png)

With simplified NMS

![part_2_edge_sup_lenna](https://user-images.githubusercontent.com/62924398/207544101-a633517c-f4a6-4bc8-a0f8-f4d58f9ac71b.png)

### A1_corner_detection

Perform corner detection without NMS

![part_3_corner_bin_lenna](https://user-images.githubusercontent.com/62924398/207544186-729f77f2-1055-404f-b953-7f5a08b40464.png)

With winSize NMS

![part_3_corner_sup_lenna](https://user-images.githubusercontent.com/62924398/207544265-80651869-e705-4d15-844f-127abb02587e.png)

## Assign 2

### A2_2d_transformations

Keys

![keys](https://user-images.githubusercontent.com/62924398/207544703-62648e70-a063-40dc-9811-6fe970c977b7.png)

Visualization of a transformed image on a 2D plane.

![re](https://user-images.githubusercontent.com/62924398/207544841-b00c85a2-f3a5-43bf-a6db-8e98b1bd2a22.png)

### A2_homography

Feature matching with ORB descriptor

![fe](https://user-images.githubusercontent.com/62924398/207546932-e4044d3e-af49-4b51-8d2b-c7167b27e3f5.png)

Computing homography and warping

![homo](https://user-images.githubusercontent.com/62924398/207547022-c6d74891-57ec-4c95-bc3a-0d754eec5a86.png)

![homo_wraping](https://user-images.githubusercontent.com/62924398/207547099-b9c02c77-b7ab-4720-b9fd-8c39bda0e243.png)

Homography with RANSAC

![ransac_wraping](https://user-images.githubusercontent.com/62924398/207547291-3d93e05b-5cbb-4ef7-96bf-996d7c031076.png)

Harry Potter wraping

![hp_wraping](https://user-images.githubusercontent.com/62924398/207547112-081eea20-3ae7-425c-ae09-203e2f734d4d.png)

Image stitching

![stitching](https://user-images.githubusercontent.com/62924398/207548694-8a0b2110-c26d-40f4-9efd-3795fd0dee9a.png)

Image stitching and blending

![sandb](https://user-images.githubusercontent.com/62924398/207548823-e36db9e2-4df2-4a87-9a9e-245f4bf32e4c.png)

## Assign 3

### A3_Fmat

    Average Reprojection Errors (temple1.png and temple2.png)
    
    Raw = 0.8050090158803994
    Norm = 0.3595533893873907
    Mine = 0.351722375664347
    OpenCV 8POINT = 0.35920028823193695

Draw Epipolar lines

![ep](https://user-images.githubusercontent.com/62924398/207552853-3427c187-f86a-48a9-a821-36019ebc9395.png)

## Assign 4

### A4_compute_descriptors

    Use Kmeans clustering & VLAD
    
    L1: 3.32200 / L2: 3.3210
    Your Accuracy = 3.322000
    
    
