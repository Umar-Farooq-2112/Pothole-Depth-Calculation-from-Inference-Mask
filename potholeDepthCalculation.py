import numpy as np
import cv2

def fit_plane_to_depth_map(depth_map, bounding_box):
    x_min, y_min, w, h = bounding_box
    x_max, y_max = x_min+w,y_min+h
    
    x_vals = np.arange(x_min, x_max)
    y_vals = np.arange(y_min, y_max)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = depth_map[y_min:y_max, x_min:x_max]

    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    A = np.vstack((X_flat, Y_flat, np.ones_like(X_flat))).T
    B = Z_flat

    plane_coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    return plane_coeffs

def fit_plane_to_contour(depth_map, contour):
    contour_points = contour.reshape(-1, 2)  # Reshape to (num_points, 2)

    x_vals = contour_points[:, 0]
    y_vals = contour_points[:, 1]

    Z_vals = depth_map[y_vals, x_vals]

    A = np.vstack((x_vals, y_vals, np.ones_like(x_vals))).T
    B = Z_vals

    plane_coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    return plane_coeffs


def get_depth_at_coordinates(plane_coeffs, x_input, y_input):
    a, b, c = plane_coeffs
    z_input = a * x_input + b * y_input + c
    return z_input

def get_real_depth(pixelIntensity, depthParameter):
    try:
        pixelIntensity = float(pixelIntensity)
        depthParameter= float(depthParameter)

        depthh = depthParameter + 0.15 - ((255-pixelIntensity)/155) * 0.15 if pixelIntensity > 0 else None
        return depthh
    except Exception as e:
        print("An error occurred:", str(e))
        return None


def getMaxDepthRelativeBoundary(depthmap,cnt,depthParameter,show = [False]):
    
    x, y, w, h = cv2.boundingRect(cnt)
    
    pothole_cropped = depthmap[y:y+h,x:x+w]
    
    max_index = np.unravel_index(np.argmax(pothole_cropped), pothole_cropped.shape)
    max_depth = pothole_cropped[max_index]
    # plane_coeff = fit_plane_to_depth_map(depthmap,(x,y,w,h))
    plane_coeff = fit_plane_to_contour(depthmap,cnt)
    plane_depth = get_depth_at_coordinates(plane_coeff,max_index[0]+x,max_index[1]+y)
    
    print("Max Index: ", max_index)
    print("max Depth: ", max_depth)
    print("Plane Coeff: ", plane_coeff)
    print("Plane Depth: ", plane_depth)
    
    max_depth = get_real_depth(max_depth,depthParameter)
    plane_depth = get_real_depth(plane_depth,depthParameter)
    # print("Actual Depth: ",depth)
    if max_depth is None or plane_depth is None:
        return -1
    depth = abs(max_depth-plane_depth)
    
    # depth = max_depth-plane_depth
    # print("Actual Depth: ",depth)
    
    
    if show[0]:
        img = cv2.imread(show[1])
        inf_mask = cv2.imread(show[2])
        dmap = cv2.imread(show[3])
        
    
        cv2.rectangle(inf_mask,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(dmap,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.putText(img,f"{round(depth, 2)}",(x,y),5,5,(255,0,0),5)
        cv2.putText(inf_mask,f"{round(depth, 2)}",(x,y),5,5,(255,0,0),5)
        cv2.putText(dmap,f"{round(depth, 2)}",(x,y),5,5,(255,0,0),5)
        
        cv2.imwrite(show[1],img)
        cv2.imwrite(show[2],inf_mask)
        cv2.imwrite(show[3],dmap)
    
    return depth

# if __name__ == '__main__':
#     print("Starting............................")

#     image = '00000005.png'
#     basePath = "Dataset/"
#     mask = cv2.imread(basePath+ "stiched/masks/"+ image)
#     depthParameter =   np.load(basePath +'parameters/depthParameter.npy')
#     imageScale = np.load(basePath + 'parameters/imageScale.npy')/1000
#     depthmap = cv2.imread(basePath+"stiched/depthmaps/"+image)

#     if mask is not None and depthmap is not None and depthmap is not None and imageScale is not None:
#         print("All the Data loaded successfully.............")
#     print("Depth Parameter", depthParameter)
#     print("Image Scale", imageScale)

#     phMask = cv2.inRange(mask, np.array([0,250,250]),np.array([50,255,255]))

#     _,thresh_img = cv2.threshold(phMask, 150, 255, cv2.THRESH_BINARY)
#     phContours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     i = 0
#     debug = True
#     path = "dmap.png"
#     path2 = "mask.png"
    
#     cv2.imwrite(path,depthmap)
#     cv2.imwrite(path2,mask)
    
    
#     for cnt in phContours:
#         rect = cv2.boundingRect(cnt)
#         x,y,w,h = rect
#         if h*imageScale > 0.1 and w*imageScale > 0.1:     
#             depth = getMaxDepthRelativeBoundary(phMask,cv2.cvtColor(depthmap,cv2.COLOR_BGR2GRAY),cnt,depthParameter,[debug,path,i,path2])
#             print("Actual Depth: ", depth)
#         i+=1
#     print("Done................................")
                
             


if __name__ == '__main__':
    print("Starting............................")
    number_of_images = 41
    images = []
    for i in range(0,10):
        text = f"0000000{i}"
        images.append(text+".png")
    for i in range(10,number_of_images):
        text = f"000000{i}"
        images.append(text+".png")



    basePath = "Dataset/"
    depthParameter =   np.load(basePath +'parameters/depthParameter.npy')
    imageScale = np.load(basePath + 'parameters/imageScale.npy')/1000
    print("Depth Parameter", depthParameter)
    print("Image Scale", imageScale)

    for i in range(number_of_images):
        image = images[i]
        mask = cv2.imread(basePath+ "stiched/masks/"+ image)
        depthmap = cv2.imread(basePath+"stiched/depthmaps/"+image)
        img = cv2.imread(basePath+"stiched/images/"+image)

        if mask is not None and depthmap is not None and depthParameter is not None and imageScale is not None:
            print("All the Data loaded successfully.............")

        # phMask = cv2.inRange(mask, np.array([0,0,250]),np.array([0,0,255]))
        phMask = cv2.inRange(mask, np.array([0,250,250]),np.array([50,255,255]))

        _,thresh_img = cv2.threshold(phMask, 150, 255, cv2.THRESH_BINARY)
        phContours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        debug = True
        depthMapPath = basePath+"results/depthmap/"+image
        maskPath = basePath+"results/masks/"+image
        imagePath = basePath+"results/images/"+image
        
        cv2.imwrite(depthMapPath,depthmap)
        cv2.imwrite(maskPath,mask)
        cv2.imwrite(imagePath,img)
    
        
        for cnt in phContours:
            rect = cv2.boundingRect(cnt)
            x,y,w,h = rect
            if h*imageScale > 0.1 and w*imageScale > 0.1:     
                depth = getMaxDepthRelativeBoundary(cv2.cvtColor(depthmap,cv2.COLOR_BGR2GRAY),cnt,depthParameter,[debug,imagePath,maskPath,depthMapPath])
                # depth = getMaxDepthRelativeBoundaryOld(phMask,cv2.cvtColor(depthmap,cv2.COLOR_BGR2GRAY),cnt,depthParameter)
                print('Actual Depth',depth)
            i+=1
        print("Done................................")


