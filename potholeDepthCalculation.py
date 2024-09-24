import numpy as np
import cv2


# def getMaxDepthRelativeBoundary(phMask,depthmap,cnt,depthParameter):
    
#     mask = np.zeros_like(phMask)
#     cv2.drawContours(mask, [cnt], 0, (255), -1)
#     masked_image = cv2.bitwise_and(depthmap, depthmap, mask=mask)

#     contourdraw = np.zeros_like(phMask)
#     cv2.drawContours(contourdraw, [cnt], -1, (255,255,255), -1)
#     maskdilated = cv2.dilate(contourdraw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(55,55)), iterations=2) 
#     maskpotholeboundary = cv2.bitwise_and( cv2.bitwise_not(contourdraw), maskdilated)
#     maskedImageBoundary = cv2.bitwise_and(depthmap, depthmap, mask=maskpotholeboundary)
#     maskedImageBoundaryIntArray = depthmap[maskedImageBoundary>0]

#     max_pixel_value = np.max(masked_image)
#     max_pixel_value_boundary = np.mean(maskedImageBoundaryIntArray)
#     # logIt('max_pixel_value', max_pixel_value, 'max_pixel_value_boundary', max_pixel_value_boundary)
#     relativeDepthValue = max_pixel_value - max_pixel_value_boundary 
#     # logIt('relativeDepthValue', relativeDepthValue, 'max_pixel_value_boundary', max_pixel_value_boundary, 'max_pixel_value', max_pixel_value)
    
#     if max_pixel_value==0 or relativeDepthValue==0:
#         return -1
#     else:
#         positiveDepthRange = 0.15
#         bMean = 100                             # changed from 155 to 100 to match value in generateDepthMap in orthonormal part
#         bU = 255            
#         # z8bit = bU - ((bU-bMean)/positiveDepthRng) * (d1scaled + positiveDepthRng - ptZnScaled)
#         # depth = ((positiveDepthRange*max_pixel_value) - (positiveDepthRange*bU) + (bMean*(depthParameter + positiveDepthRange))) / bMean
#         # depthRelative = ((positiveDepthRange*relativeDepthValue) - (positiveDepthRange*bU) + (bMean*(depthParameter + positiveDepthRange))) / bMean


#         # ------------------------- changed on 5 Sept 2024 by Dr Mathavan ------------------------------
#         depthRelative2 = relativeDepthValue*positiveDepthRange/(bU-bMean)
#         depthRelativeTest = depthParameter + positiveDepthRange + ((max_pixel_value-255)/(bU - bMean)) * positiveDepthRange
#         # depthRelative3 = depthParameter + positiveDepthRange + (( ((-1*relativeDepthValue)-255)/(bU - bMean)) * positiveDepthRange)

#         # logIt('depthRelative2 meters', depthRelative2, 'depth intensity check', depthRelativeTest)

#         return depthRelative2

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

def get_depth_at_coordinates(plane_coeffs, x_input, y_input):
    a, b, c = plane_coeffs
    z_input = a * x_input + b * y_input + c
    return z_input

def getMaxDepthRelativeBoundary(phMask,depthmap,cnt,depthParameter,show = [False]):
    
    x, y, w, h = cv2.boundingRect(cnt)
    
    pothole_cropped = depthmap[y:y+h,x:x+w]
    
    max_index = np.unravel_index(np.argmax(pothole_cropped), pothole_cropped.shape)
    max_depth = pothole_cropped[max_index]
    plane_coeff = fit_plane_to_depth_map(depthmap,(x,y,w,h))
    plane_depth = get_depth_at_coordinates(plane_coeff,max_index[0]+x,max_index[1]+y)
    
    print("Max Index: ", max_index)
    print("max Depth: ", max_depth)
    print("Plane Coeff: ", plane_coeff)
    print("Plane Depth: ", plane_depth)
    
    depth = depthParameter*(max_depth-plane_depth)
    # depth = max_depth-plane_depth
    print("Actual Depth: ",depth)
    
    
    if show[0]:
        dmap = cv2.imread(show[1])
        inf_mask = cv2.imread(show[3])
    
        cv2.rectangle(inf_mask,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(dmap,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.putText(inf_mask,f"{show[2]}, {round(depth, 2)}",(x,y),5,5,(255,0,0),5)
        cv2.putText(dmap,f"{show[2]}, {round(depth, 2)}",(x,y),5,5,(255,0,0),5)
        
        cv2.imwrite(show[3],inf_mask)
        cv2.imwrite(show[1],dmap)
    
    return depth

if __name__ == '__main__':
    print("Starting............................")

    image = '00000005.png'
    basePath = "Dataset/"
    mask = cv2.imread(basePath+ "stiched/masks/"+ image)
    depthParameter =   np.load(basePath +'parameters/depthParameter.npy')
    imageScale = np.load(basePath + 'parameters/imageScale.npy')/1000
    depthmap = cv2.imread(basePath+"stiched/depthmaps/"+image)

    if mask is not None and depthmap is not None and depthmap is not None and imageScale is not None:
        print("All the Data loaded successfully.............")
    print("Depth Parameter", depthParameter)
    print("Image Scale", imageScale)

    phMask = cv2.inRange(mask, np.array([0,250,250]),np.array([50,255,255]))

    _,thresh_img = cv2.threshold(phMask, 150, 255, cv2.THRESH_BINARY)
    phContours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    debug = True
    path = "dmap.png"
    path2 = "mask.png"
    
    cv2.imwrite(path,depthmap)
    cv2.imwrite(path2,mask)
    
    
    for cnt in phContours:
        rect = cv2.boundingRect(cnt)
        x,y,w,h = rect
        if h*imageScale > 0.1 and w*imageScale > 0.1:     
            depth = getMaxDepthRelativeBoundary(phMask,cv2.cvtColor(depthmap,cv2.COLOR_BGR2GRAY),cnt,depthParameter,[debug,path,i,path2])
        i+=1
    print("Done................................")
                
             
    