
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch, cv2, os, csv
import supervision as sv
import numpy as np

# install correct package reqs for gpu use
#conda create -n sam python==3.8 ipykernel cudatoolkit=10.2
#conda activate sam
#conda install -y pytorch==1.9.1 torchvision==0.10.1 ; \
    
# OR

#conda install -y python==3.8 ipykernel cudatoolkit=10.2 pytorch==1.9.1 torchvision==0.10.1 ; \

#python3 -m pip install opencv-python-headless supervision Pillow  ; \
#python3 -m pip install /segment-anything/ ;


def window_img_prompt(image):
    if True:
        cv2.imshow('image',image)
        cv2.waitKey(0)
        
def get_a_and_c_values(mask):
    segmentation_binary = np.array(mask["segmentation"], dtype=np.uint8)
    contours, _ = cv2.findContours(segmentation_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aprox_contours = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        contour = cv2.approxPolyDP(cnt,0.01*peri,True)
        aprox_contours.append(contour)
    
    maxLength = 0
    maxIndex = -1
    for index, contour_point_list in enumerate(aprox_contours):
        if len(contour_point_list) > maxLength:
            maxLength = len(contour_point_list)
            maxIndex = index
    if maxLength == 0 or maxLength > 10: return False # more than 10 points in aproximated contour; bad
    (x_measurement,y_measurement),dist_c = cv2.minEnclosingCircle(aprox_contours[maxIndex])
    x_measurement = int(x_measurement)
    y_measurement = int(y_measurement)
    M = cv2.moments(aprox_contours[maxIndex])
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx = x_measurement
        cy = y_measurement
    dist_a = cv2.pointPolygonTest(aprox_contours[maxIndex],(cx,cy),True)
    dist_a = int(dist_a)
    dist_c = int(dist_c)
    if dist_a <= 0 or dist_c <= 0:
        return False
    ret_dict = {
        "dist_a":dist_a,
        "dist_c":dist_c,
        "center_x":x_measurement,
        "center_y":y_measurement,
        "aprox_contours":aprox_contours,
        "max_contour": aprox_contours[maxIndex],
        "center_x_cont": cx,
        "center_y_cont": cy,
    }
    return ret_dict
    
def annotate_image(image,masks, visualize_bbox=False):
    mask_annotator = sv.MaskAnnotator()
    if len(masks) == 0:
        print("You filtered all the masks >:(")
        return annotated_image
    detections = sv.Detections.from_sam(masks)
    annotated_image = mask_annotator.annotate(image, detections)
    if visualize_bbox:
        for mask in masks:
            x_val = int(mask["bbox"][0])
            y_val = int(mask["bbox"][1])
            if x_val + 5 >= annotated_image.shape[1] or y_val + 10 >= annotated_image.shape[0]:
                continue
            
            measurement_dict = get_a_and_c_values(mask)
            if not measurement_dict: continue
            
            dist_a = measurement_dict["dist_a"]
            dist_c = measurement_dict["dist_c"]
            center_x = measurement_dict["center_x"]
            center_y = measurement_dict["center_y"]
            aprox_contours = measurement_dict["aprox_contours"]
            cx = measurement_dict["center_x_cont"]
            cy = measurement_dict["center_y_cont"]
            
            annotated_image = cv2.drawContours(annotated_image, aprox_contours, -1, (0,255,0), 3)
            annotated_image = cv2.circle(annotated_image, (center_x,center_y), radius=3, color=(255, 0, 0), thickness=-1)
            annotated_image = cv2.circle(annotated_image, (center_x,center_y), radius=dist_c, color=(255,0,0), thickness=1)
            
            annotated_image = cv2.circle(annotated_image, (cx,cy), radius=3, color=(0, 255, 255), thickness=-1)
            annotated_image = cv2.circle(annotated_image, (cx,cy), radius=dist_a, color=(0,255,255), thickness=1)

            text = f"A:{dist_a*2}|C:{dist_c*2}|area:{mask['area']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            text_color = (0, 0, 255)
            font_thickness = 1
            x,y  = x_val + 5, y_val + 10
            annotated_image = cv2.putText(annotated_image, text,
                                (x,y), font, fontScale=font_scale, color=text_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    return annotated_image
    
device_cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device_cuda}")

#model_name = "sam_hq_vit_h.pth"
model_name = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=f"/home/zanz/ml_stuff/ml_playground/data/facebook_sam/sam_checkpoints/{model_name}")
sam.to(device=device_cuda)
torch.cuda.empty_cache()

root_image_folder = "/home/zanz/ml_stuff/ml_playground/data/facebook_sam/science_data/images/Glass, TPABr (2h, 5h, 20h)/A"
file_list = sorted(os.listdir(root_image_folder))

# defaults:
# pred_iou_thresh: float = 0.88,
# stability_score_thresh: float = 0.95,
# box_nms_thresh: float = 0.7,
# crop_n_layers: int = 0,
# crop_nms_thresh: float = 0.7,
# crop_overlap_ratio: float = 512 / 1500,

iou_tresh = 0.95 # 97 before # (0.88 to 0.97 likely good)
stability_score_tresh = 0.95 # (0.94 - 0.97 likely good)
box_nms_t = 0.7
crop_n_lay = 2 # 2 is more accurate than 1, 3 takes very long, do not go past 3
crop_nms_t = 0.7
crop_overlap_r = 512/1500

step_list = np.linspace(0,2,3)
print(step_list)
for step_val in step_list:
    crop_n_lay = int(step_val)
    print(f"Current step value: {crop_n_lay}")
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=iou_tresh,
        stability_score_thresh=stability_score_tresh,
        box_nms_thresh=box_nms_t,
        crop_n_layers=crop_n_lay,
        crop_nms_thresh=crop_nms_t,
        crop_overlap_ratio=crop_overlap_r
    )

    # set the image you want to see here
    image_range_start = 0 # 36
    image_range_end = len(file_list) # 37

    file_list.reverse()
    csv_output = []
    for filename in file_list[image_range_start:image_range_end]:
        image_path = f"{root_image_folder}/{filename}"
        image_uncropped = cv2.imread(image_path)
        img_width = 1024
        img_height = 1024
        print(f"shape before: {image_uncropped.shape}")
        if image_uncropped.shape[1] > img_height or image_uncropped.shape[2] > img_width:
            image_rgb = image_uncropped[240:1524, 300:1524]  # custom crop sizes
        else:
            image_rgb = image_uncropped[0:img_width, 0:img_height] 
        # idk why, but for some reason the image now becomes 1024 x 1024 
        #instead of the original 1088 x 1024, but its good since the extra pixels are the microscope UI at the bottom
        
        print(f"shape: {image_rgb.shape}, np amax:{np.amax(image_rgb)}, np amin:{np.amin(image_rgb)}")

        print(f"filename: {filename}")
        
        masks = mask_generator.generate(image_rgb)
        filter_count = 0
        
        #annotated_image = annotate_image(image_rgb,masks,visualize_bbox=True)
        #window_img_prompt(annotated_image)
        
        # Post processing below
        filtered_masks = []
        stats_area_list = []
        for mask in masks:
            bbox_width = mask["bbox"][2]
            bbox_height = mask["bbox"][3]
            area = mask["area"]
            segmentation_binary = mask["segmentation"]
            if bbox_height <= bbox_width:
                a_rect = bbox_height
                b_rect = bbox_width
            else:
                b_rect = bbox_height
                a_rect = bbox_width
            # We need to calculate the ratio of b/a to filter out detections where b is much too long
            #     |-------------|
            #   a |< (crystal) >| good crystal
            #     |-------------|
            #            b
            # (the longest of the two sides will always become b because of the above check)
            #     |----------------------------------------------|
            #   a |<                   (crystal)                >| bad crystal
            #     |----------------------------------------------|
            #                              b   
            if a_rect == 0:
                filter_count += 1
                continue
            else:
                b_ratio = b_rect / a_rect # good examples are 4/1, 3/1, 2/1, 1/1 
            
            if area < 1000 or b_ratio > 4: # remove all tic-tacs with an area smaller than 1000 or where the b side is 5x times bigger than the a side
                filter_count += 1
                continue
            
            
            measurement_dict = get_a_and_c_values(mask)
            if not measurement_dict: 
                filter_count += 1
                continue
            
            dist_a = measurement_dict["dist_a"]
            dist_c = measurement_dict["dist_c"]
            center_x = measurement_dict["center_x"]
            center_y = measurement_dict["center_y"]
            center_x_cont = measurement_dict["center_x_cont"]
            center_y_cont = measurement_dict["center_y_cont"]
            aprox_contours = measurement_dict["aprox_contours"]
            max_contour = measurement_dict["max_contour"]
            
            #if dist_a * 2 > dist_c or dist_a * 3 < dist_c:
            #    filter_count += 1
            #    continue
            
            #M = cv2.moments(max_contour)
            #if M['m00'] != 0:
            #    cx = int(M['m10']/M['m00'])
            #    cy = int(M['m01']/M['m00'])
                #print(f"Containing circle xy: {center_x}|{center_y} Contour xy: {cx}|{cy}")
            #    if not (center_x-8 < cx and center_x + 8 > cx and center_y-8 < cy and center_y + 8 > cy):
            #        filter_count += 1
            #        continue
            
            stats_area_list.append(mask["area"])
            filtered_masks.append(mask)
            
        #annotated_image = annotate_image(image_rgb,filtered_masks,visualize_bbox=True)
        #window_img_prompt(annotated_image)
            
        stats_area_array = np.array(stats_area_list)
        
        area_median = np.median(stats_area_array)
        area_mad = np.median(np.absolute(stats_area_array - np.median(stats_area_array)))
        
        area_mean = np.mean(stats_area_array)
        area_std = np.std(stats_area_list)
        
        write_file_name = f"outputs/{filename}_median_mad1_st{stability_score_tresh}_it{iou_tresh}_l{crop_n_lay}_over{crop_overlap_r:.2f}"
        stat_filtered_masks = []
        for mask in filtered_masks:
            area = mask["area"]
            if area < area_median-1*area_mad or area > area_median+1*area_mad:
                # if they dont fall within 1 standard deviation of the 
                # filtered mean, then we dont want them
                filter_count += 1
                continue
            
            measurement_dict = get_a_and_c_values(mask)
            if not measurement_dict: 
                filter_count += 1
                continue
            
            dist_a = measurement_dict["dist_a"]
            dist_c = measurement_dict["dist_c"]
            center_x = measurement_dict["center_x"]
            center_y = measurement_dict["center_y"]
            center_x_cont = measurement_dict["center_x_cont"]
            center_y_cont = measurement_dict["center_y_cont"]
            aprox_contours = measurement_dict["aprox_contours"]
            max_contour = measurement_dict["max_contour"]
            
            stat_filtered_masks.append(mask)
            csv_output.append([write_file_name,dist_a*2,dist_c*2,center_x,center_y,center_x_cont,center_y_cont,area])

        print(f"Filtered {filter_count} masks")
        annotated_image = annotate_image(image_rgb,stat_filtered_masks,visualize_bbox=True)
        #window_img_prompt(annotated_image)
            
        
        cv2.imwrite(f"{write_file_name}.jpg",annotated_image,[cv2.IMWRITE_JPEG_QUALITY, 100]) # have to be in "facebook_sam" folder

    with open(f'outputs/dataset_median_mad1_st{stability_score_tresh}_it{iou_tresh}_l{crop_n_lay}_over{crop_overlap_r:.2f}.csv', 'w', newline='\n') as csvfile:
        datawriter = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        datawriter.writerow(["filename","dist_a","dist_c","center_x","center_y","center_x_cont","center_y_cont","area"]) # header
        for row in csv_output:
            datawriter.writerow(row)