import cv2
from util import get_parking_spots_bboxes, empty_or_not
mask_path =  '.\masks\mask_crop.png'
video_path = '.\sample_parking_lots\parking_crop.mp4'

captured = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)

connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    
parking_spots = get_parking_spots_bboxes(connected_components)

status_f_spots = [None for spot in parking_spots]
step = 30
frame_no = 0
ret= True
while ret:
    ret, frame = captured.read()

    if frame_no % step == 0:
        for p_index, place in enumerate(parking_spots):

            x1, y1, w ,h =place

            cropped_spot = frame[y1:y1 + h, x1:x1 + w, :]

            status = empty_or_not(cropped_spot)
            status_f_spots[p_index] = status
    
    for p_index, place in enumerate(parking_spots):
        status = status_f_spots[p_index]
        x1, y1, w ,h = parking_spots[p_index]
        if status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1+ h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_no += 1

captured.release()

cv2.destroyAllWindows()