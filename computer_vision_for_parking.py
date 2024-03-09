import cv2
from util import get_parking_spots_bboxes, empty_or_not
mask_path =  '.\masks\mask_crop.png'
video_path = '.\sample_parking_lots\parking_crop.mp4'

captured = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)

connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
    
parking_spots = get_parking_spots_bboxes(connected_components)

print(parking_spots[0])

ret= True
while ret:
    ret, frame = captured.read()
    for place in parking_spots:

        x1, y1, w ,h =place

        cropped_spot = frame[y1:y1 + h, x1:x1 + w, :]

        status = empty_or_not(cropped_spot)

        if status:
            frame = cv2.rectagle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (x1,y1), (x1 + w, y1+ h), (255, 0, 0), 2)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

captured.release()

cv2.destroyAllWindows()