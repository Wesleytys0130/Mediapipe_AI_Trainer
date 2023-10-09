import cv2
import mediapipe as mp
import numpy as np

#計算關節角度
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle

#計算身體前傾角度
def body_tilt_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    if c == 1:
        horizontal_distance = b[0] - a[0]
        body_height = a[1] - b[1]
    elif c==0:
        horizontal_distance = a[0] - b[0]
        body_height = a[1] - b[1]
    # if horizontal_distance > 0:
    angle = np.degrees(np.arctan(body_height / horizontal_distance))
    # else:
    #     angle = 0 
    return angle

# horizontal_distance = abs(shoulder[0] - hip[0])
# body_height = abs(shoulder[1] - hip[1])
# body_angle = body_tilt_angle(body_height,horizontal_distance)



#計算肢段長度
def body_length(a,b):
    a = np.array(a)
    b = np.array(b)
    length = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return length



#打開mp的pose功能
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

 

#全域變數
counter = 0 
stage = "start"
video_width = 1080
video_index = 6862
#左邊是 0 右邊是 1
video_direction = 0


#讀影片檔
cap = cv2.VideoCapture(f'../data/IMG_{video_index}.MOV')
#開啟電腦攝影機
# cap = cv2.VideoCapture(1) 


## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #計數方塊
        cv2.rectangle(image, (0,0), (1080,210), (245,100,20), -1)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[11+video_direction].x,landmarks[11+video_direction].y]
            hip = [landmarks[23+video_direction].x,landmarks[23+video_direction].y]
            knee = [landmarks[25+video_direction].x,landmarks[25+video_direction].y]
            ankle = [landmarks[27+video_direction].x,landmarks[27+video_direction].y]
            foot_index = [landmarks[31+video_direction].x,landmarks[31+video_direction].y]
            # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            
            hip_angle = calculate_angle(shoulder, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            ankle_angle = calculate_angle(knee, ankle, foot_index)
            # print(f'hip_angle:{int(hip_angle)}\nknee_angle:{int(knee_angle)}\nankle_angle:{int(ankle_angle)}')
            
            
            
            cv2.putText(image, str(int(hip_angle)), 
                        tuple(np.multiply(hip, [video_width, 1920]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
            cv2.putText(image, str(int(knee_angle)), 
                        tuple(np.multiply(knee, [video_width, 1920]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
            cv2.putText(image, str(int(ankle_angle)), 
                        tuple(np.multiply(ankle, [video_width, 1920]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                            )
            
            body_angle = body_tilt_angle(shoulder,hip,video_direction)
            print(body_angle)

            cv2.putText(image, f"body tilt angle:{str(int(body_angle))}",
                    (575,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # cv2.putText(image, f"body tilt angle",
            #         (470,200), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            # cv2.putText(image, str(str(int(body_angle))),
            #         (485,250), 
            #         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            

            # 計算軀幹長度
            torso_length = body_length(hip, shoulder)
            thigh_length = body_length(knee, hip)
            calf_length = body_length(ankle, knee)
            # print(f"{torso_length*10:.2f},{thigh_length*10:.2f},{calf_length*10:.2f}")
            
            # torso_to_thigh_ratio = torso_length / thigh_length
            # thigh_to_calf_ratio = thigh_length / calf_length
            # torso_to_lag_ratio = torso_length / (thigh_length+calf_length)

            cv2.putText(image, f"torso : thigh : calf",
                    (570,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(f"{torso_length*10:.1f}  :  {thigh_length*10:.1f}  : {calf_length*10:.1f}"),
                    (590,110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            
            if hip_angle < 70:
                stage = "down"
            if hip_angle > 110 and stage =='down':
                stage="up"
                counter +=1
                # print(counter)

        except:
            pass
        
        

        # Rep data
        cv2.putText(image, 'REPS', (30,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (50,190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'STAGE', (250,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (250,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2, cv2.LINE_AA)

        # Render detections
        #第一個畫布參數為關節點、第二個為連接線
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        #負數為暫停偵
        # if cv2.waitKey(-1) & 0xFF == ord('q'):  
        #     break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        

    cap.release()
    cv2.destroyAllWindows()