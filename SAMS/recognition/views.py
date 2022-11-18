from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from SAMS.settings import BASE_DIR
import dlib
import cv2
import pickle
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import datetime
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.manifold import TSNE
from users.models import Present, Time



#utilities
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True
    
    return False

def create_dataset(username):
    id = username
    if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory='face_recognition_data/training_dataset/{}/'.format(id)

    # Detect face
    #Loading the HOG face detector and the shape predictpr for allignment

    print("[INFO] Loading facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
    fa = FaceAligner(predictor , desiredFaceWidth = 96)
    #capture images from the webcam and process and detect the face
    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=0).start()
    #time.sleep(2.0) ####CHECK######

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        #vs.read each frame
        frame = vs.read()
        #Resize each image
        frame = imutils.resize(frame ,width = 800)
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = detector(gray_frame,0)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        

        for face in faces:
            print("inside for loop")
            (x,y,w,h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame,gray_frame,face)
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            
            if face is None:
                print("face is none")
                continue


            cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
            face_aligned = imutils.resize(face_aligned ,width = 400)
            #cv2.imshow("Image Captured",face_aligned)
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(50)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Add Images",frame)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>300):
            break
    
    #Stoping the videostream
    vs.stop()
    # destroying all the windows
    cv2.destroyAllWindows()

def predict(face_aligned,svc,threshold=0.7):
    face_encodings=np.zeros((1,128))
    try:
        x_face_locations=face_recognition.face_locations(face_aligned)
        faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
        if(len(faces_encodings)==0):
            return ([-1],[0])

    except:

        return ([-1],[0])

    prob=svc.predict_proba(faces_encodings)
    result=np.where(prob[0]==np.amax(prob[0]))
    if(prob[0][result[0]]<=threshold):
        return ([-1],prob[0][result[0]])

    return (result[0],prob[0][result[0]])

def vizualize_Data(embedded, targets,):
    
    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1));
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()	
    plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
    plt.close()

def update_attendance_in_db_in(present):
    today=datetime.date.today()
    time=datetime.datetime.now()
    for person in present:
        user=User.objects.get(username=person)
        try:
           qs=Present.objects.get(user=user,date=today)
        except :
            qs= None
        
        if qs is None:
            if present[person]==True:
                        a=Present(user=user,date=today,present=True)
                        a.save()
            else:
                a=Present(user=user,date=today,present=False)
                a.save()
        else:
            if present[person]==True:
                qs.present=True
                qs.save(update_fields=['present'])
        if present[person]==True:
            a=Time(user=user,date=today,time=time, out=False)
            a.save()




# Create your views here.
def login(request):

    return render(request, 'recognition/home.html')

@login_required
def dashboard(request):
    if(request.user.username=='admin'):
        print("admin")
        return render(request, 'recognition/adminDashboard.html')
    else:
        print("not admin")
        return render(request,'recognition/attendance.html')

@login_required
def not_authorized(request):

    return render(request,'recognition/notAuthorized.html')

def attendanceIn(request):
    
    detector = dlib.get_frontal_face_detector()
    
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
    svc_save_path="face_recognition_data/svc.sav"	

      
    with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
    fa = FaceAligner(predictor , desiredFaceWidth = 96)
    encoder=LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')


    faces_encodings = np.zeros((1,128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()
    
    sampleNum = 0
    
    while(True):
        
        frame = vs.read()
        
        frame = imutils.resize(frame ,width = 800)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray_frame,0)
        
        


        for face in faces:
            print("INFO : inside for loop")
            (x,y,w,h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame,gray_frame,face)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                    
            (pred,prob)=predict(face_aligned,svc)

            
            if(pred!=[-1]):
                
                person_name=encoder.inverse_transform(np.ravel([pred]))[0]
                pred=person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred,0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                     count[pred] = 0
                else:
                #if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred,0) + 1
                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            else:
                person_name="unknown"
                cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            
            
            #cv2.putText()
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            #cv2.waitKey(50)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Mark Attendance - In - Press q to exit",frame)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        #cv2.waitKey(1)
        #To get out of the loop
        key=cv2.waitKey(50) & 0xFF
        if(key==ord("q")):
            break
    
    #Stoping the videostream
    vs.stop()

    # destroying all the windows
    cv2.destroyAllWindows()
    update_attendance_in_db_in(present)
    return redirect('home')



def attendanceOut(request):
    
    detector = dlib.get_frontal_face_detector()
    
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
    svc_save_path="face_recognition_data/svc.sav"	

            
    with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
    fa = FaceAligner(predictor , desiredFaceWidth = 96)
    encoder=LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')


    faces_encodings = np.zeros((1,128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = dict()
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    

    vs = VideoStream(src=0).start()
    
    sampleNum = 0
    
    while(True):
        
        frame = vs.read()
        
        frame = imutils.resize(frame ,width = 800)
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray_frame,0)
        
        


        for face in faces:
            print("INFO : inside for loop")
            (x,y,w,h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame,gray_frame,face)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                    
            
            (pred,prob)=predict(face_aligned,svc)
            

            
            if(pred!=[-1]):
                
                person_name=encoder.inverse_transform(np.ravel([pred]))[0]
                pred=person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred,0) + 1

                if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
                     count[pred] = 0
                else:
                #if count[pred] == 4 and (time.time()-start) <= 1.5:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred,0) + 1
                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            else:
                person_name="unknown"
                cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            
            
            #cv2.putText()
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            #cv2.waitKey(50)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        #cv2.waitKey(1)
        #To get out of the loop
        key=cv2.waitKey(50) & 0xFF
        if(key==ord("q")):
            break
    
    #Stoping the videostream
    vs.stop()

    # destroying all the windows
    cv2.destroyAllWindows()
    update_attendance_in_db_out(present)
    return redirect('home')
