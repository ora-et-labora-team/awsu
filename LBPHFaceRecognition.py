# Face Recognition with OpenCV

# ### Import Required Modules

# Before starting the actual coding we need to import the required modules for coding. So let's import them first.
#
# - **cv2:** is _OpenCV_ module for Python which we will use for face detection and face recognition.
# - **os:** We will use this Python module to read our training directories and file names.
# - **numpy:** We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from intersect import intersection
from DeepFaceRecognition import concatenate_images_in_grid

def LBPH_study():
    # Setup initial folders
    if not os.path.exists("images"):
        os.mkdir("images")

    if not os.path.exists("images/lbph"):
        os.mkdir("images/lbph")

    if not os.path.exists("images/graphs"):
        os.mkdir("images/graphs")

    impostors = 0
    genuines = 0
    total = 0
    for dir in os.listdir("./training-data"):
        total += 1

    # se un elemento è in probe ma non in training data, è un impostore
    for dir in os.listdir("./probe"):
        if dir not in os.listdir("./training-data"):
            impostors += 1
        else:
            genuines += 1

    print("GENUINES: " + str(genuines))
    print("IMPOSTORS: " + str(impostors))

    # total genuine in the probes
    global genuineProbes
    genuineProbes = 0
    # total genuine in the gallery
    global genuineGallery
    genuineGallery = 0
    global dataset_size
    dataset_size = 0

    # list of all the subjects in our dataset
    subjects = [""]
    for dir in os.listdir("./training-data"):
        if not dir.startswith("s"):
            continue
        subjects.append(str(dir))

    # calcolo dei genuine Probes
    for subfolder in os.listdir("./probe"):
        current_subfolder = os.path.join("./probe", subfolder)

        if os.path.isdir(current_subfolder) and subfolder in subjects:
            subfolder_size = len(os.listdir(current_subfolder))

            for index in range(subfolder_size):
                genuineProbes += 1

    # FACE DETECTION with OpenCv
    def detect_face(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("opencv-files/lbpcascade_frontalface.xml")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            return None, None

        (x, y, w, h) = faces[0]
        return gray[y : y + w, x : x + h], faces[0]

    # this function will return 2 lists of the same size, one list of faces and another
    # list of labels for each face.
    # In this step we will also divide the dataset in impostor and genuine and in probes and gallery.
    # All impostors are only in probes while the 30% of the images of every genuine is in the probes and the remaining
    # 70% of the images for each genuine will be in the gallery.

    def prepare_training_data(data_folder_path):
        # ------STEP-1--------
        # get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)
        # list to hold all subject faces
        faces = []
        # list to hold labels for all subjects
        labels = []

        # let's go through each directory and read images within it
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue

            label = int(dir_name.replace("s", ""))
            print(label)
            # sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name

            # get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)

            # ------STEP-3--------
            # go through each image name, read image,
            # detect face and add face to list of faces
            for image_name in subject_images_names:
                globals()["dataset_size"] += 1
                # ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue

                # build image path
                # sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                # read image
                image = cv2.imread(image_path)

                # Converti l'immagine in scala di grigi
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Ridimensiona l'immagine alle dimensioni desiderate
                resized_image = cv2.resize(gray_image, (100, 100))

                print(f"Reading image: {image_path}, Size: {resized_image.shape}")

                # display an image window to show the image
                # cv2.imshow("My Gallery...", cv2.resize(resized_image, (400, 500)))
                # cv2.waitKey(100)

                # detect face
                face, rect = detect_face(image)

                # ------STEP-4--------
                # for the purpose of this tutorial
                # we will ignore faces that are not detected
                if face is not None:
                    print(f"Face detected for {dir_name} in {image_path}")
                else:
                    print(f"No face detected for {dir_name} in {image_path}")
                if face is not None:
                    # add face to list of faces
                    faces.append(face)
                    # add label for this face
                    labels.append(label)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        return faces, labels

    # The following function takes the path, where training subjects' folders are stored, as parameter, and follows the same four prepare data substeps mentioned above.
    #
    # 1) On "line 8" the method `os.listdir` is used to read names of all folders stored on path passed to function as parameter, and "line 10-13" labels and faces vectors are defined.
    #
    # 2) After traversing through all subjects' folder names and from each subject's folder name on "line 27" the label information is extracted.
    #
    # 3) On "line 34", all the images names of the current subject being traversed are read and analysed
    #
    # 4) On "line 62-66", the detected face and label are added to their respective vectors.

    # In[5]:

    # let's first prepare our training data
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    # create the LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))

    # function to draw rectangle on image
    # according to given (x, y) coordinates and
    # given width and heigh
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # function to draw text on give image starting from
    # passed (x, y) coordinates.
    def draw_text(img, text, x, y):
        cv2.putText(
            img, text, (x - 1, y - 1), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2
        )

    # Now that we have the drawing functions, we just need to call the face recognizer's `predict(face)` method to test our face recognizer on test images.

    # this function recognizes the person in image passed
    # and draws a rectangle around detected face with name of the subject

    def predict(test_img):
        # make a copy of the image as we don't want to change the original image
        img = test_img.copy()

        face, rect = detect_face(img)
        if face is None:
            return (None, None, None)

        # QUESTA PREDICT è QUELLA DEL FACE RECOGNIZER
        # Un valore basso di confidence indica maggiore sicurezza dell'affidabilità
        # mentre un valore alto indica maggiore incertezza --> sarebbe la similarità
        label, confidence = face_recognizer.predict(
            face
        )  # restituisce una label che sta nel training data
        # get name of respective label returned by face recognizer
        label_text = "s" + str(label)
        # la dir la devo prendere in base alla label estratta
        dir = label_text
        result = label_text
        # print("FORMATO LABEL TEXT", label_text)
        if label_text in subjects:
            # if (label_text!="IMPOSTOR"):
            subject_dir_path = "training-data/" + dir
            images_names = os.listdir(subject_dir_path)
            subject_name = images_names[0]  # mi aspetto  Richard_Lagos_002.jpg
            parts = subject_name.split("_")
            if len(parts) >= 2:
                result = "_".join(parts[:-1])
                # print(result)
            else:
                print("La stringa non contiene un formato atteso")
        # print("LA LABEL è QUESTA", label, "MENTRE LA LABEL TEXT è:", label_text)
        # draw a rectangle around face detected
        draw_rectangle(img, rect)
        # draw name of predicted person
        draw_text(img, result, rect[0], rect[1] - 5)

        return (label_text, img, confidence)

    def calculate_metrics(t, test_img, predicted_img, subjects, DI, FR, FA, GR):
        (label_pred, predicted, confidence) = predict(test_img)
        if predicted is not None:
            # print("CURRENT SUBFOLDER", subfolder, "AND LABEL_PRED", label_pred)
            if confidence <= t:
                # se indovina chi è --> TRUE POSITIVE
                # se la label predetta è = a quella che sto analizzando (current_subfolder senza la s)
                if label_pred == subfolder:
                    predicted_img.append((label_pred, predicted))
                    DI += 1
                # se lo associa ad un impostore
                else:
                    if (
                        subfolder not in subjects
                    ):  # se sto analizzando una probe degli impostori
                        FA += 1
                    # else:  # se sto analizzando una probe dei genuini e non me la vede con la label giusta
                    #     FR += 1
            else:
                # se la probe e la presumed identity (identity[0]) coincidono --> FALSE REJECTION
                if label_pred != subfolder:
                    GR += 1
                # se le due invece erano diverse "Ai_Sugiyama" e "Leo_DiCaprio" --> GENUINE REJECTION
                # else:
                #     GR += 1
        else:
            GR += 1

        return (DI, FR, FA, GR)

    # Now that we have the prediction function well defined, next step is to actually call this function on our test images and display those test images to see if our face recognizer correctly recognized them. So let's do it. This is what we have been waiting for.

    dir = []
    far = []
    roc = []
    threshold_array = []

    print("Predicting images...")
    input_threshold = 90
    current_threshold = 60
    global input_point
    input_point = (0, 0)

    print("ALL SUBJECT IN TRAINING-DATA", subjects)

    while current_threshold <= 140:
        # changing threshold
        print("CURRENT THRESHOLD: " + str(current_threshold))
        # face_recognizer.setThreshold(current_threshold)

        # load test images
        predicted_img = []
        DI = 0  # true positive
        FA = 0  # false positive
        FR = 0  # false negative
        GR = 0  # true negative

        # nell'array metto tutti gli indici dei genuini del dataset
        # index_genuine = []
        # for index in range(1,len(subjects)):
        #     if subjects[index]!="IMPOSTOR":
        #         index_genuine.append(index)

        totalProbes = 0  # tutte le probes che ho sia
        for subfolder in os.listdir("./probe"):
            current_subfolder = "./probe/" + subfolder  # sarebbero gli s1,s2..
            subfolder_size = len(os.listdir(current_subfolder))
            # considera solo i primi 30% che fanno parte dei probes di ogni soggetto sia genuino
            # che impostore perchè sto nei probes
            for index in range(0, subfolder_size):
                totalProbes += 1
                path = os.listdir(current_subfolder)[index]
                test_img = cv2.imread(current_subfolder + "/" + str(path))
                # perform a prediction
                (DI, FR, FA, GR) = calculate_metrics(
                    current_threshold, test_img, predicted_img, subjects, DI, FR, FA, GR
                )

        # compute impostor probes for testing (total-genuine)
        impostorProbes = totalProbes - genuineProbes

        # check performance
        print("PROBES TOTALI", totalProbes)
        print("GENUINE PROBES", genuineProbes)
        print("IMPOSTOR PROBES", impostorProbes)
        print("TRUE POSITIVE", DI)
        print("FALSE ACCEPTANCE", FA)
        print("FALSE REJECTION", FR)
        print("GENUINE REJECTION", GR)
        DIR = DI / genuineProbes  # Detection and Identification Rate, at rank 1
        FAR = FA / impostorProbes  # False Acceptance Rate = #FA / impostors
        FRR = 1 - DIR  # False Rejection Rate = 1-DIR(t,1) with threshold t
        # EER = 0  # Equal Error Rate = point where FRR==FAR
        dir.append(DIR)
        far.append(FAR)
        threshold_array.append(current_threshold)
        roc.append((FAR, DIR))

        if current_threshold == input_threshold:
            # STAMPA SOLO I SOGGETTI ACCETTATI (compresi quelli con wrong identity)
            for label_pred, predicted in predicted_img:
                cv2.imwrite("images/lbph/" + label_pred + ".jpg", predicted)
                # cv2.imshow(label_pred, cv2.resize(predicted, (400, 500)))
                

            globals()["input_point"] = (FAR, DIR)
            print(
                "THRESHOLD "
                + str(input_threshold)
                + " --> FAR = "
                + str(FAR)
                + ", FRR = "
                + str(FRR)
                + ", and DIR = "
                + str(DIR)
            )
        # increment threshold by default
        current_threshold += 5

    # Plotting the ROC first and then the FAR-FRR curves (EER)

    print(dir)
    print(far)
    #########first figure
    x = np.array(far)
    y = np.array(dir)
    plt.xlabel("FAR")
    plt.ylabel("DIR")
    plt.plot(x, y)
    # plt.axline((0, 0), slope=1, linestyle="--")
    # save into a png file
    plt.savefig("images/graphs" + "/ROC_lbph.png")
    plt.show()

    

    # first EER
    # dx = np.array([0,1])
    # dy = np.array([1,0])
    # x1, y1 = intersection(x, y, dx, dy)
    # plt.plot(x1, y1, "ro")
    # print("ERR raw curve: found for FAR = "+str(x1[0])+" and DIR = "+str(y1[0])+" (so FRR = "+str(float(1)-float(y1[0]))+")")

    # let's sanitize the two arrays
    # newX = []
    # newY = []
    # e = 0
    # while e < len(x):
    #     if x[e] not in newX:
    #         newX.append(x[e])
    #         newY.append(y[e])
    #     e += 1
    # xPlot = np.array(newX)
    # yPlot = np.array(newY)
    # # let's use the monotone cubic spline
    # X_Y_Spline = PchipInterpolator(xPlot, yPlot)
    # X_ = np.linspace(xPlot.min(), xPlot.max(), 500)
    # Y_ = X_Y_Spline(X_)
    # plt.plot(X_, Y_, color="green")

    # second EER
    # x2, y2 = intersection(X_, Y_, dx, dy)
    # plt.plot(x2, y2, "ko")
    # print("ERR smooth curve: found for FAR = "+str(x2[0])+" and DIR = "+str(y2[0])+" (so FRR = "+str(float(1)-float(y2[0]))+")")
    # plt.plot(input_point[0],input_point[1], "mo")

    # # draw first figure
    # plt.title("Plot ROC Curve (marking EER point)")
    # plt.xlabel("FAR")
    # plt.ylabel("DIR")
    # plt.show()

    #########second figure
    # plot FAR with threshold
    xPlot = np.array(threshold_array)
    yPlot = np.array(far)
    # let's use the monotone cubic spline
    X_Y_Spline = PchipInterpolator(xPlot, yPlot)
    X_ = np.linspace(xPlot.min(), xPlot.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color="green")

    # plot FRR with threshold
    frr_array = []
    for i in range(0, len(dir)):
        frr_array.append(float(1) - float(dir[i]))
    xPlot = np.array(threshold_array)
    yPlot = np.array(frr_array)
    # let's use the monotone cubic spline
    X_Y_Spline = PchipInterpolator(xPlot, yPlot)
    X1_ = np.linspace(xPlot.min(), xPlot.max(), 500)
    Y1_ = X_Y_Spline(X1_)
    plt.plot(X1_, Y1_, color="blue")

    # third EER
    x3, y3 = intersection(X1_, Y1_, X_, Y_)
    plt.plot(x3, y3, "ro")
    print("Final ERR: found for THRESHOLD t = "+str(x3[0])+" and FRR(t) = FAR(t) = "+str(y3[0]))
    plt.axvline(x3[0], color = 'r', linestyle = "--")

    # draw second figure
    plt.title("Plot FAR and FRR intersection in the EER")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.savefig("images/graphs" + "/EER_lbph.png")
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return

