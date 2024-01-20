import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from intersect import intersection
from scipy.interpolate import PchipInterpolator

# Specify the paths to the folders
db_path = "training-data"
probe_path = "probe"
img_path = "images/deepface"



def concatenate_images_in_grid(folder_path, output_path, grid_cols=3):
    images = []

    # if output_path exists, delete it
    if os.path.exists(output_path):
        os.remove(output_path)

    # Load all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            images.append(img)

    # Check if there are any images
    if not images:
        print("No images found in the specified folder.")
        return

    # Calculate the number of rows required in the grid
    grid_rows = math.ceil(len(images) / grid_cols)

    # Calculate the size of each cell in the grid
    cell_height = images[0].shape[0]
    cell_width = images[0].shape[1]

    # Resize all images to match the dimensions of each cell
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (cell_width, cell_height))

    # Create an empty grid
    grid = 255 * np.ones((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)

    # Populate the grid with images
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        grid[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width] = img

    # Save the grid image
    cv2.imwrite(output_path, grid)


def face_recogn(face, global_faces={}):
    try:
        # Se la faccia é già stata riconosciuta, ritorna il risultato
        if face in global_faces:
            return global_faces[face]

        # Altrimenti, riconosci la faccia
        result = DeepFace.find(
            img_path=face,
            db_path=db_path,
            model_name="VGG-Face",
            enforce_detection=False,
        )
        identities = []
        if len(result) != 0:
            # mi prendo il primo risultato
            distances = result[0]["VGG-Face_cosine"].tolist()
            for index in range(0, len(result[0]["identity"])):
                identities.append(
                    result[0]["identity"][index].split("/")[-1].split(".")[0]
                )
            p = result[0]["identity"][0].split("/")[-1].split(".")[0]

            # Extract face bounding box coordinates
            bounding_box_df = result[0].loc[:, ['target_x', 'target_y', 'target_w', 'target_h']]
            bounding_box_values = bounding_box_df.iloc[0].tolist()
            x, y, w, h = bounding_box_values

            global_faces[face] = (distances, identities, p, (x, y, w, h))
            return (distances, identities, p, (x, y, w, h))
    except Exception as e:
        print(f"Error in face_recogn: {e}")
        return (None, None, None, None)


def draw_rectangle(image_path, bounding_box):
    image = cv2.imread(image_path)
    x, y, w, h = bounding_box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    cv2.imwrite(img_path + "/" + image_path.split("/")[-1], image)



def DeepFace_study():
    # Setup initial folders
    if not os.path.exists("images"):
        os.mkdir("images")

    if not os.path.exists("images/graphs"):
        os.mkdir("images/graphs")

    if not os.path.exists("images/deepface"):
        os.mkdir("images/deepface")

    # impostori = 10
    # genuini = len(os.listdir(db_path))  # 39 a livello di identitá
    total_probe_number = 0
    genuine_probes = 0  # 93    a livello di immagini
    impostor_probes = 0  # 65    a livello di immagini

    DI = 0
    GR = 0
    FA = 0
    FR = 0

    # per ogni cartella in probe, creami un dizionario con chiave il nome della cartella
    # e valore false, essendo il dizionario degli impostori
    # se la cartella NON é in training, allora é un impostore,
    # e quindi setto il valore a true
    impostors_subjects = {s: False for s in os.listdir(probe_path)}
    for s in impostors_subjects:
        if s not in os.listdir(db_path):
            impostors_subjects[s] = True

    # Calcolo del numero totale delle probe
    for s_probes in os.listdir(probe_path):
        total_probe_number += len(os.listdir(probe_path + "/" + s_probes))

        for _ in os.listdir(probe_path + "/" + s_probes):
            if s_probes in impostors_subjects and impostors_subjects[s_probes]:
                impostor_probes += 1
            else:
                genuine_probes += 1

    dir_array = []
    far_array = []
    threshold_array = []
    start_threshold = 0.01
    end_threshold = 0.99
    step = 0.01
    global_faces = {}

    for threshold in np.arange(start_threshold, end_threshold, step):
        threshold_array.append(threshold)
        DI = 0
        GR = 0
        FA = 0
        FR = 0
        # Ciclo su tutte le cartelle s della probe
        for s_probes in os.listdir(probe_path):
            # Ciclo su tutte le cartelle s del database
            # Per ogni faccia
            # Per ogni immagine in ogni cartellina S del probe (faccia)

            for face_probe in os.listdir(probe_path + "/" + s_probes):
                probed_face = probe_path + "/" + s_probes + "/" + face_probe
                print("I'm probing: " + probed_face)
                (distances, identities, presumed_identity, bounding_box) = face_recogn(probed_face, global_faces)

                # questi valori mi servono per le distanze per confrontarle con
                # threshold
                # print("distance", distances)
                # print("identity",identities)
                # print("presumed_identity", presumed_identity)
                if distances and identities and bounding_box:
                    draw_rectangle(probed_face, bounding_box)
                    cleaned_presumed_identity = "".join(
                        char for char in presumed_identity if not char.isdigit()
                    )
                    cleaned_face_probe = "".join(
                        char for char in face_probe.split(".")[0] if not char.isdigit()
                    )

                    if distances[0] <= threshold:  # la prima distanza è quella giusta
                        # accetta la threshold e l'identità è corretta
                        if cleaned_presumed_identity == cleaned_face_probe:
                            DI += 1
                        else:
                            # accetta la threshold ma le label non coincidono:
                            # se si tratta di un genuine FA
                            # se si tratta di un impostor FR
                            if (
                                s_probes in impostors_subjects
                                and impostors_subjects[s_probes]
                            ):
                                FA += 1
                            # else:
                            #     FR += 1
                    # se la soglia è 0.11, cioè è più bassa della distanza più piccola
                    # False Acceptance non esiste perchè avendo una distanza < della
                    # threshold non avremmo nessun ACCEPTANCE
                    else:
                        # se la probe e la presumed identity (identity[0]) coincidono --> FALSE REJECTION
                        if cleaned_presumed_identity != cleaned_face_probe:
                            GR += 1
                        # se le due invece erano diverse "Ai_Sugiyama" e "Leo_DiCaprio" --> GENUINE REJECTION
                        # else:
                        #     GR += 1
                else:  # se non riconosce nessuna identity
                    GR += 1  # true negative
                # per semplicità una faccia non riconosciuta la considero come un GR
                # anche se in realtà dovrebbe essere tolta dalle probe totali

        # l'obiettivo è per ogni distanza che viene calcolata capire se è >=< della threshold e in base a questo
        # aggiornare i rispettivi FR,FA,GA,GR

        # valori
        # DIR = true_pos / genuini          #Detection and Identification Rate, at rank 1
        # FAR = false_pos / impostor_probe  #False Acceptance Rate = #FA / impostors
        # FRR = 1 - DIR                     #False Rejection Rate = 1-DIR(t,1) with threshold t

        DIR = DI / genuine_probes
        FAR = FA / impostor_probes
        FRR = 1 - DIR
        dir_array.append(DIR)
        far_array.append(FAR)
        with open("results.txt", "a") as f:
            f.write(f"DI: {DI}\n")

            f.write(f"GR: {GR}\n")
            f.write(f"FA: {FA}\n")
            f.write(f"FR: {FR}\n\n")
            f.write("============================================\n\n")
            f.write(
                f"At threshold: {threshold} we measured DIR: {DIR} FAR: {FAR}, and FRR:{FRR}\n\n"
            )
            f.write(f"DIR= {dir_array}")
            f.write(f"FAR= {far_array}")

    # TODO: visualizzare graficamente le facce

    ###FIRST FIGURE
    # PLOT ROC
    x = np.array(far_array)
    y = np.array(dir_array)
    plt.plot(x, y)
    plt.xlabel("FAR")
    plt.ylabel("1-FRR")
    plt.title("ROC")
    plt.axline((0, 0), slope=1, linestyle="--")
    plt.savefig("images/graphs" + "/ROC_deepface.png")
    plt.show()

    ###SECOND FIGURE

    # plot FAR with threshold
    xPlot = np.array(threshold_array)
    yPlot = np.array(far_array)
    # let's use the monotone cubic spline
    X_Y_Spline = PchipInterpolator(xPlot, yPlot)
    X_ = np.linspace(xPlot.min(), xPlot.max(), 500)
    Y_ = X_Y_Spline(X_)
    plt.plot(X_, Y_, color="green")

    # plot FRR with threshold
    frr_array = []
    for i in range(0, len(dir_array)):
        frr_array.append(float(1) - float(dir_array[i]))
    xPlot = np.array(threshold_array)
    yPlot = np.array(frr_array)
    # let's use the monotone cubic spline
    X_Y_Spline = PchipInterpolator(xPlot, yPlot)
    X1_ = np.linspace(xPlot.min(), xPlot.max(), 500)
    Y1_ = X_Y_Spline(X1_)
    plt.plot(X1_, Y1_, color="blue")

    # EER
    x3, y3 = intersection(X1_, Y1_, X_, Y_)
    plt.plot(x3, y3, "ro")
    print(
        "Final ERR: found for THRESHOLD t = "
        + str(x3[0])
        + " and FRR(t) = FAR(t) = "
        + str(y3[0])
    )
    plt.axvline(x3[0], color="r", linestyle="--")

    plt.title("Plot FAR and FRR intersection in the EER")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.savefig("images/graphs" + "/EER_deepface.png")
    plt.show()

