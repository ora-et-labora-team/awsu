from DeepFaceRecognition import DeepFace_study, concatenate_images_in_grid
from LBPHFaceRecognition import LBPH_study
import matplotlib.pyplot as plt
import os
import tkinter

def _show_graphs(img1, img2, img3, img4):
    # Define the paths to the images
    eer_path_lbph = img1
    roc_path_lbph = img2
    eer_path_deepface = img3
    roc_path_deepface = img4
    
    # Check if the image files exist
    if not (os.path.exists(eer_path_lbph) and os.path.exists(roc_path_lbph)):
        print("Image files not found.")
        return

    # Check if the image files exist
    if not (os.path.exists(eer_path_deepface) and os.path.exists(roc_path_deepface)):
        print("Image files not found.")
        return

    # Load and display EER.png
    eer_img_lbph = plt.imread(eer_path_lbph)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(eer_img_lbph)
    plt.title("EER Graph")
    plt.axis('off')

    # Load and display ROC.png
    roc_img_lbph = plt.imread(roc_path_lbph)
    plt.subplot(2, 1, 2)
    plt.imshow(roc_img_lbph)
    plt.title("ROC Graph")
    plt.axis('off')

    eer_img_deepface = plt.imread(eer_path_deepface)
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(eer_img_deepface)
    plt.title("EER Graph")
    plt.axis('off')

    # Load and display ROC.png
    roc_img_deepface = plt.imread(roc_path_deepface)
    plt.subplot(2, 1, 2)
    plt.imshow(roc_img_deepface)
    plt.title("ROC Graph")
    plt.axis('off')


    # Show the images in a new window
    plt.tight_layout()

    # hide axis
    plt.axis('off')
    plt.show()
    

def show_graphs():
    concatenate_images_in_grid("images/lbph", "images/lbph/final.jpg", grid_cols=4)
    concatenate_images_in_grid("images/deepface", "images/deepface/final.jpg", grid_cols=13)
    _show_graphs("images/graphs/EER_lbph.png", "images/graphs/ROC_lbph.png", "images/graphs/EER_deepface.png", "images/graphs/ROC_deepface.png")

window = tkinter.Tk()
window.title("Recognizer Comparator")

frame = tkinter.Frame(window)
frame.pack()

# Saving Info
info_frame = tkinter.LabelFrame(frame, text="Recognizer technical informations")
info_frame.grid(row=0, column=0, padx=20, pady=20)
for widget in info_frame.winfo_children():
    widget.grid_configure(padx=10, pady=20)
button = tkinter.Button(frame, text="         LBP         ", command=LBPH_study)
button.grid(row=0, column=0, sticky="news", padx=20, pady=10)

button = tkinter.Button(frame, text="Deep Learning", command=DeepFace_study)
button.grid(row=0, column=2, sticky="news", padx=20, pady=10)

button = tkinter.Button(frame, text="Show Graphs", command=show_graphs)
button.grid(row=0, column=4, sticky="news", padx=20, pady=10)

window.mainloop()
