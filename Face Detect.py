import cv2
import glob

"""The Face detection code in this python file was modified from Paul Vangent's image classifier found here:
van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from:
http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/"""


def detect_faces(emotion):
    # Get list of all images with emotion
    files = glob.glob("FolderContainingImages\\%s\\*" %emotion)
    file_number = 0
    # declare a list of all the Face Detection HAAR Classifiers
    face_classifiers = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml",
                           "haarcascade_frontalface_alt.xml", "haarcascade_frontalface_alt_tree.xml"]

    for f in files:
        # Open image and convert to gray scale
        frame = cv2.imread(f)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for c in face_classifiers:
            face_detect = cv2.CascadeClassifier(c)
            face = face_detect.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

            if len(face) == 1:
                print("face found in file: %s" % f)
                get_features(face, gray_image, emotion, file_number)
                break

        # Increment image number
        file_number += 1


def get_features(face_features, face_image, emotion, file_number):
    # Cut and save face
    for (x, y, w, h) in face_features:
        # get coordinates and size of rectangle containing face, Cut the frame to size
        face_image = face_image[y:y + h, x:x + w]

        try:
            # Resize face so all images have same size and write to file
            # output = cv2.resize(face_image, (300, 300))
            # cv2.imwrite("validation_set_7_emotions\\%s\\%s.jpg" % (emotion, file_number), face_image)
            cv2.imwrite("FolderToSave\\%s\\%s.jpg" % (emotion, emotion + str(file_number)), face_image)

        except:
            pass


def main():
    for e in ["Happy", "Not_Happy"]:
        # detect faces under each emotion category
        detect_faces(e)
    return 0


if __name__ == "__main__":
    main()