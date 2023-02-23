# Usage: python3 recognize.py -i <path_to_test_image_dir> -o <path_to_output_dir> -c <path_to_collect_dir> [-l ]

import os, sys
import cv2 
from imutils import paths
import pickle
import face_recognition
import argparse, shutil

# load the reference encodings created in the script album.py
data = pickle.loads(open('friends_face_encodings.pickle', "rb").read())

# make the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# provide:
# a path to the directory containing test images and 
# a path to the directory where you would like to save your output data and
# a path to the directory where you would like to save clean copies of the images where a match was found
# set the limit option if you want only matched images in the output directory
ap.add_argument("-i", "--test_directory", required=True,
	help="path to the test image directory")   
ap.add_argument("-o", "--output_directory", required=True,
	help="path to the output directory")
ap.add_argument("-c", "--collect_directory", required=True,
	help="path to the collect directory")
ap.add_argument ('-l', "--limit", action = "store_true", dest = "OnlyMatched",
	help="limit files copied to output directory only to files with a match")
args = vars(ap.parse_args())


test_dir = args["test_directory"]
output_dir = args["output_directory"]
collect_dir = args["collect_directory"]
OnlyMatched = args["OnlyMatched"]

#  Open for write a file which will contain a list of the files with matched faces
ListFiles = collect_dir+"/matched.csv"
x = open (ListFiles, "w")

# initialize a map linking the faces and the filenames they are found in the output
filemap = {names: [] for names in data["names"]}

# loop over all the images in the test directory tree
# for count, image in enumerate(os.listdir(test_dir)):
for subdir, dirs, files in os.walk(test_dir):
    for filename in sorted(files):
        imagepath = os.path.join(subdir, filename)   
        filename = imagepath.split(os.path.sep)[-1]
        
        dummy, file_extension = os.path.splitext(filename)

        # Skip non image files
        if(file_extension.lower() in [".png",".jpg",".jpeg",".gif" ]):

            # load the image
            testimage = cv2.imread(imagepath)

            # extract the position of bounding box of the face and their corresponding face encodings
            bboxes = face_recognition.face_locations(testimage, model='hog')
            encodings = face_recognition.face_encodings(testimage, bboxes)

            names = []
            save = False
            # loop over the found encodings and compare it to the encodings in the reference database
            for encoding in encodings:

                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"

                # if the test image has even a single face that matched a face in the database
                if True in matches:

                    # extract the matched indices 
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # extract the corresponding names of the matched indices and get a vote count for each matched face name 
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # the name of the face with maximum number of votes wins
                    name = max(counts, key=counts.get)
                    save = True

                names.append(name)
            # Copuy the matched file to collect_dir
            if save:
                shutil.copy2(imagepath, collect_dir)
                x.write(imagepath+'\n')
                
                
            # draw the bounding box around the faces with their detected names
            for ((top, right, bottom, left), name) in zip(bboxes, names):

                cv2.rectangle(testimage, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(testimage, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if name != 'Unknown':
                    filemap[name].append(filename)
            if (not OnlyMatched or save): 
                outputname = "output_%s" % filename
                outputpath = os.path.join(output_dir, outputname)
                cv2.imwrite(outputpath, testimage)
