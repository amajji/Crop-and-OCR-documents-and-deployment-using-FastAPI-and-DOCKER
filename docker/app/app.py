from app.schemas import GETPATH
import io

# import pytesseract
import re
import time
import cv2
import uvicorn
from fastapi import FastAPI, Request
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from pdf2image import convert_from_path, convert_from_bytes

from fastapi.logger import logger
import logging








try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
from sklearn.cluster import KMeans

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'your path to the exec file'

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
settings = GETPATH()


gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)



def centroid_histogram(clt):
    """
    This function returns a color's histogram from centroid's position
    input :
        - clt : kmeans model after training
    """
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    """
    This function plots the colors's histogram
    input :
        - clt : kmeans model after training
        - centroids : centroids's
    """
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(
            bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1
        )
        startX = endX

    # return the bar chart
    return bar


def get_croners(bckgnd_masked, cntr_1, M, result_finale):
    """
    This function return the four corners of the document to
    zoom in it
    Input :
        - bckgnd_masked : mask of the image after rotation
        - cntr_1 : the external countour
        - M : Matrix of rotation
        - result_finale : input image
    """

    output_image = result_finale.copy()

    approx = cv2.approxPolyDP(cntr_1, 0.009 * cv2.arcLength(cntr_1, True), True)

    # draws boundary of contours.
    cv2.drawContours(bckgnd_masked, [approx], 0, (0, 0, 255), 5)

    # Used to flatted the array containing
    # the co-ordinates of the vertices.
    n = approx.ravel()
    i = 0
    points = []
    for j in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]

            # String containing the co-ordinates.
            string = str(x) + " " + str(y)
            points.append([x, y])
        i = i + 1

    # rotate the points
    # using the rotation matrix
    points = np.array(points)

    # add ones
    ones = np.ones(shape=(len(points), 1))

    points_ones = np.hstack([points, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T

    # Get the x and y axis coordinates
    x_axis = transformed_points[:, 0].tolist()
    y_axis = transformed_points[:, 1].tolist()

    # In case we have 4 corners
    if len(x_axis) == 4:
        # In this part, we ge the coordinates of the lower point in
        # the upper half of the image and the highest point in the down
        # half of the image
        # Get the corner coordinates of the upper half of the image
        y_axis_array = np.array(y_axis)
        upper_index_y = np.where(y_axis_array > int(result_finale.shape[0] // 2))[
            0
        ].tolist()

        # Get the corner coordinates of the down half of the image
        down_index_y = [i for i in range(len(y_axis)) if i not in upper_index_y]

        # Get the corner coordinates of the right half of the image
        x_axis_array = np.array(x_axis)
        upper_index_x = np.where(x_axis_array > int(result_finale.shape[1] // 2))[
            0
        ].tolist()

        # Get the corner coordinates of the left half of the image
        down_index_x = [i for i in range(len(x_axis)) if i not in upper_index_x]

        # Get the y coordinate of the highest corner in the upper half of the image
        index_y1 = [y_axis[ele] for ele in upper_index_y].index(
            np.min([y_axis[ele] for ele in upper_index_y])
        )
        y1 = [y_axis[ele] for ele in upper_index_y][index_y1]

        # Get the y coordinate of the lowest corner in the upper half of the image
        index_y2 = [y_axis[ele] for ele in down_index_y].index(
            np.max([y_axis[ele] for ele in down_index_y])
        )
        y2 = [y_axis[ele] for ele in down_index_y][index_y2]

        # Get the x coordinate of the highest corner in the left half of the image
        index_x1 = [x_axis[ele] for ele in upper_index_x].index(
            np.min([x_axis[ele] for ele in upper_index_x])
        )
        x1 = [x_axis[ele] for ele in upper_index_x][index_x1]

        # Get the x coordinate of the highest corner in the right half of the image
        index_x2 = [x_axis[ele] for ele in down_index_x].index(
            np.max([x_axis[ele] for ele in down_index_x])
        )
        x2 = [x_axis[ele] for ele in down_index_x][index_x2]

        # Output image
        output_image = result_finale[
            int(np.min([y1, y2])) : int(np.max([y1, y2])),
            int(np.min([x1, x2])) : int(np.max([x1, x2])),
            :,
        ]

    return output_image




def cropp_image_kmeans(filename, file_path, uploader_path):
    """
    This function cropp any document from any image

    inputs :
        - file_path : the input image's path
    output :
        - the extracted image of the document
    """
    # Output text 
    text = ""

    # Read the image and transform it to HSV color space
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

    # plt.figure(figsize = (20,25))
    # plt.imshow(image)

    # reshape the image to be a list of pixels
    image_reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster the pixel intensities using Kmeans
    clt = KMeans(n_clusters=4)
    cluster_labels = clt.fit_predict(image_reshape)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    # plt.figure(figsize = (20,25))
    # plt.imshow(bar)

    # Get the lightest color
    list_colors = [np.sum(list_ele) for list_ele in clt.cluster_centers_.tolist()]

    # normalize the list of colors to get the intensity
    list_colors_norm = list_colors / np.max(list_colors)

    # get the color's index with an intensity upper to 0.85
    index_colors = [
        i for i in range(len(list_colors_norm)) if list_colors_norm[i] > 0.85
    ]

    # get the cluster including their index in index_colors
    labels = clt.labels_
    labels = np.array([1 if ele in index_colors else 0 for ele in labels.tolist()])

    # create a mask from label vector
    mask = (labels.reshape(*gray.shape) * 255).astype(np.uint8)

    # plt.figure(figsize = (20,25))
    # plt.imshow(mask)

    # Taking a matrix of size 3 as the kernel
    kernel = np.ones((3, 3), np.uint8)

    # dilate the mask to smooth it
    img_dilation = cv2.dilate(mask, kernel, iterations=1)

    # get edges
    canny = cv2.Canny(img_dilation, 50, 200)

    # shapes
    hh, ww = canny.shape[:2]

    # plt.figure(figsize = (20,25))
    # plt.imshow(img_dilation)

    # get contours
    contours = cv2.findContours(
        img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Get the biggest countour and filter out small regions
    list_cntr_area, list_area = [], []
    for cntr in contours:
        area = cv2.contourArea(cntr)
        list_cntr_area.append((area, cntr))
        list_area.append(area)

    # filter out small regions
    cimg = np.zeros_like(canny)
    index_max_1 = list_area.index(np.max(list_area))
    cntr_1 = list_cntr_area[index_max_1][1]

    # draw countour
    cv2.drawContours(cimg, [cntr_1], 0, 255, 1)

    # plt.figure(figsize = (20,25))
    # plt.imshow(cimg)

    # get convex hull and draw on input
    points = np.column_stack(np.where(cimg.transpose() > 0))
    hull = cv2.convexHull(points)
    himg = image.copy()
    cv2.polylines(himg, [hull], True, (0, 0, 255), 1)

    # draw convex hull as filled mask
    mask = np.zeros_like(cimg, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # blacken out input using mask
    mimg = image.copy()
    mimg = cv2.bitwise_and(mimg, mimg, mask=mask)

    # invert mask so shapes are white on black background
    mask_inv = 255 - mask

    # create new (white) background
    bckgnd = np.full_like(image, (255, 255, 255))

    # apply inverse mask to background
    bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

    # add together
    result_intermedaire = cv2.add(mimg, bckgnd_masked)

    # plt.figure(figsize = (20,25))
    # plt.imshow(result_intermedaire)

    # get rotate rectangle
    rotrect = cv2.minAreaRect(hull)
    (center), (width, height), angle = rotrect
    box = cv2.boxPoints(rotrect)
    boxpts = np.int0(box)

    # draw rotated rectangle on copy of input
    rimg = result_intermedaire.copy()
    cv2.drawContours(rimg, [boxpts], 0, (0, 0, 255), 1)

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle tends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, check width vs height
    else:
        if width > height:
            angle = -(90 + angle)

        else:
            angle = -angle

    # negate the angle to unrotate
    neg_angle = -angle
    print("unrotation angle:", neg_angle)
    print("")

    # Get rotation matrix
    # center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, neg_angle, scale=1.0)

    # unrotate to rectify
    result = cv2.warpAffine(
        result_intermedaire,
        M,
        (ww, hh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # plt.figure(figsize = (20,25))
    # plt.imshow(result)

    # Rotated mask
    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (ww, hh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # invert mask so shapes are white on black background
    mask_inv_rotated = 255 - rotated_mask

    # create new (blue) background
    bckgnd = np.full_like(result, (255, 255, 255))

    # apply inverse mask to background
    bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv_rotated)

    # add together
    result_finale = cv2.add(result, bckgnd_masked)

    # zoom in the document
    result_finale = get_croners(bckgnd_masked, cntr_1, M, result_finale)

    # get the coordinate of the image center
    a, b = int(result_finale.shape[0]) // 2, int(result_finale.shape[1]) // 2

    # save the image bloc
    cv2.imwrite(
        uploader_path + "/" + filename.split(".")[0] + "____." + filename.split(".")[1],
        result_finale[:a, :, :],
    )

    try:

        # Use pytesseract to get the letters's orientation
        rotate = pytesseract.image_to_osd(
            uploader_path
            + "/"
            + filename.split(".")[0]
            + "____."
            + filename.split(".")[1]
        )

        # rotate the document
        angle_rotation = re.search("(?<=Rotate: )\d+", rotate).group(0)
        result_finale = Image.fromarray(result_finale, "RGB").rotate(
            int(angle_rotation)
        )

    except:
        # exception in case pytesseract couldn't get the text orientation
        print("Pytesseract failed !")

    # Get the output
    #text = str(pytesseract.image_to_string(result_finale, config="--psm 6"))

    return result_finale, text






@app.get("/")
def window_princip(request: Request):
    return templates.TemplateResponse(
        "window_second.html", context={"request": request}
    )




@app.post("/uploader_")
async def uploader(request: Request, file_1: UploadFile = File(...)):
    # if request.method == 'POST':
    doc = await file_1.read()
    path = settings.UPLOAD_FOLDER

    # Store all the pages of the PDF in a variable
    # pages = convert_from_path(file_1.filename)
    print("---------------filename --- ", file_1.filename)

    file_bytes = file_1.file.read()
    with open(path + file_1.filename, "wb") as f:
        f.write(doc)

    # path of the image
    path_img = path + file_1.filename

    result_finale, text = cropp_image_kmeans(file_1.filename, path_img, path)

    # save the output text in a txt file
    with open(path + "output_text.txt", "w") as file:
        file.write(text)
        file.close()

    # save the image bloc
    cv2.imwrite(path + "resultat_finale.jpg", np.array(result_finale))

    # cv2.imwrite(app.config['UPLOAD_FOLDER']  + "/img_result.jpg",  result_finale)
    # cv2.imwrite(path + "/" + "/img_result.jpg", result_finale)
    return templates.TemplateResponse("window_download.html", {"request": request})




@app.get("/download")
async def download(request: Request):
    filename = "resultat_finale.jpg"
    path = settings.UPLOAD_FOLDER + filename
    time.sleep(5)
    return FileResponse(path, filename=filename)




@app.get("/Acceuil")
def Acceuil(request: Request):
    return templates.TemplateResponse("window_second.html", {"request": request})





# Run the application
if __name__ == "__main__":
    uvicorn.run(app, debug=True)
