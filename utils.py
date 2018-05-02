import os
import pickle
import random
import xml.etree.ElementTree as ET

import numpy as np
import svgwrite
from IPython.display import SVG, display


def get_one_letter(data, n):
    """Get one letter and return absolute coords.

    Last point is labeled by 1 in stroke column
    """
    num = np.sum(data[:, 2])  # total number of pen lifts, number of letters is +1
    idx = np.where(data[:, 2] == 1)[0]
    idx = np.concatenate(([0], idx))  # 0 is starting index, se -1 as we increment next
    r = range(idx[n] + 1, idx[n + 1] + 1)  # include 1 as last point from letter, but start from next after 1
    x = np.cumsum(data[r, 0])
    y = np.cumsum(data[r, 1])
    return x, y


def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    # strokes contain first point recorded as absolute, next as relative to previous.
    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        # find boundaries after tracking those relative and absolute
        # it soes not capture possible pen lifts up
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

# old version, where each path is entire stroke (smaller svg size, but
# have to keep same color)


def draw_strokes(data, factor=10, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1

    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)

    command = "m"

    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "

    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

    dwg.save()
    display(SVG(dwg.tostring()))


def draw_strokes_eos_weighted(
        stroke,
        param,
        factor=10,
        svg_filename='sample_eos.svg'):
    c_data_eos = np.zeros((len(stroke), 3))
    for i in range(len(param)):
        # make color gray scale, darker = more likely to eos
        c_data_eos[i, :] = (1 - param[i][6][0]) * 225
    draw_strokes_custom_color(
        stroke,
        factor=factor,
        svg_filename=svg_filename,
        color_data=c_data_eos,
        stroke_width=3)


def draw_strokes_random_color(
        stroke,
        factor=10,
        svg_filename='sample_random_color.svg',
        per_stroke_mode=True):
    c_data = np.array(np.random.rand(len(stroke), 3) * 240, dtype=np.uint8)
    if per_stroke_mode:
        switch_color = False
        for i in range(len(stroke)):
            if switch_color == False and i > 0:
                c_data[i] = c_data[i - 1]
            if stroke[i, 2] < 1:  # same strike
                switch_color = False
            else:
                switch_color = True
    draw_strokes_custom_color(
        stroke,
        factor=factor,
        svg_filename=svg_filename,
        color_data=c_data,
        stroke_width=2)


def draw_strokes_custom_color(
        data,
        factor=10,
        svg_filename='test.svg',
        color_data=None,
        stroke_width=1):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1
    abs_x = 25 - min_x
    abs_y = 25 - min_y

    for i in range(len(data)):

        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor

        prev_x = abs_x
        prev_y = abs_y

        abs_x += x
        abs_y += y

        if (lift_pen == 1):
            p = "M " + str(abs_x) + "," + str(abs_y) + " "
        else:
            p = "M +" + str(prev_x) + "," + str(prev_y) + \
                " L " + str(abs_x) + "," + str(abs_y) + " "

        lift_pen = data[i, 2]

        the_color = "black"

        if (color_data is not None):
            the_color = "rgb(" + str(int(color_data[i, 0])) + "," + str(
                int(color_data[i, 1])) + "," + str(int(color_data[i, 2])) + ")"

        dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill(the_color))
    dwg.save()
    display(SVG(dwg.tostring()))


def draw_strokes_pdf(data, param, factor=10, svg_filename='sample_pdf.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    abs_x = 25 - min_x
    abs_y = 25 - min_y

    num_mixture = len(param[0][0])

    for i in range(len(data)):

        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor

        for k in range(num_mixture):
            pi = param[i][0][k]
            if pi > 0.01:  # optimisation, ignore pi's less than 1% chance
                mu1 = param[i][1][k]
                mu2 = param[i][2][k]
                s1 = param[i][3][k]
                s2 = param[i][4][k]
                sigma = np.sqrt(s1 * s2)
                dwg.add(dwg.circle(center=(abs_x + mu1 * factor,
                                           abs_y + mu2 * factor),
                                   r=int(sigma * factor)).fill('red',
                                                               opacity=pi / (sigma * sigma * factor)))

        prev_x = abs_x
        prev_y = abs_y

        abs_x += x
        abs_y += y

    dwg.save()
    display(SVG(dwg.tostring()))


class DataLoader():
    """
    Examples:
    draw_strokes(data_loader.raw_data[0]) - one stroke (many letters - this is one sentence written by person)
    """

    def __init__(
            self,
            batch_size=50,
            seq_length=300,
            scale_factor=10,
            limit=500):
        self.data_dir = "./data"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.scale_factor = scale_factor  # divide data by this factor
        self.limit = limit  # removes large noisy gaps in the data

        data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")
        raw_data_dir = self.data_dir + "/lineStrokes"

        if not (os.path.exists(data_file)):
            print("creating training data pkl file from raw source")
            self.preprocess(raw_data_dir, data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_dir, data_file):
        # create data file from raw xml files from iam handwriting source.

        # build the list of xml files
        filelist = []
        # Set the directory you want to start from
        rootDir = data_dir
        for dirName, subdirList, fileList in os.walk(rootDir):
            #print('Found directory: %s' % dirName)
            for fname in fileList:
                #print('\t%s' % fname)
                filelist.append(dirName + "/" + fname)

        # function to read each individual xml file
        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()

            result = []

            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100

            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append(
                        [float(point.attrib['x']) - x_offset, float(point.attrib['y']) - y_offset])
                result.append(points)

            return result

        # converts a list of arrays into a 2d numpy int16 array
        def convert_stroke_to_array(stroke):

            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j]) - 1)):  # end of stroke
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        # build stroke database of every xml file inside iam database
        strokes = []
        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                print('processing ' + filelist[i])
                strokes.append(
                    convert_stroke_to_array(
                        getStrokes(
                            filelist[i])))

        f = open(data_file, "wb")
        pickle.dump(strokes, f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        f = open(data_file, "rb")
        # contains array of strokes (each can be many letters). Each letter starts with absolute coords
        # and ends with 1 in pen column:
        # raw_data[10][:100,:] =
        # array([[ 322,  306,    0],
        # [  -4,    4,    0],
        # [  -9,   13,    0],
        # [ -15,   21,    0],
        # [ -27,   26,    0],
        # [ -31,   38,    0],
        # [ -31,   45,    0],
        # [ -31,   48,    0],
        # [ -32,   41,    0],
        # [ -26,   33,    0],
        # [ -11,   35,    0],
        # [  -5,   17,    0],
        # [   6,   16,    0],
        # [   9,    8,    0],
        # [  18,    7,    0],
        # [  19,    4,    0],
        # [  19,  -12,    1],
        # [ 239, -302,    0],
        # [  -4,    5,    0],
        # [  -4,   13,    0],
        self.raw_data = pickle.load(f)
        f.close()

        # goes thru the list, and only keeps the text entries that have more
        # than seq_length points
        # list of processed (scaled, removed long distances) strokes. Each stroke has many letters (1s in 3rd column)
        self.data = []
        self.valid_data = []
        counter = 0

        # every 1 in 20 (5%) will be used for validation data
        cur_data_counter = 0
        for data in self.raw_data:  # over strokes (not letters)
            if len(data) > (self.seq_length + 2):
                # removes large gaps from the data
                data = np.minimum(data, self.limit)  # replace values in data by limit if the are larger than limit
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)  # convert to float
                data[:, 0:2] /= self.scale_factor
                cur_data_counter = cur_data_counter + 1
                if cur_data_counter % 20 == 0:
                    self.valid_data.append(data)
                else:
                    self.data.append(data)
                    # number of equiv batches this datapoint is worth
                    # data are points that form one stroke (many letters)
                    # XXX: seq_length seems to be number of points ber batch
                    # From each stroke we will take seq_length points (only once, see next_batch)
                    counter += int(len(data) / ((self.seq_length + 2)))

        print("train data: {}, valid data: {}".format(
            len(self.data), len(self.valid_data)))
        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(counter / self.batch_size)

    def validation_data(self):
        # returns validation data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.valid_data[i % len(self.valid_data)]
            idx = 0
            x_batch.append(np.copy(data[idx:idx + self.seq_length]))
            y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
        return x_batch, y_batch

    def next_batch(self):
        """
        Assumes that all strokes are longer than demanded seq_length. From each stroke we pick one
        seqence of length of seq_length starting from randomly selected point.
        For each stroke we return x and y while y are shifted per one point.
        Each batch is collected from different stroke. So batch_size stands for number of strokes (different) taken
        to one batch. If stroke is long we allow to sample it again (do not move batch pointer) which makes sense
        because we use random starting point for each stroke.
        Each stroke can contain many pen lifts up
        """
        # returns a randomised, seq_length sized portion of the training data
        x_batch = []
        y_batch = []
        for i in range(self.batch_size):
            data = self.data[self.pointer]  # select next stroke
            # number of equiv batches this datapoint is worth
            n_batch = int(len(data) / ((self.seq_length + 2)))
            idx = random.randint(0, len(data) - self.seq_length - 2)  #
            x_batch.append(np.copy(data[idx:idx + self.seq_length]))
            y_batch.append(np.copy(data[idx + 1:idx + self.seq_length + 1]))
            # adjust sampling probability.
            if random.random() < (1.0 / float(n_batch)):
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()
        return x_batch, y_batch

    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0
