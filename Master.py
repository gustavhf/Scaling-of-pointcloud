import laspy
import numpy as np
import random
import tqdm
import math
import pathlib
import os
import shutil
import sklearn.cluster
import ransac
import pickle
import matplotlib.pyplot as plt

class LAS():

    data_formats = {('ctypes.c_ubyte', 0) : 0, ('ctypes.c_ubyte', 1) : 1, ('ctypes.c_char', 1) : 2, ('ctypes.c_ushort', 1) : 3, ('ctypes.c_short', 1) : 4, ('ctypes.c_ulong', 1) : 5, \
        ('ctypes.c_long', 1) : 6, ('ctypes.c_ulonglong', 1) : 7, ('ctypes.c_longlong', 1) : 8, ('ctypes.c_float', 1) : 9, ('ctypes.c_double', 1) : 10, ('ctypes.c_ubyte', 2) : 11, \
        ('ctypes.c_char', 2) : 12, ('ctypes.c_ushort', 2) : 13, ('ctypes.c_short', 2) : 14, ('ctypes.c_ulong', 2) : 15, ('ctypes.c_long', 2) : 16, ('ctypes.c_ulonglong', 2) : 17, \
        ('ctypes.c_longlong', 2) : 18, ('ctypes.c_float', 2) : 19, ('ctypes.c_double', 2) : 20, ('ctypes.c_ubyte', 3) : 21, ('ctypes.c_char', 3) : 22, ('ctypes.c_ushort', 3) : 23, \
        ('ctypes.c_short', 3) : 24, ('ctypes.c_ulong', 3) : 25, ('ctypes.c_long', 3) : 26, ('ctypes.c_ulonglong', 3) : 27, ('ctypes.c_longlong', 3) : 28, ('ctypes.c_float', 3) : 29, \
        ('ctypes.c_double', 3) : 30}

    def __init__(self, filename):
        self.filename = filename

    def set_file(self, filename):
        self.filename = filename

    def getXYPoint(self, idx): 
        las = laspy.file.File(self.filename, mode="rw")
        return las.x[idx], las.y[idx]

    def remove_offset(self):
        las = laspy.file.File(self.filename, mode="rw")
        las.header.offset = [0, 0, 0]
        las.close()

    def copy(self, new_filename):
        las = laspy.file.File(self.filename, mode="r")
        outlas = laspy.file.File(new_filename, mode="w", header=laspy.header.Header(point_format = 2))

        # Add all dimensions from existing file
        existing_dimensions = [x.name for x in outlas.point_format]
        for spec in las.point_format:
            if spec.name not in existing_dimensions:
                new_data_type = LAS.data_formats[(spec.Format, spec.num)]
                outlas.define_new_dimension(name=spec.name, data_type=new_data_type, description=spec.name)

        outlas.header.scale = las.header.scale
        outlas.header.offset = las.header.offset


        # Copy header attributes
        for spec in las.header.header_format:
            try:
                setattr(outlas, spec.name, getattr(las, spec.name))
            except:
                pass

        las.close()
        return outlas

    def scale(self, value):
        las = laspy.file.File(self.filename, mode="rw")
        las.header.scale = [las.header.scale[0] * value,\
                            las.header.scale[1] * value, \
                            las.header.scale[2] * value]
        las.close()

    def add_dimension(self, dimension_name, data_type, description, copy=False):
        """Adds a new dimension to the existing file"""
        if copy:
            # Copy the file in case of fuckups
            shutil.copy2(self.filename, "{}-kopi.las".format(self.filename[:-4]))

        las = laspy.file.File(self.filename, mode="r")
        outlas = self.copy("tmp.las")

        # Add the new dimension if it do not already exist
        try:
            getattr(outlas, dimension_name)
        except:
            outlas.define_new_dimension(name=dimension_name, data_type=data_type, description=description)
        
        # Add the point records
        for spec in las.point_format:
            setattr(outlas, spec.name, getattr(las, spec.name))

        # Close the files
        las.close()
        outlas.close()

        # Replace the original file with the temporary file
        os.remove(self.filename)
        os.rename("tmp.las", self.filename)

    def reset_dimension(self, dimension):
        las = laspy.file.File(self.filename, mode="rw")
        data = getattr(las, dimension)
        data[:] = 0
        setattr(las, dimension, data)
        las.close()

    def index_to_new_dimension(self):
        self.add_dimension("index", 5, "Index of point")

        las = laspy.file.File(self.filename, mode="rw")

        idxs = np.arange(las.header.records_count)

        setattr(las, "index", idxs)

        las.close()

    def slice_to_new_dimension(self, slice_dimension, slices):
        # Add slice dimension
        self.add_dimension("slice", 3, "")

        # Open file in read-write mode
        las = laspy.file.File(self.filename, mode="rw")

        # 
        data = getattr(las, slice_dimension)
        mins = data.min()
        maxs = data.max()
        total = maxs - mins
        slice_length = total / slices

        for i in range(slices):
            minimum = mins + (i * slice_length)
            maximum = mins + ((i+1) * slice_length)
            
            idx = (data > mins + i * slice_length) & (data <= mins + (i+1) * slice_length)

            slice_data = getattr(las, "slice")
            slice_data[idx] = i
            setattr(las, "slice", slice_data)

        las.close()

    def ransac_new_dimension(self, n_unit_threshold, slice_start, n_circles):
        # Add a new dimension holding the circle numbers
        self.add_dimension("circle", 3, "")
        self.reset_dimension("circle")

        # Open the file
        las = laspy.file.File(self.filename, mode="rw")

        mins = las.header.min
        maxs = las.header.max
        points = las.header.records_count
        area = (maxs[0]-mins[0])*(maxs[1]-mins[1])
        
        # Threshold distance
        unit_threshold = area / points
        threshold = n_unit_threshold*unit_threshold
        print(threshold)

        slice_data = getattr(las, "slice")
        circle_data = getattr(las, "circle")
        n_slices = slice_data.max()

        circle_number = 1

        # Circle format for evlrs: { slice: [[circle_number center_point, radius]]}
        circles = []

        for i in tqdm.tqdm(reversed(range(slice_start, n_slices)), total=n_slices-slice_start):
            
            for _ in range(n_circles):
                
                
                idx = (slice_data == i) & (circle_data == 0)
                if len(idx) < 3:
                    continue
                data = np.vstack([las.x[idx], las.y[idx]]).T
                new_idx = getattr(las, "index")[idx]
                
                # make ransac class
                # n: how many times try sampling
                rransac = ransac.RANSAC(las.x[idx].T, las.y[idx].T, 100)
                
                # execute ransac algorithm
                rransac.execute_ransac(threshold)
                
                # get best model from ransac
                a, b, r = rransac.best_model[0], rransac.best_model[1], rransac.best_model[2]

                if a is None or b is None or r is None:
                    continue
                # get the inliers or outliers
                inlier_idxs = []
                for idx, (x, y) in enumerate(zip(las.x[idx], las.y[idx])):

                    if np.abs(math.sqrt((x-a)**2 + (y-b)**2) - r) < threshold:
                        inlier_idxs.append(idx)

                original_idxs = new_idx[inlier_idxs]
                circle_data[original_idxs] = circle_number
                circles.append(Circle(circle_number, a, b, las.z[original_idxs], r, slice_number=i, support=len(inlier_idxs), original_idxs=original_idxs))
                setattr(las, "circle", circle_data)
                circle_number += 1
    

        las.close()
        return circles

    def ransac_2_new_dimension(self, n_unit_threshold, cylinder_value, circle_number_start):
        # Add a new dimension holding the circle numbers
        self.add_dimension("circle", 3, "")
        #self.reset_dimension("circle")

        # Open the file
        las = laspy.file.File(self.filename, mode="rw")

        mins = las.header.min
        maxs = las.header.max
        points = las.header.records_count
        area = (maxs[0]-mins[0])*(maxs[1]-mins[1])
        
        # Threshold distance
        unit_threshold = area / points
        threshold = n_unit_threshold*unit_threshold
        print(threshold)

        slice_data = getattr(las, "slice")
        cylinder_data = getattr(las, "cylinder")
        circle_data = getattr(las, "circle")
        n_slices = slice_data.max()
        
        circle_number = circle_number_start
        # Circle format for evlrs: { slice: [[circle_number center_point, radius]]}
        circles = []

        for i in tqdm.tqdm(reversed(range(n_slices)), total=n_slices):
            
            idx = (cylinder_data == cylinder_value) & (slice_data == i)
            if len(las.x[idx]) < 3:
                continue

            data = np.vstack([las.x[idx], las.y[idx]]).T
            new_idx = getattr(las, "index")[idx]
            
            # make ransac class
            # n: how many times try sampling
            rransac = ransac.RANSAC(las.x[idx].T, las.y[idx].T, 100)
            
            # execute ransac algorithm
            rransac.execute_ransac(threshold)
            
            # get best model from ransac
            a, b, r = rransac.best_model[0], rransac.best_model[1], rransac.best_model[2]
            if a is None or b is None or r is None:
                continue
            # get the inliers or outliers
            inlier_idxs = []
            for idx, (x, y) in enumerate(zip(las.x[idx], las.y[idx])):

                if np.abs(math.sqrt((x-a)**2 + (y-b)**2) - r) < threshold:
                    inlier_idxs.append(idx)

            original_idxs = new_idx[inlier_idxs]
            circle_data[original_idxs] = circle_number
            circles.append(Circle(circle_number, a, b, las.z[original_idxs], r, slice_number=i, support=len(inlier_idxs), original_idxs=original_idxs))
            setattr(las, "circle", circle_data)
            circle_number += 1

        las.close()
        return circles, circle_number

    def update_cylinders(self, cylinders):
        self.add_dimension("cylinder", 3, "")
        self.reset_dimension("cylinder")

        las = laspy.file.File(self.filename, mode="rw")

        for i, cyl in enumerate(cylinders):
            inliers = cyl.get_original_idxs()

            cylinder_data = getattr(las, "cylinder")
            cylinder_data[inliers] = (i+1)
            setattr(las, "cylinder", cylinder_data)

        las.close()

    def set_cylinders(self, cylinders, radius_multiplier=1.2):
        self.add_dimension("cylinder", 3, "")
        self.reset_dimension("cylinder")
        las = laspy.file.File(self.filename, mode="rw")

        # Get the x and y points
        data = np.vstack([las.x, las.y]).T

        for i, cyl in enumerate(cylinders):
            cyl.update()
            print("x: {:.3f}\ty: {:.3f}\tr: {:.3f}".format(cyl.x, cyl.y, cyl.radius))
            
            # Filter by disk around center point
            idx = ((data[:, 0] - cyl.x)**2 + (data[:, 1] - cyl.y)**2 < (cyl.radius*radius_multiplier)**2) & (((data[:, 0] - cyl.x)**2 + (data[:, 1] - cyl.y)**2) > (cyl.radius/radius_multiplier)**2)
            
            cylinder_data = getattr(las, "cylinder")
            cylinder_data[idx] = i+1
            setattr(las, "cylinder", cylinder_data)

        las.close()

    def set_cylinders_by_circles(self, cylinders):
        self.add_dimension("cylinder", 3, "")
        self.reset_dimension("cylinder")
        las = laspy.file.File(self.filename, mode="rw")

        for i, cyl in enumerate(cylinders):
            cyl.update()
            print("x: {:.3f}\ty: {:.3f}\tr: {:.3f}".format(cyl.x, cyl.y, cyl.radius))
            
            # Filter by disk around center point
            idxs = []
            [idxs.extend(x.original_idxs) for x in cyl]
            idx = np.zeros(las.header.records_count, dtype=bool)
            for i in idxs:
                idx[i] = True

            cylinder_data = getattr(las, "cylinder")
            cylinder_data[idx] = i+1
            setattr(las, "cylinder", cylinder_data)

        las.close()

    def cylinders_to_new_file(self, new_filename, a, b, radius):
        las = laspy.file.File(self.filename, mode="r")
        outlas = self.copy(new_filename)

        # Get the x and y points
        data = np.vstack([las.x, las.y]).T

        # Filter by circle around center point
        idx = (data[:, 0] - a)**2 + (data[:, 1] - b)**2 < (radius*1.1)**2
        print(np.sum(idx))

        # Add the point records to the empty file
        for spec in las.point_format:
            setattr(outlas, spec.name, getattr(las, spec.name)[idx])

        las.close()
        outlas.close()

    def filter(self, dimension, value):
        las = laspy.file.File(self.filename, mode="r")
        idxs = getattr(las, dimension) == value
        las.close()
        return idxs


def pickle_data(filename, data):
    with open(filename, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(data, filehandle)

def unpickle_data(filename):
    with open(filename, 'rb') as filehandle:
    # read the data as binary data stream
        data = pickle.load(filehandle)
    return data

class Cylinders():
    def __init__(self, cylinders=[], support_threshold=0):
        self.cylinders = cylinders
        self.support_threshold = support_threshold

    @classmethod
    def fromLAS(cls, filename):
        las = laspy.file.File(filename, mode="r")

        circle_data = las.circle
        circles = []
        for circle in range(1, circle_data.max):
            idx = (circle_data == circle)

            center_x, center_y, radius = ls_circle(las.x[idx], las.y[idx])

            circles.append(Circle(circle, center_x, center_y, las.z[idx], radius, original_idxs=[x for x in idx if idx]))


    def update(self):
        for cylinder in self.cylinders:
            cylinder.update()

    def addCylinder(self, cylinder):
        self.cylinders.append(cylinder)

    def getScale(self, true_radius, hw_ratio=[]):
        self.update()
        radius = sum([cylinder.radius for cylinder in self.cylinders]) / len(self.cylinders)
        height = sum([cylinder.height for cylinder in self.cylinders]) / len(self.cylinders)
        if hw_ratio:
            ratio = height / radius
            print(ratio)
            print(hw_ratio)
            if ratio > min(hw_ratio) and ratio < max(hw_ratio):
                print("hw_ratio is good: {}".format(ratio))
                return true_radius/radius
            return
        return true_radius/radius
        #return self.cylinders[0].get_scaling(true_radius, hw_ratio=hw_ratio)

    def setScale(self, scale):
        for cylinder in self.cylinders:
            cylinder.setScale(scale)
        self.update()

    def create_cylinders(self, circles):
        circles = [x for x in circles if x.support >= self.support_threshold]
        if len(circles) == 0:
            return self
        circles = sorted(circles, key = Circle.get_support, reverse=True)
        circles = sorted(circles, key = Circle.get_slice_number, reverse=True)
        self.cylinders.append(Cylinder([circles[0]]))
        for circ in tqdm.tqdm(circles, total=len(circles), desc="Creating cylinders"):
            if self.best_cylinder(circ) is not None:
                self.best_cylinder(circ).append_circle(circ)
            else:
                self.cylinders.append(Cylinder([circ]))
        self.remove_circle_cylinders()
        return self

    def best_cylinder(self, circ):
        best_cyl = None
        best_score = 0
        for cyl in self.cylinders:
            score = cyl.circle_score(circ)
            if score > best_score:
                best_score = score
                best_cyl = cyl
        return best_cyl

    def remove_breaches(self, threshold):
        for cyl in self.cylinders:
            cyl.remove_breaches(threshold)
        return self

    def remove_circle_cylinders(self):
        i = 0
        while i < len(self):
            if len(self.cylinders[i]) <= 1:
                self.cylinders.pop(i)
                i -= 1
            i += 1

    def remove_overlapping_cylinders(self):
        i = 0
        while i < len(self.cylinders):
            j = 0
            if i == j:
                i += 1
                continue
            while j < len(self.cylinders):
                cylinder1 = self.cylinders[i]
                cylinder2 = self.cylinders[j]
                if abs(cylinder1.x - cylinder2.x) < cylinder1.radius and abs(cylinder1.y - cylinder2.y) < cylinder1.radius:
                    if cylinder1.support > cylinder2.support:
                        self.cylinders.pop(j)
                        j -= 1
                    elif cylinder1.support < cylinder2.support:
                        self.cylinders.pop(i)
                        i -= 1
                j += 1
            i += 1

    def keep_best_cylinder(self):
        best_idx = 0
        best_support = 0
        for idx, cyl in enumerate(self.cylinders):
            if cyl.support > best_support:
                best_support = cyl.support
                best_idx = idx
        self.cylinders = [self.cylinders[best_idx]]
        return self.cylinders[0]

    def keep_n_best_cylinders(self, n):
        best_idxs = []
        best_supports = []
        for idx, cyl in enumerate(self.cylinders):
            haveAdded = False
            for i, support in enumerate(best_supports):
                if cyl.support > support:
                    best_supports.insert(i, cyl.support)
                    best_idxs.insert(i, idx)
                    haveAdded = True
                    break
            if len(best_supports) < n and not haveAdded:
                best_supports.append(cyl.support)
                best_idxs.append(idx)

            if len(best_supports) > n:
                best_supports.pop()
                best_idxs.pop()
        self.cylinders = [self.cylinders[i] for i in best_idxs]
        return self.cylinders[:n]

    def __len__(self):
        return len(self.cylinders)

    def __iter__(self):
        for cyl in self.cylinders:
            yield cyl

    @classmethod
    def fromPickle(cls, filename, support_threshold):
        return cls([], support_threshold).create_cylinders(unpickle_data(filename))

    def __str__(self):
        string = ""
        i = 1
        for cyl in self.cylinders:
            cyl.update()
            string += str(cyl) + ", support: {}\n".format(cyl.support)
            for circle in cyl.circles:
                string += "\t" + str(circle) + "\n\n"
            string += "\n"
            i += 1
        return string

class Cylinder():
    def __init__(self, circles=[]):
        self.circles = circles
        self.support = 0
        self.height = None
        self.radius = None
        self.x = None
        self.y = None
        self.z_points = []
        self.update()

    def get_scaling(self, radius, hw_ratio=[]):
        self.update()
        if hw_ratio:
            ratio = self.height / self.radius
            print(ratio)
            print(hw_ratio)
            if ratio > min(hw_ratio) and ratio < max(hw_ratio):
                print("hw_ratio is good: {}".format(ratio))
                return radius/self.radius
            return
        return radius/self.radius     
    
    def setScale(self, scale):
        for circle in self.circles:
            circle.setScale(scale)
        self.update()

    def append_circle(self, circ):
        if self.circle_score(circ) > 0:
            self.circles.append(circ)
            self.update_support()
            return True
        return False

    def circle_score(self, circ):
        """returns the score between cylinder and circle"""
        if circ.slice_number in [x.slice_number for x in self.circles]:
            return 0
        self.update()
        return circ.score(Circle(None, self.x, self.y, [], self.radius))

    
    def distance(self, circ):
        """Returns the minimum distance between cylinder and circle"""
        pass

    def remove_breaches(self, threshold):
        """
        Inspects that every circle in cylinder is inside given threshold
        Removes circles not fulfilling requirement
        """
        change = True
        while change:
            change = False
            idx = 0
            while idx < len(self.circles):
                if self.circles[idx].score(Circle(None, self.x, self.y, [], self.radius)) < threshold:
                    self.circles.pop(idx)
                    change = True
                    idx -= 1
                idx += 1
            self.update()
        return self


    def update(self):
        self.update_support()
        self.update_z()
        self.calculate_height()
        self.calculate_radius()
        self.calculate_center()
        

    def update_support(self):
        self.support = sum([x.support for x in self.circles])

    def calculate_height(self):
        max_height = max(self.z_points)
        min_height = min(self.z_points)
        self.height = max_height-min_height
        """
        max_height = max([x.slice_number for x in self.circles])
        min_height = min([x.slice_number for x in self.circles])
        self.height = max_height - min_height
        """
        return self.height

    def calculate_radius(self):
        self.radius = sum([x.radius for x in self.circles]) / len(self.circles)
        return self.radius

    def calculate_center(self):
        self.x = sum([x.x for x in self.circles]) / len(self.circles)
        self.y = sum([x.y for x in self.circles]) / len(self.circles)
        return self.x, self.y

    def update_z(self):
        [self.z_points.extend(circle.z) for circle in self.circles]

    def get_support(self):
        return self.support

    def get_ratio(self):
        return self.height / self.radius

    def get_original_idxs(self):
        original_idxs = []
        [original_idxs.extend(x.original_idxs) for x in self.circles]
        return original_idxs

    def print_circles(self):
        for circle in self.circles:
            print(circle)

    def __getitem__(self, index):
        return self.circles[index]

    def __str__(self):
        string = "support: {}, height: {:.6f}, radius: {:.6f}, center: ({:.6f},{:.6f})".format(self.support, self.height, self.radius, self.x, self.y)
        return string

    def __len__(self):
        return len(self.circles)

class Circle():
    def __init__(self, iid, x, y, z, radius, slice_number=0, support=0, original_idxs=[]):
        self.id = iid
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.slice_number = slice_number
        self.support = support
        self.original_idxs = original_idxs

    def setScale(self, scale):
        self.x *= scale
        self.y *= scale
        self.z *= scale
        self.radius *= scale

    def get_support(self):
        return self.support

    def get_slice_number(self):
        return self.slice_number

    @staticmethod
    def slice_distance(c1, c2):
        return abs(c1.slice_number-c2.slice_number)

    @staticmethod
    def distance(c1, c2):
        return math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2 + (c1.slice_number-c2.slice_number)**2)

    @staticmethod
    def distance_2d(c1, c2):
        return math.sqrt((c1.x-c2.x)**2 + (c1.y-c2.y)**2)

    def area_overlap(self, circle):
        "returns from 0 to 1 how good the overlap is"
        pass

    def radius_overlap(self, circle, threshold):
        """
        returns from 0 to 1 how good the overlap is
        threshold: cutoff distance in relation to radius
        """
        delta_center = math.sqrt((self.x-circle.x)**2 + (self.y-circle.y)**2)
        overlap = min(delta_center / (threshold * self.radius), 1)
        value = Circle.normalize(0, 1, overlap)
        return 1 - value

    def least_squares(self, circ):
        dx = self.x-circ.x
        dy = self.y-circ.y
        dr = self.radius-circ.radius
        return math.sqrt(dx**2 + dy**2 + 2*dr**2)

    def score(self, circ):
        # Cutoff / Threshold / Brukerdefiner tall (3.5) in last parameter in next line and nextnext line
        score = min(self.least_squares(circ), 3)
        return 1 - Circle.normalize(0, 3, score)

    @staticmethod
    def normalize(minimum, maximum, value):
        return (value-minimum)/(maximum-minimum)

    def __str__(self):
        return "Circle: {}, slice: {}, support: {}, center: ({:.3f}, {:.3f}), radius: {:.3f}".format(self.id, self.slice_number, self.support, self.x, self.y, self.radius)

def show_circle(x_outliers, y_outliers, x_inliers, y_inliers, a, b, r):
    # show data by scatter type
    plt.scatter(x_outliers, y_outliers, c='red', marker='x', label='outliers')
    plt.scatter(x_inliers, y_inliers, c='green', marker='o', label='inliers')

    circle = plt.Circle((a, b), radius=r, color='r', fc='y', fill=False)
    plt.gca().add_patch(circle)

    plt.axis('scaled')

    # plt.tight_layout()
    plt.show()

def ransac_per_cluster(filename, n_unit_threshold):
    # Add a new dimension holding the circle numbers
    add_dimension(filename, "circle", 7, "")
    reset_dimension(filename, "circle")

    # Open the file
    las = laspy.file.File(filename, mode="rw")

    mins = las.header.min
    maxs = las.header.max
    points = las.header.records_count
    area = (maxs[0]-mins[0])*(maxs[1]-mins[1])
    
    # Threshold distance
    unit_threshold = area / points
    threshold = n_unit_threshold*unit_threshold
    print(threshold)

    slice_data = getattr(las, "slice")
    circle_data = getattr(las, "circle")
    cluster_data = getattr(las, "cluster")
    n_slices = slice_data.max()

    circle_number = 1
    #for i in range(np.amax(cluster_data)):
    for i in tqdm.tqdm(range(200, 300), total=100):
        idx = (cluster_data == i) & (circle_data == 0)
        slice_number = slice_data[idx]
        test_data_idx = (slice_data == slice_number[0])
        data = np.vstack([las.x[idx], las.y[idx]]).T
        test_data = np.vstack([las.x[test_data_idx], las.y[test_data_idx]]).T
        new_idx = getattr(las, "index")[test_data_idx]
        
        radius, center, inlier_idxs = ransac_cluster(data, test_data, "circle", 500, threshold)

        if inlier_idxs:

            print("Circle: {}".format((radius, center)))
            original_idxs = new_idx[inlier_idxs]
            print("Circle shape: {}".format(original_idxs.shape))

            circle_data[original_idxs] = circle_number

            circle_number += 1

            setattr(las, "circle", circle_data)

    las.close()

def circle(points):
    p, q, r = points[0], points[1], points[2]
    xp, yp, xq, yq, xr, yr = p[0], p[1], q[0], q[1], r[0], r[1]

    # Translate to the origin
    xq -= xp
    yq -= yp
    q2 = xp * xp + yp * yp
    xr -= xq
    yr -= yq
    r2 = xr * xr + yr * yr

    p2 = xp**2 + yp**2
    # Solve for the center coordinates
    d = 2 * (xp * yq - xq * yp)
    xc = (p2 * yq - q2 * yp) / d
    yc = (p2 * xp - q2 * xq) / d

    # Radius
    r = math.sqrt(xc * xc + yc * yc)

    # Untranslate
    xc += xp
    yc += yp

    return r, [xc, yc]



def ransac_cluster(sample_data, test_data, model, iterations, threshold):
    xAxis = np.amax(test_data[:, 0])-np.amin(test_data[:, 0])
    yAxis = np.amax(test_data[:, 1])-np.amin(test_data[:, 1])
    k = 3

    if len(sample_data) < k:
        return

    best_inliers = []
    best_model = None, [None, None]
    # Find random sample points and do ransac on them
    for it in range(iterations):
        # Randomly pick k points
        sample = sample_data[random.sample(range(sample_data.shape[0]), k), :]
        # Determine radius and center coordinates
        radius, center = circle(sample)

        # Check if radius makes sense
        """
        if radius > np.amin([xAxis, yAxis]) / 2:
            continue
        """
        if radius < 6 or radius > 8:
            continue
        # Check how many inliers we get for this circle
        inliers = []
        for idx, point in enumerate(test_data):
            if np.abs(math.sqrt(np.sum(point-center)**2) - radius) < threshold:
                inliers.append(idx)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = radius, center
    radius, center = best_model

    return radius, center, best_inliers

def cluster(filename, n_unit_threshold):
    # Add cluster dimension
    add_dimension(filename, "cluster", 3, "")

    # Open file in read-write mode
    las = laspy.file.File(filename, mode="rw")


    slice_data = getattr(las, "slice")
    cluster_data = getattr(las, "cluster")
    n_slices = slice_data.max()

    # Threshold
    mins = las.header.min
    maxs = las.header.max
    points = las.header.records_count
    area = (maxs[0]-mins[0])*(maxs[1]-mins[1])
    
    # Threshold distance
    unit_threshold = area / points
    threshold = n_unit_threshold*unit_threshold

    cluster = 0
    for i in tqdm.tqdm(reversed(range(n_slices)), total=n_slices):
        idx = (slice_data == i)
        data = np.vstack([las.x[idx], las.y[idx]]).T
        
        clustering = sklearn.cluster.DBSCAN(eps=threshold).fit(data)
        cluster_data[idx] = clustering.labels_ + cluster

        cluster += np.amax(clustering.labels_)
    
    setattr(las, "cluster", cluster_data)

    las.close()

if __name__ == "__main__":
    
    # PARAMETERS
    LAS_FILE = "files/5cyl.las"
    OUTPUT_LAS_FILE  = "files/5cyl_scaled.las"
    NUMBER_OF_RANSAC_CIRCLE_DETECTION = 5
    NUMBER_OF_SLICES = 20
    SLICE_START = 4
    UNIT_THRESHOLD_SCALE = 30
    CYLINDER_COUNT = 5
    HW_RATIO =  [4.2, 6.2] #[4.5, 6.5] (4.5-6.5 for larger cylinder (model 1,2,3,4))
    TRUE_RADIUS =  2.93 #4 (4 for larger cylinder (model 1,2,3,4)) 
    
    # Step 1
    las = LAS(LAS_FILE)
    las.index_to_new_dimension()
    las.slice_to_new_dimension("z", NUMBER_OF_SLICES)
    circles = las.ransac_new_dimension(UNIT_THRESHOLD_SCALE, SLICE_START, NUMBER_OF_RANSAC_CIRCLE_DETECTION)
    pickle_data("circle.data", circles)
    
    circles = unpickle_data("circle.data")
    cyls = Cylinders(support_threshold=250).create_cylinders(circles).remove_breaches(0.5)
    cyls.remove_overlapping_cylinders()
    cyls.keep_n_best_cylinders(CYLINDER_COUNT)
    las.set_cylinders(cyls, radius_multiplier=1.2)
    

    
    # Step 2
    shutil.copy2(LAS_FILE, OUTPUT_LAS_FILE)

    las2 = LAS(OUTPUT_LAS_FILE)
    
    circle_number_start = 1
    for cylinder_value in range(1, CYLINDER_COUNT+1):
        las2.reset_dimension("cylinder")
        circles, circle_number_start = las2.ransac_2_new_dimension(UNIT_THRESHOLD_SCALE, cylinder_value, circle_number_start)
        pickle_data("circles_cyl{}.data".format(cylinder_value), circles)

    circles = []
    for cylinders in range(1, CYLINDER_COUNT+1):
        circles.extend(unpickle_data("circles_cyl{}.data".format(cylinders)))
    print("Number of circles: ", len(circles))

    cylinders = Cylinders(support_threshold=250).create_cylinders(circles).remove_breaches(0.85)
    print(cylinders)
    
    
    cylinders.keep_n_best_cylinders(CYLINDER_COUNT)

    #Scaling

    las2.set_cylinders_by_circles(cylinders)

    scaling = cylinders.getScale(TRUE_RADIUS, hw_ratio=HW_RATIO)
    print("Scaling factor: ", scaling)
    las2.scale(scaling)
    cylinders.setScale(scaling)
    print(cylinders)
"""
    x1, y1 = las2.getXYPoint(3616567)
    x2, y2 = las2.getXYPoint(3604702)
    print("20cm in measured value is now: {}".format(math.sqrt((x1-x2)**2+(y1-y2)**2)))
    # scaling = 0.5662502993091958
    
    """