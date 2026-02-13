import numpy as np
import geopandas as gpd

from shapely import make_valid
from shapely.geometry import Point, LineString


class GeometricalFeatures:
    def __init__(self, poly):
        self.poly = poly

    def basic(self):
        area = self.poly.area
        perimeter = self.poly.length
        return area, perimeter

    def edge_lengths(self):
        bounding_box = self.poly.minimum_rotated_rectangle
        xs, ys = bounding_box.exterior.coords.xy
        length1 = Point(xs[0], ys[0]).distance(Point(xs[1], ys[1]))
        length2 = Point(xs[1], ys[1]).distance(Point(xs[2], ys[2]))
        edge_lengths = (length1, length2)

        short_side = min(edge_lengths)
        long_side = max(edge_lengths)
        return short_side, long_side

    def rotated_bounding_box_properties(self):
        bounding_box = self.poly.minimum_rotated_rectangle
        area_bb = bounding_box.area
        perimeter_bb = bounding_box.length
        return area_bb, perimeter_bb

    def max_diameter_polygon(self):
        points_list = list(self.poly.exterior.coords)
        points_series = gpd.GeoSeries(Point(coords) for coords in points_list)
        max_diameter = 0

        for point in points_series:
            max_distance_point = points_series.distance(point).max()
            if max_distance_point > max_diameter:
                max_diameter = max_distance_point

        return max_diameter

    def max_diameter_features(self):
        max_diameter = self.max_diameter_polygon()
        max_diameter_per_area = max_diameter / self.poly.area
        # ideja Tool Polygon Shape Indices SAGA-GIS
        bb_area = self.poly.minimum_rotated_rectangle.area
        max_diameter_per_bb_area = max_diameter / bb_area
        return max_diameter, max_diameter_per_area, max_diameter_per_bb_area

    def convex_hull_features(self):
        ch_area = self.poly.convex_hull.area
        ch_perimeter = self.poly.convex_hull.length
        return ch_area, ch_perimeter

    def shape_complexity_index_feature(self):
        # shape complexity index whitebox tools
        # SCI = 1 - A / Ah (A is the polygon's area and Ah is the area of the convex hull containing the polygon.abs
        ch_area = self.poly.convex_hull.area
        sci = 1 - (self.poly.area / ch_area)
        return sci

    def perimeter_area_ratio(self):
        perimeter_area_ratio = self.poly.length / self.poly.area
        return perimeter_area_ratio

    def compactness_ratio(self):
        circle_area = (self.poly.length ** 2) / (4 * np.pi)
        ratio = np.sqrt(self.poly.area / circle_area)
        return ratio

    def elongation(self):
        short_side, long_side = self.edge_lengths()
        elongation = long_side / short_side
        return elongation


class NeighbourhoodFeatures:
    def __init__(self, poly, all_polys, distance_to_nonexistant=200):
        self.poly = poly
        self.all_polys = all_polys
        self.other_polys = self.remove_poly
        self.distance_to_nonexistant = distance_to_nonexistant

    @property
    def remove_poly(self):
        all_polys = self.all_polys.to_list()
        if self.poly in all_polys:
            all_polys.remove(self.poly)
        all_polys = gpd.GeoSeries(all_polys)
        return all_polys

    def num_neighbours(self, neighbourhood_distance=10):
        neighbourhood = self.poly.buffer(neighbourhood_distance)
        if not neighbourhood.is_valid:
            neighbourhood = make_valid(neighbourhood)
        neighbours = self.other_polys.clip(neighbourhood)
        num_neighbours = len(neighbours)
        return num_neighbours

    def neighbours_per_area(self, neighbourhood_distance=10):
        num_neighbours = self.num_neighbours(neighbourhood_distance=neighbourhood_distance)
        if num_neighbours == 0:
            neighbours_per_area = 0
        else:
            neighbours_per_area = self.poly.area / num_neighbours
        if neighbours_per_area == np.inf:
            neighbours_per_area = 0
        return neighbours_per_area

    def closest_distances(self):
        other_polys = self.other_polys.to_list().copy()
        if len(other_polys) == 0:
            dist1, dist2, dist3, dist4 = (self.distance_to_nonexistant, self.distance_to_nonexistant,
                                          self.distance_to_nonexistant, self.distance_to_nonexistant)
        else:
            distances_neighbours = []
            for i in range(4):
                if len(other_polys) == 0:
                    distances_neighbours.append(self.distance_to_nonexistant)

                else:
                    distances = self.poly.distance(other_polys)
                    distances = np.array(distances)
                    distance_min = np.min(distances)
                    distances_neighbours.append(distance_min)

                    closest_poly_index = np.nonzero(distances == distance_min)[0][0]
                    closest_poly = other_polys[closest_poly_index]
                    other_polys.remove(closest_poly)
            dist1, dist2, dist3, dist4 = distances_neighbours
        return dist1, dist2, dist3, dist4

    def bb_vector(self, poly=None):
        if poly is None:
            poly = self.poly
        xs, ys = poly.minimum_rotated_rectangle.exterior.coords.xy
        x1, x2, x3, x4, _ = xs
        y1, y2, y3, y4, _ = ys

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([x3, y3])

        edge_length = [Point(x1, y1).distance(Point(x2, y2)), Point(x2, y2).distance(Point(x3, y3))]
        edge_length = np.array(edge_length)
        long_side = np.max(edge_length)
        index_long_side = np.nonzero(edge_length == long_side)[0][0]

        if index_long_side == 0:
            long_side_vector = p2 - p1
        else:
            long_side_vector = p3 - p2

        return long_side_vector

    @staticmethod
    def angle_between_vectors(vector_left, vector_right):
        dot_product = np.dot(vector_left, vector_right)
        magnitude_1 = np.linalg.norm(vector_left)
        magnitude_2 = np.linalg.norm(vector_right)

        cosinus = dot_product / (magnitude_1 * magnitude_2)
        cosinus = np.minimum(1, cosinus)
        cosinus = np.maximum(-1, cosinus)
        angle = np.arccos(cosinus)
        angle = np.rad2deg(angle)
        if angle >= 90:
            angle = angle - 180
        angle = np.absolute(angle)
        return angle

    @staticmethod
    def line_vector_from_polys(poly_left, poly_right, extend_line_factor=1.0):
        # create line from poly centroid and closest poly centroid
        left_centroid = poly_left.minimum_rotated_rectangle.centroid
        left_coords = [p for p in left_centroid.coords][0]
        left_coords = np.array(left_coords)

        right_centroid = poly_right.minimum_rotated_rectangle.centroid
        right_coords = [p for p in right_centroid.coords][0]
        right_coords = np.array(right_coords)

        line_vector = left_coords - right_coords
        line_vector = extend_line_factor * line_vector
        return line_vector

    def closest_poly_features(self):
        other_polys = self.other_polys.to_list().copy()
        distances = self.poly.distance(other_polys)
        distances = np.array(distances)
        dist1, dist2, _, _ = self.closest_distances()

        if dist1 == self.distance_to_nonexistant:
            angle = 90
            line_length, min_dist_to_line, min_dist_centroid_to_line = (self.distance_to_nonexistant,
                                                                        self.distance_to_nonexistant,
                                                                        self.distance_to_nonexistant)

        else:
            # find closest poly
            closest_poly_index = np.nonzero(distances == dist1)[0][0]
            closest_poly = other_polys[closest_poly_index]
            other_polys.remove(closest_poly)

            # calculate angle between vectors of bounding boxes of poly and closest poly
            vector_1 = self.bb_vector()
            vector_2 = self.bb_vector(closest_poly)
            angle = self.angle_between_vectors(vector_1, vector_2)

            # create line from poly centroid and closest poly centroid
            poly_centroid = self.poly.minimum_rotated_rectangle.centroid
            poly_coords = [p for p in poly_centroid.coords][0]
            poly_coords = np.array(poly_coords)

            line_vector = self.line_vector_from_polys(self.poly, closest_poly, extend_line_factor=2.5)
            coords_one = poly_coords + line_vector
            coords_two = poly_coords - line_vector

            line = LineString([coords_one, coords_two])
            line_length = line.length

            if dist2 == self.distance_to_nonexistant:
                min_dist_to_line = self.distance_to_nonexistant
                min_dist_centroid_to_line = self.distance_to_nonexistant
            else:
                # calculate features from distances of other polygons to created line
                distances_to_line = line.distance(other_polys)
                distances_to_line = np.array(distances_to_line)
                min_dist_to_line = np.min(distances_to_line)

                closest_to_line_index = np.nonzero(distances_to_line == min_dist_to_line)[0][0]
                closest_to_line = other_polys[closest_to_line_index]
                closest_to_line_centroid = closest_to_line.minimum_rotated_rectangle.centroid
                min_dist_centroid_to_line = closest_to_line_centroid.distance(line)

        return angle, line_length, min_dist_to_line, min_dist_centroid_to_line

    def recognise_grid(self):
        other_polys = self.other_polys.to_list().copy()
        distances = self.poly.distance(other_polys)
        distances = np.array(distances)
        dist1, dist2, dist3, dist4 = self.closest_distances()

        if (np.array([dist1, dist2, dist3, dist4]) == self.distance_to_nonexistant).any():
            angle1, angle2, angle3, angle4, angle5, angle6 = (90, 90, 90, 90, 90, 90)

        else:
            poly1_index = np.nonzero(distances == dist1)[0][0]
            poly1 = other_polys[poly1_index]
            poly2_index = np.nonzero(distances == dist2)[0][0]
            poly2 = other_polys[poly2_index]
            poly3_index = np.nonzero(distances == dist3)[0][0]
            poly3 = other_polys[poly3_index]
            poly4_index = np.nonzero(distances == dist4)[0][0]
            poly4 = other_polys[poly4_index]

            vector1 = self.line_vector_from_polys(self.poly, poly1)
            vector2 = self.line_vector_from_polys(self.poly, poly2)
            vector3 = self.line_vector_from_polys(self.poly, poly3)
            vector4 = self.line_vector_from_polys(self.poly, poly4)

            angle1 = self.angle_between_vectors(vector1, vector2)
            angle2 = self.angle_between_vectors(vector1, vector3)
            angle3 = self.angle_between_vectors(vector1, vector4)
            angle4 = self.angle_between_vectors(vector2, vector3)
            angle5 = self.angle_between_vectors(vector2, vector4)
            angle6 = self.angle_between_vectors(vector3, vector4)

        return angle1, angle2, angle3, angle4, angle5, angle6

    def distance_to_water(self, water_gs):
        distances = self.poly.distance(water_gs)
        distance_water = distances.min()
        return distance_water
