import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask

from separate_narrow import narrow_wide_geometries
from object_features import GeometricalFeatures, NeighbourhoodFeatures


class ClassificationPreparation:
    def __init__(self, gdf, raster_path, water_gdf, min_area=5, vector_features=True, raster_features=False):
        self.gdf = self.prepare_input(gdf, min_area=min_area)
        self.water = water_gdf
        self.raster_path = raster_path
        self.vector_features = vector_features
        self.raster_features = raster_features

    @property
    def features(self):
        if self.vector_features:
            vf_cols = ['area', 'perimeter',
                       'bb_short_side_len', 'bb_long_side_len', 'bb_area', 'bb_perimeter',
                       'max_diameter', 'max_diameter_per_area', 'max_diameter_per_bb_area',
                       'ch_area', 'ch_perimeter',
                       'perimeter_area_ratio', 'elongation', 'compactness_ratio', 'shape_complexity_index',
                       'num_neighbours', 'neighbours_per_area',
                       'dist_first_closest', 'dist_second_closest', 'dist_third_closest', 'dist_fourth_closest',
                       'closest_angle', 'closest_line_lenght', 'closest_min_dist_line', 'closest_min_dist_centr_line',
                       'grid_angle_1', 'grid_angle_2', 'grid_angle_3', 'grid_angle_4', 'grid_angle_5', 'grid_angle_6',
                       'dist_water']
        else:
            vf_cols = []

        if self.raster_features:
            with rasterio.open(self.raster_path) as src:
                channel_num = src.count

            rf_cols = []
            for stat in ['mean', 'median', 'std', 'max', 'min']:
                for channel in range(1, channel_num + 1):
                    col_name = stat + '_channel' + str(channel)
                    rf_cols.append(col_name)
        else:
            rf_cols = []

        cols = [vf_cols, rf_cols]
        cols = [item for row in cols for item in row]
        return cols

    @staticmethod
    def prepare_input(gdf, min_area=5):
        # prepare geometry
        gdf = gdf.simplify(tolerance=0.5)
        gdf = narrow_wide_geometries(gdf)
        crs = gdf.crs
        gs = gdf.geometry.make_valid()
        gs = gs.explode(index_parts=False)
        gs = gs[gs.geometry.geom_type == 'Polygon']
        gs = gs[gs.area >= min_area]
        gs = gs.reset_index(drop=True)
        gdf = gpd.GeoDataFrame(geometry=gs, crs=crs)
        # gdf.to_file('gdf_non_valid.gpkg')

        return gdf

    def calculate_poly_features(self, poly):
        features = []

        geom_features = GeometricalFeatures(poly)
        features.append(geom_features.basic())
        features.append(geom_features.edge_lengths())
        features.append(geom_features.rotated_bounding_box_properties())
        features.append(geom_features.max_diameter_features())
        features.append(geom_features.convex_hull_features())
        features.append(geom_features.perimeter_area_ratio())
        features.append(geom_features.elongation())
        features.append(geom_features.compactness_ratio())
        features.append(geom_features.shape_complexity_index_feature())

        neighbour_features = NeighbourhoodFeatures(poly, self.gdf.geometry)
        features.append(neighbour_features.num_neighbours())
        features.append(neighbour_features.neighbours_per_area())
        features.append(neighbour_features.closest_distances())
        features.append(neighbour_features.closest_poly_features())
        features.append(neighbour_features.recognise_grid())
        features.append(neighbour_features.distance_to_water(self.water.geometry))

        features = [list(f) if type(f) == tuple else [f] for f in features]
        features = [item for row in features for item in row]
        features = np.array(features)
        return features

    def calculate_raster_features(self, poly):
        with rasterio.open(self.raster_path) as src:
            array, _ = rasterio.mask.mask(src, [poly], crop=True, filled=False)

        axis = (1, 2)
        means = array.mean(axis=axis).data
        medians = np.ma.median(array, axis=axis).data
        stds = array.std(axis=axis).data
        maxs = array.max(axis=axis).data
        mins = array.min(axis=axis).data
        features = np.concatenate([means, medians, stds, maxs, mins])
        return features

    def gdf_features(self):
        if self.vector_features:
            vf = self.gdf.geometry.map(self.calculate_poly_features)
            vf = pd.DataFrame.from_records(vf)
            vf = vf.to_numpy()
            """
            vf = [self.calculate_poly_features(poly) for poly in self.gdf.geometry]
            vf = np.array(vf)
            """
        else:
            vf = None

        if self.raster_features:
            rf = self.gdf.geometry.map(self.calculate_raster_features)
            rf = pd.DataFrame.from_records(rf)
            rf = rf.to_numpy()
            """
            rf = [self.calculate_raster_features(poly) for poly in self.gdf.geometry]
            rf = np.array(rf)
            """
        else:
            rf = None

        if vf is None:
            features = rf
        elif rf is None:
            features = vf
        else:
            features = np.concatenate([vf, rf], axis=1)
        return features

    @staticmethod
    def poly_ground_truth(poly, gt_gdf):
        poly_gdf = gt_gdf.clip(poly)
        intersection_area = poly_gdf.area

        if intersection_area.sum() < (poly.area / 2):
            gt_class = 0
        else:
            max_class = intersection_area == intersection_area.max()
            max_class = poly_gdf[max_class]
            gt_class = max_class.index[0]
        return gt_class

    def ground_truth(self, gt_gdf):
        gt_gdf = gt_gdf.dissolve(by='phd_class')
        gt_classes = self.gdf.geometry.map(lambda poly: self.poly_ground_truth(poly, gt_gdf))
        # gt_classes = [self.poly_ground_truth(poly, gt_gdf) for poly in self.gdf.geometry]
        gt_classes = np.array(gt_classes)
        return gt_classes

    def prepare_classification(self):
        data = self.gdf_features()
        geometry = self.gdf.geometry
        crs = self.gdf.crs
        columns = self.features

        gdf = gpd.GeoDataFrame(data=data, geometry=geometry, columns=columns, crs=crs)
        return gdf
